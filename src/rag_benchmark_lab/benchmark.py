import subprocess
import uuid
from typing import Any

import mlflow  # EKSİK IMPORT EKLENDİ
from llmops_common.client.ollama_client import OllamaClient
from llmops_common.eval.evaluator import LLMEvaluator
from llmops_common.logging.mlflow_logger import MLflowLogger
from llmops_common.utils.timer import timing_context

from rag_benchmark_lab.pipeline import RAGPipeline


class RAGBenchmarkRunner:
    """
    Automates the evaluation of various RAG configurations.
    Computes average latency, faithfulness, and relevance, then logs them to MLflow.
    """

    def __init__(self, experiment_name: str = "RAG_Optimization_Lab"):
        self.logger = MLflowLogger(experiment_name=experiment_name)
        self.ollama_client = OllamaClient()
        self.evaluator = LLMEvaluator(client=self.ollama_client)

    def get_vram_usage_mb(self) -> float:
        """It turns instant VRAM usage on NVIDIA GPU to MB."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return float(result.stdout.strip().split("\n")[0])
        # No GPU / nvidia-smi not installed -- VRAM tracking is best-effort,
        # not required for the benchmark to run.
        except Exception:  # noqa: BLE001
            return 0.0

    def run_grid_benchmark(
        self,
        raw_text: str,
        test_queries: list[str],
        chunk_sizes: list[int],
        chunk_overlaps: list[int],
        models: list[str],
    ) -> list[dict[str, Any]]:
        results_summary = []

        for model_name in models:
            try:
                for chunk_size in chunk_sizes:
                    for chunk_overlap in chunk_overlaps:

                        # 1. ETL & Data Prep
                        unique_collection = f"test_coll_{uuid.uuid4().hex[:8]}"
                        pipeline = RAGPipeline(collection_name=unique_collection)

                        pipeline.ingest_documents(
                            raw_text=raw_text,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )

                        total_latency = 0.0
                        total_faithfulness = 0.0
                        total_relevance = 0.0
                        total_tps = 0.0
                        total_ttft = 0.0
                        query_count = len(test_queries)

                        # 2. Inference & Evaluation
                        run_name = f"chunk_{chunk_size}_overlap_{chunk_overlap}_{model_name.split(':')[0]}"
                        self.logger.start_trace(run_name=run_name)

                        for query in test_queries:
                            with timing_context() as timer:
                                # Ensure pipeline returns Ollama metadata alongside answer
                                output = pipeline.answer_query(
                                    query=query, model_name=model_name
                                )

                            latency = timer["elapsed"]
                            total_latency += latency

                            # Safely extract hardware telemetry if provided by Ollama/Pipeline
                            meta = output.get("meta", {})
                            eval_count = meta.get("eval_count", 0)
                            eval_duration_ns = meta.get("eval_duration", 0)
                            prompt_eval_ns = meta.get("prompt_eval_duration", 0)

                            # Calculate TPS (Tokens per second) and TTFT (ms)
                            tps = (
                                (eval_count / (eval_duration_ns / 1e9))
                                if eval_duration_ns > 0
                                else 0.0
                            )
                            ttft_ms = prompt_eval_ns / 1e6

                            total_tps += tps
                            total_ttft += ttft_ms

                            faithfulness = self.evaluator.evaluate_faithfulness(
                                context=output["context"], response=output["answer"]
                            )
                            relevance = self.evaluator.evaluate_relevance(
                                query=output["query"], response=output["answer"]
                            )

                            total_faithfulness += faithfulness
                            total_relevance += relevance

                        avg_latency = total_latency / query_count
                        avg_faithfulness = total_faithfulness / query_count
                        avg_relevance = total_relevance / query_count
                        avg_tps = total_tps / query_count
                        avg_ttft = total_ttft / query_count

                        vram_used = self.get_vram_usage_mb()

                        # --- MLFLOW LOGGING BLOCK ---
                        mlflow.log_metric("avg_latency_seconds", avg_latency)
                        mlflow.log_metric("avg_faithfulness_score", avg_faithfulness)
                        mlflow.log_metric("avg_relevance_score", avg_relevance)
                        mlflow.log_metric("avg_tps", avg_tps)
                        mlflow.log_metric("avg_ttft_ms", avg_ttft)
                        mlflow.log_metric("vram_used_mb", vram_used)

                        self.logger.end_trace()

                        results_summary.append(
                            {
                                "run_name": run_name,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "model": model_name,
                                "avg_latency": avg_latency,
                                "avg_faithfulness": avg_faithfulness,
                                "avg_relevance": avg_relevance,
                                "vram_mb": vram_used,
                                # NOTE: previously omitted -> every downstream consumer
                                # (e.g. FaithfulnessRelevanceScorerNode in llmops-studio) always
                                # received empty context/response strings. We surface the last
                                # query's context/answer from this run so the DAG output contract
                                # (context/response) is actually satisfied.
                                "context": output.get("context", ""),
                                "response": output.get("answer", ""),
                            }
                        )

            # OOM yediğinde veya model çöktüğünde kodu durdurma, kaydet ve diğer modele geç
            except Exception as e:  # noqa: BLE001
                print(
                    f"❌ [Hardware Limit Reached] Pipeline failed for {model_name}: {e}"
                )

            finally:
                # KRİTİK NOKTA: İşlem başarılı olsa da, çökse de bu modelin VRAM'ini ZORLA BOŞALT
                print(
                    f"[HARDWARE OPTIMIZER] Releasing VRAM layers for model: {model_name}"
                )
                self.ollama_client.unload_model(model_name=model_name)

        return results_summary
