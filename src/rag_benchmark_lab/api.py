import os
import re
import uuid
from itertools import combinations
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llmops_common.observability.instrumentation import instrument_app
from llmops_common.stats.significance import mcnemar_test
from pydantic import BaseModel, Field

from rag_benchmark_lab.benchmark import RAGBenchmarkRunner
from rag_benchmark_lab.document_extraction import extract_text
from rag_benchmark_lab.pipeline import RAGPipeline

app = FastAPI(
    title="RAG Benchmark Lab API",
    description="Standalone API for evaluating RAG pipeline configurations.",
    version="0.1.0",
)
instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BenchmarkRequest(BaseModel):
    experiment_name: str = "rag_standalone_benchmark"
    raw_text: str
    test_queries: list[str]
    chunk_sizes: list[int]
    chunk_overlaps: list[int]
    models: list[str]


class BatchDatasetItem(BaseModel):
    query: str
    expected_output: str


class BatchEvalRequest(BaseModel):
    experiment_name: str = "golden_dataset_eval"
    raw_text: str
    dataset: list[BatchDatasetItem]
    models: list[str] = Field(min_length=1)
    chunk_size: int = 300
    chunk_overlap: int = 30


@app.get("/health", summary="Health Check")
def health_check():
    """Confirms the laboratory API is up and running."""
    return {"status": "healthy", "service": "rag-benchmark-lab"}


@app.post("/extract-document", summary="Extract raw text from an uploaded document")
async def extract_document(
    file: UploadFile = File(...),  # noqa: B008
) -> dict[str, Any]:
    """Extracts plain text from an uploaded .txt/.md/.pdf file so it can be
    used as `raw_text` for /benchmark or /batch-evaluate, instead of
    requiring the knowledge base to be hand-typed into a textarea."""
    content = await file.read()
    try:
        text = extract_text(filename=file.filename or "", content=content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "filename": file.filename,
        "characters": len(text),
        "text": text,
    }


@app.post("/benchmark", summary="Run RAG Grid Benchmark")
def run_benchmark(request: BenchmarkRequest) -> dict[str, Any]:
    """
    Executes the grid search over chunking strategies and models,
    evaluating Faithfulness and Relevance via the LLM-as-a-judge.
    """
    try:
        runner = RAGBenchmarkRunner(experiment_name=request.experiment_name)

        results = runner.run_grid_benchmark(
            raw_text=request.raw_text,
            test_queries=request.test_queries,
            chunk_sizes=request.chunk_sizes,
            chunk_overlaps=request.chunk_overlaps,
            models=request.models,
        )

        return {
            "status": "success",
            "experiment_name": request.experiment_name,
            "results": results,
        }
    # Any failure in the grid search (LLM error, chroma error, etc.) should
    # be a clean 500, not an unhandled exception crashing the request.
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Benchmark execution failed: {e!s}"
        )


def _run_single_model_batch(
    pipeline: RAGPipeline, model: str, dataset: list[BatchDatasetItem]
) -> dict[str, Any]:
    """
    Runs the full golden dataset against one model and strictly calculates
    'Format Penalty' based on Finwise Neuro-Symbolic Label Contracts.
    """
    results = []
    format_violations = 0
    total_items = len(dataset)

    for item in dataset:
        output = pipeline.answer_query(query=item.query, model_name=model)
        actual_answer = output["answer"]

        # --- FINWISE STRICT FORMAT ADHERENCE (LABEL CONTRACT) ---
        is_format_valid = True
        expected = item.expected_output.upper()

        # Test A: Canonical Label Check (BULLISH, BEARISH, NEUTRAL)
        if expected in ["BULLISH", "BEARISH", "NEUTRAL"]:
            if expected not in actual_answer.upper():
                is_format_valid = False

        # Test B: Neuro-Symbolic Token Match (e.g., P_8_V_9) -- ensure the
        # exact token format is present in the LLM's raw generation
        elif re.search(r"P_[0-9]_V_[0-9]", expected) and expected not in actual_answer:
            is_format_valid = False

        if not is_format_valid:
            format_violations += 1

        results.append(
            {
                "query": item.query,
                "expected": item.expected_output,
                "actual": actual_answer,
                "format_valid": is_format_valid,
            }
        )

    format_penalty_score = (
        (format_violations / total_items) * 100 if total_items > 0 else 0.0
    )
    accuracy = 100.0 - format_penalty_score

    return {
        "metrics": {
            "total_tested": total_items,
            "format_violations": format_violations,
            "strict_accuracy_percentage": accuracy,
            "format_penalty_percentage": format_penalty_score,
        },
        "detailed_results": results,
    }


@app.post("/batch-evaluate", summary="Run Golden Dataset Regression Test")
def run_batch_evaluation(request: BatchEvalRequest) -> dict[str, Any]:
    """
    Executes a batch of queries against the RAG pipeline for one or more
    models (all against the same ingested knowledge base, so results are
    paired item-for-item across models), then reports a McNemar comparison
    between every pair of models on their format-compliance results.
    """
    try:
        # 1. Initialize isolated collection for this specific batch run
        unique_collection = f"batch_{uuid.uuid4().hex[:8]}"
        pipeline = RAGPipeline(collection_name=unique_collection)

        # 2. Ingest the golden knowledge base once, shared across all models
        pipeline.ingest_documents(
            raw_text=request.raw_text,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # 3. Run the full dataset against each model
        results_by_model: dict[str, Any] = {}
        format_valid_by_model: dict[str, list[bool]] = {}
        for model in request.models:
            model_result = _run_single_model_batch(pipeline, model, request.dataset)
            results_by_model[model] = model_result
            format_valid_by_model[model] = [
                item["format_valid"] for item in model_result["detailed_results"]
            ]

        # 4. Paired McNemar comparison between every pair of models, on the
        # same (item-aligned) format-compliance results -- an
        # independent-samples test would be wrong here since every model
        # answers the exact same dataset.
        comparisons = []
        for model_a, model_b in combinations(request.models, 2):
            result = mcnemar_test(
                format_valid_by_model[model_a], format_valid_by_model[model_b]
            )
            comparisons.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "method": result.method,
                }
            )

        return {
            "status": "success",
            "results_by_model": results_by_model,
            "comparisons": comparisons,
        }

    # Any failure in the batch run (LLM error, chroma error, etc.) should be
    # a clean 500, not an unhandled exception crashing the request.
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {e!s}")
