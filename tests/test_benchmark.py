import pytest
from unittest.mock import MagicMock, patch
from rag_benchmark_lab.benchmark import RAGBenchmarkRunner

@pytest.fixture
def mock_runner():
    """Fixture to mock MLflow interactions and sub-component dependencies."""
    with patch("llmops_common.logging.mlflow_logger.MLflowLogger.__init__", return_value=None), \
         patch("llmops_common.eval.evaluator.LLMEvaluator.__init__", return_value=None), \
         patch("llmops_common.client.ollama_client.OllamaClient.__init__", return_value=None):
        
        runner = RAGBenchmarkRunner(experiment_name="test_exp")
        runner.logger = MagicMock()
        runner.evaluator = MagicMock()
        runner.ollama_client = MagicMock()
        return runner

@patch("rag_benchmark_lab.benchmark.RAGPipeline")
@patch("mlflow.log_metric")
def test_run_grid_benchmark_success(mock_log_metric, mock_pipeline, mock_runner):
    """Verify that grid search executes completely and computes metrics accurately."""
    # Setup internal pipeline mocks
    pipeline_instance = MagicMock()
    pipeline_instance.ingest_documents.return_value = 5
    pipeline_instance.answer_query.return_value = {
        "context": "MLOps ensures continuous deployment.",
        "query": "What is MLOps?",
        "answer": "MLOps handles deployment."
    }
    mock_pipeline.return_value = pipeline_instance

    # Setup evaluator return metrics
    mock_runner.evaluator.evaluate_faithfulness.return_value = 1.0
    mock_runner.evaluator.evaluate_relevance.return_value = 0.8

    # Inputs
    sample_text = "MLOps context data snippet."
    queries = ["What is MLOps?"]
    
    # Run a single combinatorics loop
    summaries = mock_runner.run_grid_benchmark(
        raw_text=sample_text,
        test_queries=queries,
        chunk_sizes=[200],
        chunk_overlaps=[20],
        models=["phi3:latest"]
    )

    # Assertions
    assert len(summaries) == 1
    assert summaries[0]["chunk_size"] == 200
    assert summaries[0]["avg_faithfulness"] == 1.0
    assert summaries[0]["avg_relevance"] == 0.8
    
    # Verify trace control boundaries were called cleanly
    mock_runner.logger.start_trace.assert_called_once()
    mock_runner.logger.end_trace.assert_called_once()