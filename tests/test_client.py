from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag_benchmark_lab.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "rag-benchmark-lab"}


def test_extract_document_success():
    response = client.post(
        "/extract-document",
        files={"file": ("notes.txt", b"Hello world.", "text/plain")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "notes.txt"
    assert data["text"] == "Hello world."
    assert data["characters"] == len("Hello world.")


def test_extract_document_unsupported_type_returns_422():
    response = client.post(
        "/extract-document",
        files={"file": ("image.png", b"\x89PNG\r\n", "image/png")},
    )

    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


@patch("rag_benchmark_lab.api.RAGBenchmarkRunner")
def test_benchmark_success(mock_runner_cls):
    mock_runner = MagicMock()
    mock_runner.run_grid_benchmark.return_value = [
        {"chunk_size": 200, "avg_faithfulness": 1.0, "avg_relevance": 0.8}
    ]
    mock_runner_cls.return_value = mock_runner

    response = client.post(
        "/benchmark",
        json={
            "experiment_name": "test_exp",
            "raw_text": "some context",
            "test_queries": ["What is MLOps?"],
            "chunk_sizes": [200],
            "chunk_overlaps": [20],
            "models": ["phi3:latest"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["results"][0]["avg_faithfulness"] == 1.0
    mock_runner_cls.assert_called_once_with(experiment_name="test_exp")


@patch("rag_benchmark_lab.api.RAGBenchmarkRunner")
def test_benchmark_pipeline_failure_returns_500(mock_runner_cls):
    mock_runner = MagicMock()
    mock_runner.run_grid_benchmark.side_effect = RuntimeError(
        "ollama connection refused"
    )
    mock_runner_cls.return_value = mock_runner

    response = client.post(
        "/benchmark",
        json={
            "raw_text": "ctx",
            "test_queries": ["q"],
            "chunk_sizes": [200],
            "chunk_overlaps": [20],
            "models": ["phi3:latest"],
        },
    )

    assert response.status_code == 500
    assert "ollama connection refused" in response.json()["detail"]


@patch("rag_benchmark_lab.api.RAGPipeline")
def test_batch_evaluate_computes_format_penalty(mock_pipeline_cls):
    mock_pipeline = MagicMock()
    mock_pipeline.answer_query.side_effect = [
        {"answer": "The signal is BULLISH based on the data."},
        {"answer": "I'm not sure, maybe positive?"},
    ]
    mock_pipeline_cls.return_value = mock_pipeline

    response = client.post(
        "/batch-evaluate",
        json={
            "raw_text": "some knowledge base",
            "dataset": [
                {"query": "q1", "expected_output": "BULLISH"},
                {"query": "q2", "expected_output": "BEARISH"},
            ],
            "model": "phi3:latest",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["metrics"]["total_tested"] == 2
    assert data["metrics"]["format_violations"] == 1
    assert data["metrics"]["strict_accuracy_percentage"] == 50.0


def test_metrics_endpoint_exposes_prometheus_data():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content
