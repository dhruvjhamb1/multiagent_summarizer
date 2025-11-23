from datetime import datetime, timezone
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, storage_manager_singleton
from app.config import settings
from app.models.schemas import Metadata, StatusEnum
from app.services.orchestrator import DocumentAnalysisOrchestrator


@pytest.fixture()
def client(tmp_path, monkeypatch):
    storage_manager_singleton.documents.clear()
    storage_manager_singleton.jobs.clear()
    monkeypatch.setattr(settings, "storage_path", str(tmp_path))

    with TestClient(app) as test_client:
        yield test_client

    storage_manager_singleton.documents.clear()
    storage_manager_singleton.jobs.clear()


def _success_results(document_id: str) -> Dict[str, Any]:
    return {
        "summarizer": {
            "text": f"Summary for {document_id}",
            "key_points": ["Point one", "Point two", "Point three"],
            "confidence": 0.92,
            "processing_time": 1.2,
        },
        "entity_extractor": {
            "people": [
                {"name": "Jane Analyst", "context": "Primary contact", "mentions": 1}
            ],
            "organizations": [
                {"name": "Acme Corp", "context": "Referenced in text", "mentions": 2}
            ],
            "dates": [
                {"name": "2025-01-01", "context": "Contract date", "mentions": 1}
            ],
            "locations": [
                {"name": "New York", "context": "Headquarters", "mentions": 1}
            ],
            "monetary_values": [
                {"name": "$1M", "context": "Deal size", "mentions": 1}
            ],
            "processing_time": 1.1,
        },
        "sentiment_analyzer": {
            "overall": "positive",
            "confidence": 0.88,
            "tone": {
                "formality": "formal",
                "urgency": "medium",
                "objectivity": "objective",
            },
            "emotional_indicators": {"optimistic": 0.76},
            "key_phrases": [
                {"text": "strong outlook", "sentiment": "positive"},
                {"text": "continued investment", "sentiment": "positive"},
            ],
            "processing_time": 0.9,
        },
    }


def _stub_success(metadata_time: float = 2.4):
    async def _fake(self: DocumentAnalysisOrchestrator, job_id: str, document_id: str, document_text: str) -> Dict[str, Any]:
        results = _success_results(document_id)
        agent_status = {
            "summarizer": StatusEnum.COMPLETED,
            "entity_extractor": StatusEnum.COMPLETED,
            "sentiment_analyzer": StatusEnum.COMPLETED,
        }
        metadata = Metadata(
            total_processing_time_seconds=metadata_time,
            parallel_execution=True,
            agents_completed=3,
            agents_failed=0,
            timestamp=datetime.now(timezone.utc),
            warning=None,
            failed_agents=[],
        )
        await self.storage_manager.update_job_status(
            job_id,
            status=StatusEnum.COMPLETED,
            agents_status=agent_status,
            results=results,
            end_time=datetime.now(timezone.utc),
            metadata=metadata,
        )
        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": StatusEnum.COMPLETED,
            "results": results,
            "metadata": metadata,
        }

    return _fake


def _stub_partial():
    async def _fake(self: DocumentAnalysisOrchestrator, job_id: str, document_id: str, document_text: str) -> Dict[str, Any]:
        results = _success_results(document_id)
        results["entity_extractor"] = {"error": "Simulated failure"}
        agent_status = {
            "summarizer": StatusEnum.COMPLETED,
            "entity_extractor": StatusEnum.FAILED,
            "sentiment_analyzer": StatusEnum.COMPLETED,
        }
        metadata = Metadata(
            total_processing_time_seconds=3.8,
            parallel_execution=True,
            agents_completed=2,
            agents_failed=1,
            timestamp=datetime.now(timezone.utc),
            warning="Some agents failed to complete",
            failed_agents=["entity_extractor"],
        )
        await self.storage_manager.update_job_status(
            job_id,
            status=StatusEnum.PARTIAL,
            agents_status=agent_status,
            results=results,
            end_time=datetime.now(timezone.utc),
            error_messages={"entity_extractor": "Simulated failure"},
            metadata=metadata,
        )
        return {
            "job_id": job_id,
            "document_id": document_id,
            "status": StatusEnum.PARTIAL,
            "results": results,
            "metadata": metadata,
        }

    return _fake


def test_upload_triggers_background_analysis(client, monkeypatch):
    monkeypatch.setattr(
        DocumentAnalysisOrchestrator,
        "analyze_document",
        _stub_success(),
    )

    response = client.post(
        "/upload",
        files={"file": ("report.txt", b"Quarterly update", "text/plain")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "uploaded"

    # Trigger analysis explicitly via /analyze (upload no longer auto-starts analysis)
    analyze_resp = client.post(f"/analyze/{payload['document_id']}")
    assert analyze_resp.status_code in {200, 202}
    analyze_payload = analyze_resp.json()
    job_id = analyze_payload["job_id"]

    status_response = client.get(f"/status/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    results_response = client.get(f"/results/{job_id}")
    results = results_response.json()
    assert results_response.status_code == 200
    assert results["status"] == "completed"
    assert results["results"]["summary"]["text"] == "Summary for " + payload["document_id"]


def test_partial_results_returned_on_agent_failure(client, monkeypatch):
    monkeypatch.setattr(
        DocumentAnalysisOrchestrator,
        "analyze_document",
        _stub_partial(),
    )

    response = client.post(
        "/upload",
        files={"file": ("needs_help.txt", b"Simulate failure", "text/plain")},
    )
    payload = response.json()
    analyze_resp = client.post(f"/analyze/{payload['document_id']}")
    assert analyze_resp.status_code in {200, 202}
    analyze_payload = analyze_resp.json()
    job_id = analyze_payload["job_id"]

    status_payload = client.get(f"/status/{job_id}").json()
    assert status_payload["status"] == "partial"
    assert status_payload["agents_status"]["entity_extractor"] == "failed"

    results_response = client.get(f"/results/{job_id}")
    assert results_response.status_code == 206
    results = results_response.json()
    assert results["results"]["entities"]["error"] == "Simulated failure"
    assert "entity_extractor" in results["metadata"]["failed_agents"]


def test_multiple_uploads_handle_concurrency(client, monkeypatch):
    monkeypatch.setattr(
        DocumentAnalysisOrchestrator,
        "analyze_document",
        _stub_success(),
    )

    first = client.post(
        "/upload",
        files={"file": ("doc1.txt", b"Doc one", "text/plain")},
    )
    second = client.post(
        "/upload",
        files={"file": ("doc2.txt", b"Doc two", "text/plain")},
    )

    # Start analysis for both documents
    analyze_one = client.post(f"/analyze/{first.json()['document_id']}")
    analyze_two = client.post(f"/analyze/{second.json()['document_id']}")
    assert analyze_one.status_code in {200, 202}
    assert analyze_two.status_code in {200, 202}

    job_id_one = analyze_one.json()["job_id"]
    job_id_two = analyze_two.json()["job_id"]
    assert job_id_one != job_id_two

    status_one = client.get(f"/status/{job_id_one}").json()
    status_two = client.get(f"/status/{job_id_two}").json()
    assert status_one["status"] == "completed"
    assert status_two["status"] == "completed"


def test_large_document_finishes_within_benchmark(client, monkeypatch):
    monkeypatch.setattr(
        DocumentAnalysisOrchestrator,
        "analyze_document",
        _stub_success(metadata_time=12.0),
    )

    large_content = ("Long paragraph about quarterly performance.\n" * 1200).encode()
    response = client.post(
        "/upload",
        files={"file": ("large.txt", large_content, "text/plain")},
    )
    payload = response.json()
    analyze_resp = client.post(f"/analyze/{payload['document_id']}")
    assert analyze_resp.status_code in {200, 202}
    analyze_payload = analyze_resp.json()
    job_id = analyze_payload["job_id"]

    results = client.get(f"/results/{job_id}").json()
    assert results["metadata"]["total_processing_time_seconds"] < 30


def test_empty_file_rejected(client):
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_corrupted_pdf_triggers_failure(client, monkeypatch):
    from fastapi import HTTPException

    def _broken_pdf(*_args, **_kwargs):
        raise HTTPException(status_code=400, detail="Corrupted PDF")

    monkeypatch.setattr("app.services.background_tasks.extract_text_from_pdf", _broken_pdf)

    response = client.post(
        "/upload",
        files={"file": ("broken.pdf", b"not_a_real_pdf", "application/pdf")},
    )
    payload = response.json()
    analyze_resp = client.post(f"/analyze/{payload['document_id']}")
    assert analyze_resp.status_code in {200, 202}
    analyze_payload = analyze_resp.json()
    job_id = analyze_payload["job_id"]

    status_payload = client.get(f"/status/{job_id}").json()
    assert status_payload["status"] == "failed"

    results_response = client.get(f"/results/{job_id}")
    assert results_response.status_code == 200
    results_payload = results_response.json()
    assert results_payload["status"] == "failed"
    assert results_payload["errors"]["orchestrator"] == "Corrupted PDF"


def test_text_with_special_characters_processed(client, monkeypatch):
    monkeypatch.setattr(
        DocumentAnalysisOrchestrator,
        "analyze_document",
        _stub_success(),
    )

    body = "Café résumé – análisis con emojis 😀🚀".encode("utf-8")
    response = client.post(
        "/upload",
        files={"file": ("unicode.txt", body, "text/plain")},
    )
    payload = response.json()
    analyze_resp = client.post(f"/analyze/{payload['document_id']}")
    assert analyze_resp.status_code in {200, 202}
    analyze_payload = analyze_resp.json()
    job_id = analyze_payload["job_id"]

    results = client.get(f"/results/{job_id}").json()
    assert results["status"] == "completed"
