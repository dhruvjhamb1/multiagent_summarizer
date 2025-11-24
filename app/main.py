import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agents.entity_extractor import EntityExtractorAgent
from .agents.sentiment_analyzer import SentimentAnalyzerAgent
from .agents.summarizer import SummarizerAgent
from .config import settings
from .models.schemas import (
    AnalysisInitiationResponse,
    AnalysisResults,
    AnalysisStatus,
    CompleteAnalysisResult,
    EntityResult,
    JobList,
    JobListItem,
    Metadata,
    SentimentResult,
    StatusEnum,
    SummaryResult,
    UploadResponse,
)
from .models.storage import StorageManager
from .services.background_tasks import BackgroundTaskService
from .services.orchestrator import DocumentAnalysisOrchestrator
from .utils.file_processor import save_uploaded_file, validate_file
from .utils.helpers import calculate_progress, generate_document_id, generate_job_id
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Multi-Agent Document Analysis System v%s", app.version)
    yield
    logger.info("Shutting down Multi-Agent Document Analysis System")
    
app = FastAPI(
    title="Multi-Agent Document Analysis System",
    description="A FastAPI application for multi-agent document analysis using CrewAI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

storage_manager_singleton = StorageManager()
summarizer_agent_singleton = SummarizerAgent()
entity_extractor_agent_singleton = EntityExtractorAgent()
sentiment_analyzer_agent_singleton = SentimentAnalyzerAgent()
orchestrator_singleton = DocumentAnalysisOrchestrator(
    storage_manager_singleton,
    summarizer_agent_singleton,
    entity_extractor_agent_singleton,
    sentiment_analyzer_agent_singleton,
)
background_service_singleton = BackgroundTaskService(orchestrator_singleton, storage_manager_singleton)


def get_storage_manager() -> StorageManager:
    return storage_manager_singleton


def get_background_service() -> BackgroundTaskService:
    return background_service_singleton


MAX_FILE_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred."},
    )


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    return {"status": "ok", "version": app.version}

UPLOAD_SUCCESS_EXAMPLE = {
    "document_id": "doc_a1b2c3",
    "filename": "sample.pdf",
    "size_bytes": 102400,
    "upload_timestamp": "2025-11-22T10:00:00Z",
    "status": "uploaded",
    "message": "Document uploaded successfully",
}

ANALYZE_ACCEPTED_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "status": "pending",
    "message": "Analysis started",
}

ANALYZE_EXISTING_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "status": "completed",
    "message": "Analysis already completed",
}

ANALYZE_NOT_FOUND_EXAMPLE = {
    "detail": "Document not found.",
}

ANALYZE_ERROR_EXAMPLE = {
    "detail": "Failed to start analysis.",
}

STATUS_SUCCESS_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "status": "processing",
    "agents_status": {
        "summarizer": "completed",
        "entity_extractor": "processing",
        "sentiment_analyzer": "pending",
    },
    "progress_percentage": 50.0,
    "start_time": "2025-11-22T10:05:00Z",
}

STATUS_NOT_FOUND_EXAMPLE = {
    "detail": "Job not found.",
}

RESULTS_PENDING_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "status": "processing",
    "message": "Analysis in progress",
}

RESULTS_FAILED_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "status": "failed",
    "errors": {
        "summarizer": "Summarizer timed out",
        "entity_extractor": "Agent returned error",
    },
}

RESULTS_PARTIAL_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "document_name": "sample.pdf",
    "status": "partial",
    "results": {
        "summary": {
            "text": "Summary here",
            "key_points": ["point1"],
            "confidence": 0.82,
            "processing_time": 2.0,
        },
        "entities": {"error": "Entity extractor failed"},
        "sentiment": {
            "overall": "neutral",
            "confidence": 0.6,
            "tone": {
                "formality": "neutral",
                "urgency": "low",
                "objectivity": "balanced",
            },
            "emotional_indicators": {"note": "Limited signal"},
            "key_phrases": [],
            "processing_time": 1.3,
        },
    },
    "metadata": {
        "total_processing_time_seconds": 4.8,
        "parallel_execution": True,
        "agents_completed": 2,
        "agents_failed": 1,
        "timestamp": "2025-11-22T10:12:00Z",
        "warning": "Some agents failed to complete",
        "failed_agents": ["entity_extractor"],
    },
}

RESULTS_COMPLETED_EXAMPLE = {
    "job_id": "job_z9y8x7",
    "document_id": "doc_a1b2c3",
    "document_name": "sample.pdf",
    "status": "completed",
    "results": {
        "summary": {
            "text": "Summary here",
            "key_points": ["point1", "point2"],
            "confidence": 0.9,
            "processing_time": 1.9,
        },
        "entities": {
            "people": [],
            "organizations": [],
            "dates": [],
            "locations": [],
            "monetary_values": [],
            "processing_time": 1.6,
        },
        "sentiment": {
            "overall": "positive",
            "confidence": 0.88,
            "tone": {
                "formality": "formal",
                "urgency": "medium",
                "objectivity": "balanced",
            },
            "emotional_indicators": {"optimistic": 0.7},
            "key_phrases": [{"text": "strong momentum", "sentiment": "positive"}],
            "processing_time": 1.1,
        },
    },
    "metadata": {
        "total_processing_time_seconds": 4.6,
        "parallel_execution": True,
        "agents_completed": 3,
        "agents_failed": 0,
        "timestamp": "2025-11-22T10:11:45Z",
        "warning": None,
        "failed_agents": [],
    },
}


@app.get("/", tags=["Health"])
def read_root() -> Dict[str, Any]:
    return {
        "message": "Welcome to the Multi-Agent Document Analysis System",
        "version": app.version,
    }


@app.get("/dashboard", tags=["Dashboard"])
def dashboard():
    """Serve the job queue dashboard."""
    dashboard_path = Path(__file__).parent.parent / "static" / "dashboard.html"
    return FileResponse(dashboard_path, media_type="text/html")


@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Document",
    description="Upload a PDF or TXT document for analysis by the multi-agent pipeline.",
    responses={
        201: {
            "description": "Document uploaded successfully",
            "content": {"application/json": {"example": UPLOAD_SUCCESS_EXAMPLE}},
        },
        400: {
            "description": "Invalid file upload.",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid file type. Only PDF and TXT files are allowed."}
                }
            },
        },
        413: {
            "description": "Uploaded file exceeds size limit.",
            "content": {
                "application/json": {
                    "example": {"detail": "File too large. Maximum size is 10MB."}
                }
            },
        },
        500: {
            "description": "Server error while storing file.",
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to store uploaded file."}
                }
            },
        },
    },
)
async def upload_document(
    file: UploadFile = File(...),
    storage: StorageManager = Depends(get_storage_manager)
) -> UploadResponse:
    try:
        contents = await file.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded file %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file.",
        ) from exc

    size_bytes = len(contents)
    del contents
    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB.",
        )

    setattr(file, "size", size_bytes)
    await file.seek(0)

    try:
        validate_file(file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file upload.",
        ) from exc

    document_id = generate_document_id()
    upload_timestamp = datetime.now(timezone.utc)

    await file.seek(0)
    try:
        file_path = await save_uploaded_file(file, document_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to store uploaded file %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded file.",
        ) from exc

    await storage.save_document(
        filename=file.filename,
        file_path=file_path,
        size_bytes=size_bytes,
        upload_timestamp=upload_timestamp,
        content_text="",
        document_id=document_id,
    )

    # BackgroundTaskService.schedule_background_analysis(
    #     background_tasks,
    #     background_service,
    #     job_id,
    #     document_id,
    # )

    logger.info("Document %s uploaded successfully", document_id)

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        size_bytes=size_bytes,
        upload_timestamp=upload_timestamp,
        status=StatusEnum.UPLOADED,
        message="Document uploaded successfully",
    )


@app.post(
    "/analyze/{document_id}",
    response_model=AnalysisInitiationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start Document Analysis",
    description="Launch asynchronous multi-agent analysis for a previously uploaded document.",
    responses={
        202: {
            "description": "Analysis job accepted for processing.",
            "content": {"application/json": {"example": ANALYZE_ACCEPTED_EXAMPLE}},
        },
        200: {
            "description": "Existing analysis job returned (idempotent response).",
            "content": {"application/json": {"example": ANALYZE_EXISTING_EXAMPLE}},
        },
        404: {
            "description": "Document not found.",
            "content": {"application/json": {"example": ANALYZE_NOT_FOUND_EXAMPLE}},
        },
        500: {
            "description": "Server error while starting analysis.",
            "content": {"application/json": {"example": ANALYZE_ERROR_EXAMPLE}},
        },
    },
)
async def analyze_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    response: Response,
    storage: StorageManager = Depends(get_storage_manager),
    background_service: BackgroundTaskService = Depends(get_background_service),
) -> AnalysisInitiationResponse:
    """Trigger background analysis for the provided document identifier."""
    try:
        document = await storage.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        existing_job = await storage.get_latest_job_for_document(document_id)
        if existing_job and existing_job.status in {
            StatusEnum.PROCESSING,
            StatusEnum.COMPLETED,
            StatusEnum.PARTIAL,
        }:
            # For PROCESSING/COMPLETED/PARTIAL we simply return the existing job
            if existing_job.status == StatusEnum.PROCESSING:
                message = "Analysis already in progress"
                response.status_code = status.HTTP_202_ACCEPTED
            elif existing_job.status == StatusEnum.PARTIAL:
                message = "Partial analysis previously completed"
                response.status_code = status.HTTP_200_OK
            else:
                message = "Analysis already completed"
                response.status_code = status.HTTP_200_OK
            logger.info(
                "Returning existing analysis job %s for document %s with status %s",
                existing_job.job_id,
                document_id,
                existing_job.status,
            )
            return AnalysisInitiationResponse(
                job_id=existing_job.job_id,
                document_id=document_id,
                status=existing_job.status,
                message=message,
            )

        # If an existing PENDING job exists, schedule analysis for it.
        if existing_job and existing_job.status == StatusEnum.PENDING:
            job_id = existing_job.job_id
        else:
            job_id = generate_job_id()
            await storage.save_job(document_id=document_id, job_id=job_id)

        BackgroundTaskService.schedule_background_analysis(
            background_tasks,
            background_service,
            job_id,
            document_id,
        )

        logger.info("Queued analysis job %s for document %s", job_id, document_id)

        return AnalysisInitiationResponse(
            job_id=job_id,
            document_id=document_id,
            status=StatusEnum.PENDING,
            message="Analysis started",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to start analysis for document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis.",
        ) from exc


@app.get(
    "/status/{job_id}",
    response_model=AnalysisStatus,
    summary="Get Analysis Status",
    description="Fetch the current status and progress metrics for an analysis job.",
    responses={
        200: {
            "description": "Current job status returned.",
            "content": {"application/json": {"example": STATUS_SUCCESS_EXAMPLE}},
        },
        404: {
            "description": "Job not found.",
            "content": {"application/json": {"example": STATUS_NOT_FOUND_EXAMPLE}},
        },
    },
)
async def get_status(
    job_id: str,
    response: Response,
    storage: StorageManager = Depends(get_storage_manager),
) -> AnalysisStatus:
    """Return the analysis job status, including per-agent progress."""
    job = await storage.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    default_agent_states = {
        "summarizer": StatusEnum.PENDING,
        "entity_extractor": StatusEnum.PENDING,
        "sentiment_analyzer": StatusEnum.PENDING,
    }
    agents_status = {**default_agent_states, **job.agents_status}

    progress_percentage = float(calculate_progress(agents_status))

    if job.status == StatusEnum.COMPLETED:
        response.headers["Cache-Control"] = "public, max-age=60"

    return AnalysisStatus(
        job_id=job.job_id,
        document_id=job.document_id,
        status=job.status,
        agents_status=agents_status,
        progress_percentage=progress_percentage,
        start_time=job.start_time,
    )


@app.get(
    "/results/{job_id}",
    response_model=CompleteAnalysisResult,
    summary="Get Analysis Results",
    description="Retrieve final or partial analysis results for a job once processing has completed.",
    responses={
        200: {
            "description": "Analysis job completed successfully.",
            "content": {"application/json": {"example": RESULTS_COMPLETED_EXAMPLE}},
        },
        202: {
            "description": "Analysis still in progress.",
            "content": {"application/json": {"example": RESULTS_PENDING_EXAMPLE}},
        },
        206: {
            "description": "Analysis completed with partial results due to agent failures.",
            "content": {"application/json": {"example": RESULTS_PARTIAL_EXAMPLE}},
        },
        404: {
            "description": "Job not found.",
            "content": {"application/json": {"example": STATUS_NOT_FOUND_EXAMPLE}},
        },
    },
)
async def get_results(
    job_id: str,
    response: Response,
    storage: StorageManager = Depends(get_storage_manager),
) -> CompleteAnalysisResult:
    """Return aggregated analysis results or intermediate status for the requested job."""
    job = await storage.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )

    if job.status in {StatusEnum.PENDING, StatusEnum.PROCESSING}:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job.job_id,
                "document_id": job.document_id,
                "status": job.status.value,
                "message": "Analysis in progress",
            },
        )

    errors = job.error_messages or {}
    if job.status == StatusEnum.FAILED and not job.results:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "job_id": job.job_id,
                "document_id": job.document_id,
                "status": job.status.value,
                "errors": errors,
            },
        )

    document = await storage.get_document(job.document_id)
    document_name = document.filename if document else job.document_id
    results_payload = job.results or {}

    def build_section(agent_key: str, model_cls):
        error_message = errors.get(agent_key)
        payload = results_payload.get(agent_key)

        if error_message:
            return {"error": error_message}

        if payload in (None, {}):
            return None

        if isinstance(payload, dict) and "error" in payload:
            return payload

        if isinstance(payload, model_cls):
            return payload

        if isinstance(payload, dict):
            try:
                return model_cls(**payload)
            except Exception as exc:
                logger.warning("Malformed %s payload for job %s: %s", agent_key, job_id, exc)
                return {"error": f"Malformed {agent_key.replace('_', ' ')} result"}

        logger.warning("Unexpected payload type for %s in job %s", agent_key, job_id)
        return {"error": f"Unexpected payload type for {agent_key}"}

    summary_section = build_section("summarizer", SummaryResult)
    entities_section = build_section("entity_extractor", EntityResult)
    sentiment_section = build_section("sentiment_analyzer", SentimentResult)

    metadata = job.metadata.model_copy() if job.metadata else None
    if metadata is None:
        total_processing_time_seconds = 0.0
        if job.end_time:
            total_processing_time_seconds = round((job.end_time - job.start_time).total_seconds(), 4)
        metadata = Metadata(
            total_processing_time_seconds=total_processing_time_seconds,
            parallel_execution=True,
            agents_completed=sum(1 for status_value in job.agents_status.values() if status_value == StatusEnum.COMPLETED),
            agents_failed=sum(1 for status_value in job.agents_status.values() if status_value == StatusEnum.FAILED),
            timestamp=job.end_time or datetime.now(timezone.utc),
            warning="Some agents failed to complete" if errors else None,
            failed_agents=list(errors.keys()),
        )
    else:
        update: Dict[str, Any] = {}
        if metadata.warning is None and errors:
            update["warning"] = "Some agents failed to complete"
        if not metadata.failed_agents and errors:
            update["failed_agents"] = list(errors.keys())
        if update:
            metadata = metadata.model_copy(update=update)

    analysis_results = AnalysisResults(
        summary=summary_section,
        entities=entities_section,
        sentiment=sentiment_section,
    )

    if job.status == StatusEnum.PARTIAL:
        response.status_code = status.HTTP_206_PARTIAL_CONTENT
    else:
        response.status_code = status.HTTP_200_OK

    if job.status == StatusEnum.COMPLETED:
        response.headers["Cache-Control"] = "public, max-age=300"

    return CompleteAnalysisResult(
        job_id=job.job_id,
        document_id=job.document_id,
        document_name=document_name,
        status=job.status,
        results=analysis_results,
        metadata=metadata,
    )


@app.get(
    "/jobs",
    response_model=JobList,
    summary="List All Jobs",
    description="Retrieve a list of all analysis jobs with their current status.",
    responses={
        200: {
            "description": "List of jobs returned successfully.",
        },
    },
)
async def list_jobs(
    storage: StorageManager = Depends(get_storage_manager),
) -> JobList:
    jobs = await storage.get_all_jobs()
    
    job_items = []
    for job in jobs:
        document = await storage.get_document(job.document_id)
        document_name = document.filename if document else job.document_id
        
        default_agent_states = {
            "summarizer": StatusEnum.PENDING,
            "entity_extractor": StatusEnum.PENDING,
            "sentiment_analyzer": StatusEnum.PENDING,
        }
        agents_status = {**default_agent_states, **job.agents_status}
        progress_percentage = float(calculate_progress(agents_status))
        
        job_item = JobListItem(
            job_id=job.job_id,
            document_id=job.document_id,
            document_name=document_name,
            status=job.status,
            progress_percentage=progress_percentage,
            start_time=job.start_time,
            end_time=job.end_time,
        )
        job_items.append(job_item)
    
    job_items.sort(key=lambda x: x.start_time, reverse=True)
    
    return JobList(
        jobs=job_items,
        total_count=len(job_items),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )