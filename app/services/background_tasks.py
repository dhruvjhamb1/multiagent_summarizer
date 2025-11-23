import asyncio
import logging
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException

from ..models.storage import StorageManager
from ..models.schemas import StatusEnum
from ..utils.file_processor import extract_text_from_pdf, extract_text_from_txt
from .orchestrator import DocumentAnalysisOrchestrator

logger = logging.getLogger(__name__)

class BackgroundTaskService:

    def __init__(self, orchestrator: DocumentAnalysisOrchestrator, storage_manager: StorageManager) -> None:
        self.orchestrator = orchestrator
        self.storage_manager = storage_manager

    async def run_analysis_task(self, job_id: str, document_id: str) -> None:
        logger.info("Starting background analysis for job %s", job_id)
        document = await self.storage_manager.get_document(document_id)
        if not document:
            logger.error("Document %s not found for job %s", document_id, job_id)
            await self.storage_manager.update_job_status(
                job_id,
                status=StatusEnum.FAILED,
                error_messages={"orchestrator": "Document not found."},
            )
            return

        file_path = Path(document.file_path)
        try:
            if file_path.suffix.lower() == ".pdf":
                logger.debug("Extracting text from PDF for job %s", job_id)
                document_text = await asyncio.to_thread(extract_text_from_pdf, str(file_path))
                from inspect import isawaitable

                if isawaitable(document_text):
                    document_text = await document_text
            else:
                logger.debug("Extracting text from TXT for job %s", job_id)
                document_text = await asyncio.to_thread(extract_text_from_txt, str(file_path))
                from inspect import isawaitable

                if isawaitable(document_text):
                    document_text = await document_text

            logger.debug("Dispatching orchestrator for job %s", job_id)

            if not document_text or not document_text.strip():
                logger.warning("No text extracted for job %s, marking job as failed", job_id)
                await self.storage_manager.update_job_status(
                    job_id,
                    status=StatusEnum.FAILED,
                    error_messages={"orchestrator": "No text could be extracted from the uploaded file."},
                )
                return

            await self.orchestrator.analyze_document(job_id, document_id, document_text)
            logger.info("Completed background analysis for job %s", job_id)
        except Exception as exc:
            logger.exception("Background analysis failed for job %s", job_id)
            if isinstance(exc, HTTPException):
                error_message = exc.detail if exc.detail else str(exc)
            else:
                error_message = str(exc)
            await self.storage_manager.update_job_status(
                job_id,
                status=StatusEnum.FAILED,
                error_messages={"orchestrator": error_message},
            )
        finally:
            self._cleanup_file(file_path)

    @staticmethod
    def schedule_background_analysis(
        background_tasks: BackgroundTasks,
        service: "BackgroundTaskService",
        job_id: str,
        document_id: str,
    ) -> None:
        """Convenience wrapper to schedule the analysis task."""
        background_tasks.add_task(service.run_analysis_task, job_id, document_id)

    @staticmethod
    def _cleanup_file(file_path: Path) -> None:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug("Removed temporary file %s", file_path)
        except Exception as exc:
            logger.warning("Failed to remove temporary file %s: %s", file_path, exc)
