from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime
import asyncio
from .schemas import Metadata, StatusEnum

@dataclass
class DocumentStorage:
    document_id: str
    filename: str
    file_path: str
    size_bytes: int
    upload_timestamp: datetime
    content_text: str

@dataclass
class JobStorage:
    job_id: str
    document_id: str
    status: StatusEnum
    agents_status: Dict[str, StatusEnum]
    results: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    error_messages: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[Metadata] = None

class StorageManager:
    def __init__(self):
        self.documents: Dict[str, DocumentStorage] = {}
        self.jobs: Dict[str, JobStorage] = {}
        self._lock = asyncio.Lock()


    async def save_document(
        self,
        filename: str,
        file_path: str,
        size_bytes: int,
        upload_timestamp: datetime,
        content_text: str,
        document_id: Optional[str] = None,
    ) -> str:
        async with self._lock:
            document_id = document_id or self._generate_doc_id()
            doc = DocumentStorage(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                size_bytes=size_bytes,
                upload_timestamp=upload_timestamp,
                content_text=content_text
            )
            self.documents[document_id] = doc
            return document_id

    async def get_document(self, document_id: str) -> Optional[DocumentStorage]:
        async with self._lock:
            return self.documents.get(document_id)

    async def save_job(
        self,
        document_id: str,
        job_id: Optional[str] = None,
        status: StatusEnum = StatusEnum.PENDING,
        agents_status: Optional[Dict[str, StatusEnum]] = None,
    ) -> str:
        async with self._lock:
            job_id = job_id or self._generate_job_id()
            job = JobStorage(
                job_id=job_id,
                document_id=document_id,
                status=status,
                agents_status=agents_status
                or {
                    "summarizer": StatusEnum.PENDING,
                    "entity_extractor": StatusEnum.PENDING,
                    "sentiment_analyzer": StatusEnum.PENDING
                },
                results={},
                start_time=datetime.now()
            )
            self.jobs[job_id] = job
            return job_id

    async def update_job_status(self, job_id: str, status: StatusEnum, agents_status: Optional[Dict[str, StatusEnum]] = None,
                               results: Optional[Dict[str, Any]] = None, end_time: Optional[datetime] = None,
                               error_messages: Optional[Dict[str, str]] = None, metadata: Optional[Metadata] = None) -> bool:
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            job.status = status
            if agents_status:
                job.agents_status.update(agents_status)
            if results:
                job.results.update(results)
            if end_time:
                job.end_time = end_time
            if error_messages:
                job.error_messages.update(error_messages)
            if metadata:
                job.metadata = metadata
            return True

    async def get_job(self, job_id: str) -> Optional[JobStorage]:
        async with self._lock:
            return self.jobs.get(job_id)

    async def get_latest_job_for_document(self, document_id: str) -> Optional[JobStorage]:
        async with self._lock:
            matches: List[JobStorage] = [
                job for job in self.jobs.values() if job.document_id == document_id
            ]
            if not matches:
                return None
            matches.sort(key=lambda job: job.start_time, reverse=True)
            return matches[0]

    async def get_all_jobs(self) -> List[JobStorage]:
        async with self._lock:
            return list(self.jobs.values())