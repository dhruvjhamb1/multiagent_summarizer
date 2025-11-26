import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from ..agents.entity_extractor import EntityExtractorAgent
from ..agents.keyword_extractor import KeywordExtractorAgent
from ..agents.sentiment_analyzer import SentimentAnalyzerAgent
from ..agents.summarizer import SummarizerAgent
from ..models.schemas import (
    AnalysisResults,
    CompleteAnalysisResult,
    Metadata,
    PartialResult,
    StatusEnum,
)
from ..models.storage import StorageManager

logger = logging.getLogger(__name__)


class DocumentAnalysisOrchestrator:
    """Coordinates agent execution for document analysis."""

    def __init__(
        self,
        storage_manager: StorageManager,
        summarizer: SummarizerAgent,
        entity_extractor: EntityExtractorAgent,
        sentiment_analyzer: SentimentAnalyzerAgent,
        keyword_extractor: KeywordExtractorAgent,
    ) -> None:
        self.storage_manager = storage_manager
        self.summarizer = summarizer
        self.entity_extractor = entity_extractor
        self.sentiment_analyzer = sentiment_analyzer
        self.keyword_extractor = keyword_extractor

    async def analyze_document(
        self,
        job_id: str,
        document_id: str,
        document_text: str,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        await self.storage_manager.update_job_status(
            job_id,
            status=StatusEnum.PROCESSING,
            agents_status={
                "summarizer": StatusEnum.PROCESSING,
                "entity_extractor": StatusEnum.PROCESSING,
                "sentiment_analyzer": StatusEnum.PROCESSING,
                "keyword_extractor": StatusEnum.PROCESSING,
            },
        )

        agents_status: Dict[str, StatusEnum] = {
            "summarizer": StatusEnum.PROCESSING,
            "entity_extractor": StatusEnum.PROCESSING,
            "sentiment_analyzer": StatusEnum.PROCESSING,
            "keyword_extractor": StatusEnum.PROCESSING,
        }
        results_payload: Dict[str, Any] = {}
        failed_agents: Dict[str, str] = {}

        async def run_and_update_agent(agent_name: str, agent, text: str):
            try:
                result = await self._run_agent(agent, text)
                agents_status[agent_name] = StatusEnum.COMPLETED
                results_payload[agent_name] = result
            except Exception as e:
                logger.error("Agent %s failed: %s", agent_name, e)
                agents_status[agent_name] = StatusEnum.FAILED
                failed_agents[agent_name] = str(e)
                results_payload[agent_name] = {"error": str(e)}
            
            # Real-time update after each agent completes
            await self.storage_manager.update_job_status(
                job_id,
                status=StatusEnum.PROCESSING,
                agents_status=agents_status.copy(),
                results=results_payload.copy(),
            )

        # Run all agents in parallel
        await asyncio.gather(
            run_and_update_agent("summarizer", self.summarizer, document_text),
            run_and_update_agent("entity_extractor", self.entity_extractor, document_text),
            run_and_update_agent("sentiment_analyzer", self.sentiment_analyzer, document_text),
            run_and_update_agent("keyword_extractor", self.keyword_extractor, document_text),
        )

        total_processing_time_seconds = round(time.perf_counter() - start_time, 4)

        summary_data = results_payload.get("summarizer")
        entity_data = results_payload.get("entity_extractor")
        sentiment_data = results_payload.get("sentiment_analyzer")
        keyword_data = results_payload.get("keyword_extractor")

        analysis_results = AnalysisResults(
            summary=summary_data,
            entities=entity_data,
            sentiment=sentiment_data,
            keywords=keyword_data,
        )

        failed_list = list(failed_agents.keys())
        warning = "Some agents failed to complete" if failed_list else None

        metadata = Metadata(
            total_processing_time_seconds=total_processing_time_seconds,
            parallel_execution=True,
            agents_completed=sum(1 for status in agents_status.values() if status == StatusEnum.COMPLETED),
            agents_failed=sum(1 for status in agents_status.values() if status == StatusEnum.FAILED),
            timestamp=datetime.now(timezone.utc),
            warning=warning,
            failed_agents=failed_list,
        )

        final_status = self._determine_status(agents_status)

        await self.storage_manager.update_job_status(
            job_id,
            status=final_status,
            agents_status=agents_status,
            results=results_payload,
            end_time=datetime.now(timezone.utc),
            error_messages=failed_agents,
            metadata=metadata,
        )

        document = await self.storage_manager.get_document(document_id)
        document_name = document.filename if document else document_id

        if final_status == StatusEnum.COMPLETED:
            complete_result = CompleteAnalysisResult(
                job_id=job_id,
                document_id=document_id,
                document_name=document_name,
                status=final_status,
                results=analysis_results,
                metadata=metadata,
            )
            return complete_result.model_dump()

        partial_result = PartialResult(
            job_id=job_id,
            document_id=document_id,
            document_name=document_name,
            status=final_status,
            results=analysis_results,
            failed_agents=list(failed_agents.keys()),
            metadata=metadata,
        )
        return partial_result.model_dump()

    async def _run_agent(self, agent, document_text: str) -> Dict[str, Any]:
        start = time.perf_counter()
        result = await agent.execute(document_text)
        duration = round(time.perf_counter() - start, 4)
        logger.info("Agent %s finished in %s seconds", agent.agent_name, duration)

        if result.get("status") == "error":
            raise RuntimeError(result.get("message", "Agent returned error."))
        return result.get("data", {})

    def _determine_status(self, agents_status: Dict[str, StatusEnum]) -> StatusEnum:
        completed = sum(1 for status in agents_status.values() if status == StatusEnum.COMPLETED)
        failed = sum(1 for status in agents_status.values() if status == StatusEnum.FAILED)

        if completed == len(agents_status):
            return StatusEnum.COMPLETED
        if failed == len(agents_status):
            return StatusEnum.FAILED
        return StatusEnum.PARTIAL

