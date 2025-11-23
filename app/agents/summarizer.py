import json
import logging
import re
import time
from typing import Any, Dict

from crewai import Agent as CrewAgent, Crew, Task
from crewai import LLM

from .base_agent import BaseDocumentAgent
from ..utils.helpers import retry_with_backoff
from ..config import settings

logger = logging.getLogger(__name__)

SUMMARY_INSTRUCTIONS = (
    "You are an expert document summarizer. Generate a concise summary between 100 and 150 words. "
    "The summary must highlight the central narrative and preserve critical details."
)
SUMMARY_EXPECTED_OUTPUT = (
    "JSON object with keys: text (100-150 word summary), key_points (3-5 bullet points), confidence (0-1 float)."
)
SUMMARY_PROMPT_TEMPLATE = (
    "Document to summarize:\n\"\"\"{document_text}\"\"\"\n\n"
    "Follow these requirements strictly:\n"
    "1. Write 100-150 words covering main themes and conclusions.\n"
    "2. Provide 3-5 bullet points capturing critical insights.\n"
    "3. Deliver a confidence score between 0 and 1 (float).\n"
    "4. Respond only with valid JSON structured as: {{\"text\": str, \"key_points\": [str, ...], \"confidence\": float}}."
)


class SummarizerAgent(BaseDocumentAgent):

    @property
    def agent_name(self) -> str:
        return "summarizer"

    async def process(self, document_text: str) -> dict:
        start_time = time.perf_counter()
        if not document_text or not document_text.strip():
            raise ValueError("Document text is required for summarization.")

        if not self.llm:
            raise RuntimeError("Summarizer requires an LLM client. Configure provider credentials.")

        crew_agent = self._build_agent()
        task = self._build_task(document_text, crew_agent)
        crew = Crew(agents=[crew_agent], tasks=[task])

        try:
            raw_output = await retry_with_backoff(
                crew.kickoff_async,
                inputs={"document_text": document_text},
                max_attempts=3,
                initial_delay=0.5,
                multiplier=2.0,
                jitter=0.1,
                call_timeout=settings.agent_timeout_seconds,
            )
        except Exception as exc:
            logger.exception("Crew execution failed for summarizer agent")
            raise RuntimeError(f"Summarization failed: {exc}") from exc

        parsed = self._parse_output(raw_output)
        processing_time = round(time.perf_counter() - start_time, 4)
        parsed["processing_time"] = processing_time
        return parsed

    def _build_agent(self) -> CrewAgent:
        agent_kwargs: Dict[str, Any] = {
            "role": "Document Summarizer",
            "goal": "Generate concise, accurate summaries of documents",
            "backstory": "Expert at distilling complex documents into clear summaries",
        }
        if self.llm:
            agent_kwargs["llm"] = LLM(
                model=self.llm.model,
                api_key=settings.openai_api_key,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
        return CrewAgent(**agent_kwargs)

    def _build_task(self, document_text: str, agent: CrewAgent) -> Task:
        description = f"{SUMMARY_INSTRUCTIONS}\n\n{SUMMARY_PROMPT_TEMPLATE.format(document_text=document_text)}"
        return Task(
            description=description,
            expected_output=SUMMARY_EXPECTED_OUTPUT,
            agent=agent,
        )

    def _parse_output(self, raw_output: Any) -> Dict[str, Any]:
        # Handle CrewOutput object
        if hasattr(raw_output, 'raw'):
            output_data = raw_output.raw
        else:
            output_data = raw_output
        
        if isinstance(output_data, dict):
            data = output_data
        else:
            data = self._decode_json(output_data)

        text = data.get("text")
        key_points = data.get("key_points")
        confidence = data.get("confidence")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Summary text missing from agent response.")

        if not isinstance(key_points, list) or not all(isinstance(item, str) for item in key_points):
            raise ValueError("Key points must be a list of strings.")
        if not 3 <= len(key_points) <= 5:
            raise ValueError("Key points must contain between 3 and 5 items.")

        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Confidence must be a numeric value.") from exc

        if not 0 <= confidence_val <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        return {
            "text": text.strip(),
            "key_points": [item.strip() for item in key_points],
            "confidence": confidence_val,
        }

    def _decode_json(self, raw_output: Any) -> Dict[str, Any]:
        if raw_output is None:
            raise ValueError("Agent returned no output.")

        if isinstance(raw_output, str):
            candidate = raw_output.strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", candidate, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError as exc:
                        logger.debug("JSON extraction failed for summarizer output: %s", candidate)
                        raise ValueError("Summarizer produced malformed JSON.") from exc
                raise ValueError("Summarizer produced non-JSON output.")

        raise ValueError("Agent response type is unsupported for JSON parsing.")