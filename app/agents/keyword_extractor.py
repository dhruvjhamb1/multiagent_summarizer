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

KEYWORD_INSTRUCTIONS = (
    "You are a keyword extraction specialist. Identify the top 5 most important keywords from the document."
)
KEYWORD_EXPECTED_OUTPUT = (
    "JSON with keys: keywords (list of exactly 5 strings)."
)
KEYWORD_PROMPT_TEMPLATE = (
    "Document to analyze:\"\"\"{document_text}\"\"\"\n\n"
    "Follow these requirements strictly:\n"
    "1. Extract exactly 5 keywords that best represent the document's main topics and themes.\n"
    "2. Prioritize keywords by importance and relevance.\n"
    "3. Respond only with valid JSON using the structure: {{\"keywords\": [str, str, str, str, str]}}."
)


class KeywordExtractorAgent(BaseDocumentAgent):
    
    @property
    def agent_name(self) -> str:
        return "keyword_extractor"

    async def process(self, document_text: str) -> dict:
        start_time = time.perf_counter()
        if not document_text or not document_text.strip():
            return self._default_response("Document text empty; returning default keywords.", start_time)

        if not self.llm:
            raise RuntimeError("Keyword extractor requires an LLM client. Configure provider credentials.")

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
            logger.exception("Crew execution failed for keyword extractor agent")
            raise RuntimeError(f"Keyword extraction failed: {exc}") from exc

        parsed = self._parse_output(raw_output)
        parsed["processing_time"] = round(time.perf_counter() - start_time, 4)
        return parsed

    def _build_agent(self) -> CrewAgent:
        agent_kwargs: Dict[str, Any] = {
            "role": "Keyword Extraction Specialist",
            "goal": "Extract the top 5 most relevant keywords from documents",
            "backstory": "Expert at identifying key terms and topics in text",
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
        description = f"{KEYWORD_INSTRUCTIONS}\n\n{KEYWORD_PROMPT_TEMPLATE.format(document_text=document_text)}"
        return Task(
            description=description,
            expected_output=KEYWORD_EXPECTED_OUTPUT,
            agent=agent,
        )

    def _parse_output(self, raw_output: Any) -> Dict[str, Any]:
        if hasattr(raw_output, 'raw'):
            output_data = raw_output.raw
        else:
            output_data = raw_output
        
        if isinstance(output_data, dict):
            data = output_data
        else:
            data = self._decode_json(output_data)

        keywords = data.get("keywords")

        if not isinstance(keywords, list):
            raise ValueError("Keywords must be a list.")
        
        if len(keywords) != 5:
            raise ValueError("Exactly 5 keywords are required.")
        
        if not all(isinstance(kw, str) and kw.strip() for kw in keywords):
            raise ValueError("All keywords must be non-empty strings.")

        return {
            "keywords": [kw.strip() for kw in keywords],
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
                        logger.debug("JSON extraction failed for keyword output: %s", candidate)
                        raise ValueError("Keyword extractor produced malformed JSON.") from exc
                raise ValueError("Keyword extractor produced non-JSON output.")

        raise ValueError("Agent response type is unsupported for JSON parsing.")

    def _default_response(self, note: str, start_time: float) -> Dict[str, Any]:
        processing_time = round(time.perf_counter() - start_time, 4)
        return {
            "keywords": ["document", "text", "content", "analysis", "data"],
            "processing_time": processing_time,
            "note": note,
        }
