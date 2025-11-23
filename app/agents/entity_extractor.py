import json
import logging
import re
import time
from typing import Any, Dict, Iterable, List

from crewai import Agent as CrewAgent, Crew, Task
from crewai import LLM  

from .base_agent import BaseDocumentAgent
from ..utils.helpers import retry_with_backoff
from ..config import settings

logger = logging.getLogger(__name__)

ENTITY_INSTRUCTIONS = (
    "You are an expert entity extraction specialist. Identify and categorize entities with supporting context."
)

ENTITY_EXPECTED_OUTPUT = (
    "JSON object with keys: people, organizations, dates, locations, monetary_values. "
    "Each key maps to a list of objects with fields for name/value, context, mentions count, and optional role/type."
)

ENTITY_PROMPT_TEMPLATE = (
    "Document for analysis:\n\"\"\"{document_text}\"\"\"\n\n"
    "Tasks:\n"
    "1. Extract the following categories: people, organizations, dates, locations, monetary values.\n"
    "2. For people include role/title when available.\n"
    "3. For organizations include organization type if known.\n"
    "4. For locations include location type (e.g., city, country, state) if known.\n"
    "5. Provide concise context snippet for each entity.\n"
    "6. Count how many times each entity appears (mentions integer) for people, organizations, and locations.\n"
    "7. Deduplicate entities by canonical name; merge mentions and contexts.\n"
    "8. Respond only with valid JSON structured as described in EXPECTED_OUTPUT.\n\n"
    "IMPORTANT:\n"
    "• Return AT MOST 5 entities per category (choose the most relevant or most frequently mentioned).\n"
    "• All descriptive fields (context, role, type) must be MAX 3 WORDS."
)


class EntityExtractorAgent(BaseDocumentAgent):

    @property
    def agent_name(self) -> str:
        return "entity_extractor"

    async def process(self, document_text: str) -> dict:
        start_time = time.perf_counter()
        if not document_text or not document_text.strip():
            raise ValueError("Document text is required for entity extraction.")

        if not self.llm:
            raise RuntimeError("Entity extractor requires an LLM client. Configure provider credentials.")

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
            logger.exception("Crew execution failed for entity extractor agent")
            raise RuntimeError(f"Entity extraction failed: {exc}") from exc

        parsed = self._parse_output(raw_output)
        processing_time = round(time.perf_counter() - start_time, 4)
        parsed["processing_time"] = processing_time
        return parsed

    def _build_agent(self) -> CrewAgent:
        agent_kwargs: Dict[str, Any] = {
            "role": "Entity Extraction Specialist",
            "goal": "Identify and categorize all entities in documents with context",
            "backstory": "Expert at NER and entity classification",
        }
        if self.llm:
            agent_kwargs["llm"] = LLM(
                model=self.llm.model,
                api_key=settings.openai_api_key,
                temperature=self.llm.temperature,
                max_tokens=5000
            )
        return CrewAgent(**agent_kwargs)

    def _build_task(self, document_text: str, agent: CrewAgent) -> Task:
        description = f"{ENTITY_INSTRUCTIONS}\n\n{ENTITY_PROMPT_TEMPLATE.format(document_text=document_text)}"
        return Task(
            description=description,
            expected_output=ENTITY_EXPECTED_OUTPUT,
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

        categories = {
            "people": self._normalize_entities(data.get("people", []), entity_type="person"),
            "organizations": self._normalize_entities(data.get("organizations", []), entity_type="organization"),
            "dates": self._normalize_entities(data.get("dates", []), entity_type="date"),
            "locations": self._normalize_entities(data.get("locations", []), entity_type="location"),
            "monetary_values": self._normalize_entities(data.get("monetary_values", []), entity_type="monetary"),
        }

        return categories

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
                        logger.debug("JSON extraction failed for entity output: %s", candidate)
                        raise ValueError("Entity extractor produced malformed JSON.") from exc
                raise ValueError("Entity extractor produced non-JSON output.")

        raise ValueError("Agent response type is unsupported for JSON parsing.")

    def _normalize_entities(self, items: Any, *, entity_type: str) -> List[Dict[str, Any]]:
        if items is None:
            return []
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
            raise ValueError(f"Entities for {entity_type} must be provided as a list.")

        if entity_type in ("person", "organization", "location"):
            dedup: Dict[str, Dict[str, Any]] = {}
            for raw in items:
                if not isinstance(raw, dict):
                    raise ValueError(f"Entity entries for {entity_type} must be objects.")

                name = self._extract_name(raw)
                if not name:
                    continue

                key = name.lower().strip()
                mentions = self._extract_mentions(raw)

                if entity_type == "person":
                    role = self._extract_role(raw)
                    item = {"name": name.strip(), "role": role, "mentions": mentions}
                elif entity_type == "organization":
                    type_ = self._extract_type(raw)
                    item = {"name": name.strip(), "type": type_, "mentions": mentions}
                elif entity_type == "location":
                    type_ = self._extract_type(raw)
                    item = {"name": name.strip(), "type": type_, "mentions": mentions}

                if key not in dedup:
                    dedup[key] = item
                else:
                    existing = dedup[key]
                    existing["mentions"] += mentions
                    # Keep the first role/type

            return list(dedup.values())

        elif entity_type in ("date", "monetary"):
            result = []
            for raw in items:
                if not isinstance(raw, dict):
                    raise ValueError(f"Entity entries for {entity_type} must be objects.")

                name = self._extract_name(raw)
                if not name:
                    continue

                context = self._compose_context(raw, entity_type)

                if entity_type == "date":
                    result.append({"date": name.strip(), "context": context})
                elif entity_type == "monetary":
                    result.append({"amount": name.strip(), "context": context})

            return result

        else:
            return []

    def _extract_name(self, raw: Dict[str, Any]) -> str:
        for key in ("name", "value", "amount", "date", "entity", "text"):
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return ""

    def _extract_mentions(self, raw: Dict[str, Any]) -> int:
        value = raw.get("mentions")
        try:
            mentions = int(value)
        except (TypeError, ValueError):
            mentions = 1
        return max(mentions, 1)

    def _extract_role(self, raw: Dict[str, Any]) -> str:
        for key in ("role", "title", "position"):
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return ""

    def _extract_type(self, raw: Dict[str, Any]) -> str:
        for key in ("type", "category", "kind"):
            candidate = raw.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return ""

    def _compose_context(self, raw: Dict[str, Any], entity_type: str) -> str:
        snippets: List[str] = []
        role_keys = ["role", "title", "type", "category"]
        context_keys = ["context", "description", "detail", "sentence"]

        for key in role_keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                snippets.append(value.strip())

        for key in context_keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                snippets.append(value.strip())

        if not snippets:
            snippets.append(f"Mentioned {entity_type} in document.")

        return " | ".join(snippets)