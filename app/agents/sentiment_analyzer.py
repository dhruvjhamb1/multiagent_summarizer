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

SENTIMENT_INSTRUCTIONS = (
    "You are a sentiment and tone analyst. Provide nuanced emotional analysis for the document."
)
SENTIMENT_EXPECTED_OUTPUT = (
    "JSON with keys: overall (positive/negative/neutral), confidence (0-1 float), tone (formality, urgency, objectivity), "
    "emotional_indicators (dict of emotion: score), key_phrases (list of {text, sentiment})."
)
SENTIMENT_PROMPT_TEMPLATE = (
    "Document to analyze:\n\"\"\"{document_text}\"\"\"\n\n"
    "Deliverables:\n"
    "1. Overall sentiment label (positive, negative, neutral).\n"
    "2. Confidence score between 0 and 1.\n"
    "3. Tone analysis: formality (formal/informal), urgency (urgent/casual), objectivity (objective/subjective).\n"
    "4. Emotional indicators with scores from 0-1 (e.g., optimistic, confident, cautious, joyful, frustrated).\n"
    "5. Provide 3-5 key phrases with their sentiment labels.\n"
    "6. If text is empty or non-English, report neutral sentiment with note in emotional indicators.\n"
    "7. Respond only with valid JSON using the structure described in EXPECTED OUTPUT."
)


class SentimentAnalyzerAgent(BaseDocumentAgent):
    
    @property
    def agent_name(self) -> str:
        return "sentiment_analyzer"

    async def process(self, document_text: str) -> dict:
        start_time = time.perf_counter()
        if not document_text or not document_text.strip():
            return self._neutral_response("Document text empty; defaulting to neutral sentiment.", start_time)

        if not self.llm:
            raise RuntimeError("Sentiment analyzer requires an LLM client. Configure provider credentials.")

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
            logger.exception("Crew execution failed for sentiment analyzer agent")
            raise RuntimeError(f"Sentiment analysis failed: {exc}") from exc

        parsed = self._parse_output(raw_output)
        parsed["processing_time"] = round(time.perf_counter() - start_time, 4)
        return parsed

    def _build_agent(self) -> CrewAgent:
        agent_kwargs: Dict[str, Any] = {
            "role": "Sentiment & Tone Analyst",
            "goal": "Analyze document sentiment, tone, and emotional indicators",
            "backstory": "Expert at understanding emotional tone and sentiment in text",
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
        description = f"{SENTIMENT_INSTRUCTIONS}\n\n{SENTIMENT_PROMPT_TEMPLATE.format(document_text=document_text)}"
        return Task(
            description=description,
            expected_output=SENTIMENT_EXPECTED_OUTPUT,
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

        overall = data.get("overall") or data.get("sentiment")
        confidence = data.get("confidence")
        tone = data.get("tone")
        emotional_indicators = data.get("emotional_indicators", {})
        key_phrases = data.get("key_phrases", [])

        if not isinstance(overall, str) or overall.lower() not in {"positive", "neutral", "negative"}:
            raise ValueError("Overall sentiment must be one of: positive, neutral, negative.")

        try:
            confidence_val = float(confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("Confidence must be a numeric value.") from exc
        if not 0 <= confidence_val <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        tone_dict = self._normalize_tone(tone)
        emotions = self._normalize_emotions(emotional_indicators)
        phrases = self._normalize_key_phrases(key_phrases)

        return {
            "overall": overall.lower(),
            "confidence": confidence_val,
            "tone": tone_dict,
            "emotional_indicators": emotions,
            "key_phrases": phrases,
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
                        logger.debug("JSON extraction failed for sentiment output: %s", candidate)
                        raise ValueError("Sentiment analyzer produced malformed JSON.") from exc
                raise ValueError("Sentiment analyzer produced non-JSON output.")

        raise ValueError("Agent response type is unsupported for JSON parsing.")

    def _normalize_tone(self, tone: Any) -> Dict[str, str]:
        if not isinstance(tone, dict):
            raise ValueError("Tone must be an object with formality, urgency, objectivity.")

        def select_value(key: str, default: str) -> str:
            value = tone.get(key, default)
            if isinstance(value, str) and value.strip():
                return value.strip()
            return default

        return {
            "formality": select_value("formality", "neutral"),
            "urgency": select_value("urgency", "neutral"),
            "objectivity": select_value("objectivity", "balanced"),
        }

    def _normalize_emotions(self, emotions: Any) -> Dict[str, float]:
        if emotions is None:
            return {}
        if not isinstance(emotions, dict):
            raise ValueError("Emotional indicators must be a dictionary of scores.")

        normalized: Dict[str, float] = {}
        for key, value in emotions.items():
            if not isinstance(key, str) or not key.strip():
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                logger.debug("Skipping emotion %s due to non-numeric score", key)
                continue
            normalized[key.strip().lower()] = max(0.0, min(score, 1.0))
        return normalized

    def _normalize_key_phrases(self, phrases: Any) -> List[Dict[str, str]]:
        if phrases is None:
            return []
        if not isinstance(phrases, Iterable) or isinstance(phrases, (str, bytes)):
            raise ValueError("Key phrases must be provided as a list.")

        normalized: List[Dict[str, str]] = []
        for item in phrases:
            if isinstance(item, dict):
                text = item.get("text") or item.get("phrase")
                sentiment = item.get("sentiment") or item.get("label")
            elif isinstance(item, str):
                text = item
                sentiment = "neutral"
            else:
                raise ValueError("Key phrase entries must be objects or strings.")

            if not isinstance(text, str) or not text.strip():
                continue

            sentiment_val = sentiment if isinstance(sentiment, str) else "neutral"
            normalized.append({
                "text": text.strip(),
                "sentiment": sentiment_val.strip().lower(),
            })

        if len(normalized) > 5:
            normalized = normalized[:5]
        return normalized

    def _neutral_response(self, note: str, start_time: float) -> Dict[str, Any]:
        processing_time = round(time.perf_counter() - start_time, 4)
        return {
            "overall": "neutral",
            "confidence": 0.0,
            "tone": {
                "formality": "neutral",
                "urgency": "neutral",
                "objectivity": "balanced",
            },
            "emotional_indicators": {"note": note},
            "key_phrases": [],
            "processing_time": processing_time,
        }