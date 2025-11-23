from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    UPLOADED = "uploaded"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class UploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename of the uploaded document")
    size_bytes: int = Field(..., description="Size of the document in bytes")
    upload_timestamp: datetime = Field(..., description="Timestamp when the document was uploaded")
    status: StatusEnum = Field(..., description="Status of the upload")
    message: str = Field(..., description="Additional message about the upload")

    @field_validator('size_bytes')
    @classmethod
    def validate_size(cls, v):
        if v < 0:
            raise ValueError('Size must be non-negative')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "doc_12345",
                "filename": "sample.pdf",
                "size_bytes": 1024000,
                "upload_timestamp": "2023-11-22T10:00:00Z",
                "status": "uploaded",
                "message": "Document uploaded successfully"
            }
        }
    }

class AnalysisStatus(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the analysis job")
    document_id: str = Field(..., description="ID of the document being analyzed")
    status: StatusEnum = Field(..., description="Overall status of the analysis")
    agents_status: Dict[str, StatusEnum] = Field(..., description="Status of each agent involved")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    start_time: datetime = Field(..., description="Time when analysis started")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "job_67890",
                "document_id": "doc_12345",
                "status": "processing",
                "agents_status": {"summarizer": "completed", "entity_extractor": "processing"},
                "progress_percentage": 66.7,
                "start_time": "2023-11-22T10:05:00Z"
            }
        }
    }

class AnalysisInitiationResponse(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the analysis job")
    document_id: str = Field(..., description="ID of the document to analyze")
    status: StatusEnum = Field(..., description="Status of the analysis job")
    message: str = Field(..., description="Message describing initiation result")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "job_a1b2c3",
                "document_id": "doc_12345",
                "status": "pending",
                "message": "Analysis started"
            }
        }
    }

class SummaryResult(BaseModel):
    text: str = Field(..., description="The summarized text")
    key_points: List[str] = Field(..., description="List of key points extracted")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    processing_time: float = Field(..., ge=0, description="Time taken to process in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This document discusses AI advancements...",
                "key_points": ["AI is transforming industries", "Key challenges remain"],
                "confidence": 0.85,
                "processing_time": 2.5
            }
        }
    }

class PersonEntity(BaseModel):
    name: str = Field(..., description="Name of the person")
    role: str = Field(..., description="Role or title of the person")
    mentions: int = Field(..., ge=0, description="Number of mentions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "John Smith",
                "role": "CEO",
                "mentions": 5
            }
        }
    }

class OrganizationEntity(BaseModel):
    name: str = Field(..., description="Name of the organization")
    type: str = Field(..., description="Type of the organization")
    mentions: int = Field(..., ge=0, description="Number of mentions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Acme Corp",
                "type": "company",
                "mentions": 12
            }
        }
    }

class DateEntity(BaseModel):
    date: str = Field(..., description="The date")
    context: str = Field(..., description="Context where the date appears")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2024-Q3",
                "context": "reporting period"
            }
        }
    }

class LocationEntity(BaseModel):
    name: str = Field(..., description="Name of the location")
    type: str = Field(..., description="Type of the location")
    mentions: int = Field(..., ge=0, description="Number of mentions")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "New York",
                "type": "city",
                "mentions": 4
            }
        }
    }

class MonetaryEntity(BaseModel):
    amount: str = Field(..., description="The monetary amount")
    context: str = Field(..., description="Context where the amount appears")

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": "$45.2M",
                "context": "quarterly revenue"
            }
        }
    }

class EntityResult(BaseModel):
    people: List[PersonEntity] = Field(default_factory=list, description="List of people entities")
    organizations: List[OrganizationEntity] = Field(default_factory=list, description="List of organization entities")
    dates: List[DateEntity] = Field(default_factory=list, description="List of date entities")
    locations: List[LocationEntity] = Field(default_factory=list, description="List of location entities")
    monetary_values: List[MonetaryEntity] = Field(default_factory=list, description="List of monetary value entities")
    processing_time: float = Field(..., ge=0, description="Time taken to process in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "people": [{"name": "John Smith", "role": "CEO", "mentions": 5}],
                "organizations": [{"name": "Acme Corp", "type": "company", "mentions": 12}],
                "dates": [{"date": "2024-Q3", "context": "reporting period"}],
                "locations": [{"name": "New York", "type": "city", "mentions": 4}],
                "monetary_values": [{"amount": "$45.2M", "context": "quarterly revenue"}],
                "processing_time": 4.1
            }
        }
    }

class Tone(BaseModel):
    formality: str = Field(..., description="Formality level (e.g., formal, informal)")
    urgency: str = Field(..., description="Urgency level (e.g., high, low)")
    objectivity: str = Field(..., description="Objectivity level (e.g., objective, subjective)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "formality": "formal",
                "urgency": "medium",
                "objectivity": "objective"
            }
        }
    }

class KeyPhrase(BaseModel):
    text: str = Field(..., description="The key phrase text")
    sentiment: str = Field(..., description="Sentiment of the phrase (positive, negative, neutral)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "excellent results",
                "sentiment": "positive"
            }
        }
    }

class SentimentResult(BaseModel):
    overall: str = Field(..., description="Overall sentiment (positive, negative, neutral)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    tone: Tone = Field(..., description="Detailed tone analysis")
    emotional_indicators: Dict[str, Any] = Field(default_factory=dict, description="Emotional indicators dictionary")
    key_phrases: List[KeyPhrase] = Field(default_factory=list, description="List of key phrases with sentiment")
    processing_time: float = Field(..., ge=0, description="Time taken to process in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "overall": "positive",
                "confidence": 0.92,
                "tone": {"formality": "formal", "urgency": "low", "objectivity": "objective"},
                "emotional_indicators": {"joy": 0.8, "anger": 0.1},
                "key_phrases": [{"text": "great success", "sentiment": "positive"}],
                "processing_time": 1.2
            }
        }
    }

class AnalysisResults(BaseModel):
    summary: Optional[Union[SummaryResult, Dict[str, Any]]] = Field(None, description="Summary result or error payload")
    entities: Optional[Union[EntityResult, Dict[str, Any]]] = Field(None, description="Entity extraction result or error payload")
    sentiment: Optional[Union[SentimentResult, Dict[str, Any]]] = Field(None, description="Sentiment analysis result or error payload")

class Metadata(BaseModel):
    total_processing_time_seconds: float = Field(..., ge=0, description="Total time for all processing")
    parallel_execution: bool = Field(..., description="Whether agents ran in parallel")
    agents_completed: int = Field(..., ge=0, description="Number of agents that completed successfully")
    agents_failed: int = Field(..., ge=0, description="Number of agents that failed")
    timestamp: datetime = Field(..., description="Completion timestamp")
    warning: Optional[str] = Field(None, description="Optional warning message about analysis outcome")
    failed_agents: List[str] = Field(default_factory=list, description="List of agents that failed during analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_processing_time_seconds": 5.5,
                "parallel_execution": True,
                "agents_completed": 2,
                "agents_failed": 1,
                "timestamp": "2023-11-22T10:10:00Z",
                "warning": "Some agents failed to complete",
                "failed_agents": ["entity_extractor"],
            }
        }
    }

class JobListItem(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the analysis job")
    document_id: str = Field(..., description="ID of the document being analyzed")
    document_name: str = Field(..., description="Name of the document")
    status: StatusEnum = Field(..., description="Overall status of the analysis")
    progress_percentage: float = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    start_time: datetime = Field(..., description="Time when analysis started")
    end_time: Optional[datetime] = Field(None, description="Time when analysis completed")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "job_67890",
                "document_id": "doc_12345",
                "document_name": "sample.pdf",
                "status": "processing",
                "progress_percentage": 66.7,
                "start_time": "2023-11-22T10:05:00Z",
                "end_time": None
            }
        }
    }

class JobList(BaseModel):
    jobs: List[JobListItem] = Field(..., description="List of analysis jobs")
    total_count: int = Field(..., description="Total number of jobs")

    model_config = {
        "json_schema_extra": {
            "example": {
                "jobs": [
                    {
                        "job_id": "job_67890",
                        "document_id": "doc_12345",
                        "document_name": "sample.pdf",
                        "status": "processing",
                        "progress_percentage": 66.7,
                        "start_time": "2023-11-22T10:05:00Z",
                        "end_time": None
                    }
                ],
                "total_count": 1
            }
        }
    }


class CompleteAnalysisResult(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the analysis job")
    document_id: str = Field(..., description="ID of the analyzed document")
    document_name: str = Field(..., description="Name of the analyzed document")
    status: StatusEnum = Field(..., description="Status of the analysis")
    results: AnalysisResults = Field(..., description="Results from all agents")
    metadata: Metadata = Field(..., description="Metadata about the analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "job_67890",
                "document_id": "doc_12345",
                "document_name": "sample.pdf",
                "status": "completed",
                "results": {
                    "summary": {"text": "Summary here", "key_points": ["point1"], "confidence": 0.8, "processing_time": 2.0},
                    "entities": {"people": [], "organizations": [], "processing_time": 1.5},
                    "sentiment": {"overall": "neutral", "confidence": 0.7, "tone": {"formality": "neutral"}, "processing_time": 1.0}
                },
                "metadata": {
                    "total_processing_time_seconds": 4.5,
                    "parallel_execution": True,
                    "agents_completed": 3,
                    "agents_failed": 0,
                    "timestamp": "2023-11-22T10:10:00Z",
                    "warning": None,
                    "failed_agents": []
                }
            }
        }
    }

class PartialResult(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the analysis job")
    document_id: str = Field(..., description="ID of the analyzed document")
    document_name: str = Field(..., description="Name of the analyzed document")
    status: StatusEnum = Field(default=StatusEnum.PARTIAL, description="Status (partial)")
    results: AnalysisResults = Field(..., description="Partial results from completed agents")
    failed_agents: List[str] = Field(default_factory=list, description="List of agents that failed")
    metadata: Metadata = Field(..., description="Metadata about the analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "job_67890",
                "document_id": "doc_12345",
                "document_name": "sample.pdf",
                "status": "partial",
                "results": {
                    "summary": {"text": "Summary here", "key_points": ["point1"], "confidence": 0.8, "processing_time": 2.0},
                    "entities": None,
                    "sentiment": None
                },
                "failed_agents": ["entity_extractor", "sentiment_analyzer"],
                "metadata": {
                    "total_processing_time_seconds": 2.0,
                    "parallel_execution": True,
                    "agents_completed": 1,
                    "agents_failed": 2,
                    "timestamp": "2023-11-22T10:07:00Z",
                    "warning": "Some agents failed to complete",
                    "failed_agents": ["entity_extractor", "sentiment_analyzer"]
                }
            }
        }
    }