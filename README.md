# Multi-Agent Document Analysis System

A production-ready FastAPI application that performs intelligent document analysis using a parallel multi-agent architecture powered by CrewAI. The system automatically extracts summaries, entities, and sentiment analysis from PDF and TXT documents using OpenAI's GPT-4 models.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Latest-orange.svg)](https://www.crewai.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                     (Async Request Handler)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Document Upload Endpoint                       │
│              (File Validation & Storage)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Background Task Orchestrator                       │
│           (Manages Agent Lifecycle & Coordination)              │
└────────┬────────────────┬────────────────┬───────────────────────┘
         │                │                │
    ┌────▼────┐      ┌────▼────┐     ┌────▼────┐
    │Summarizer│      │ Entity  │     │Sentiment│
    │  Agent   │      │Extractor│     │Analyzer │
    │ (GPT-4)  │      │ (GPT-4) │     │ (GPT-4) │
    └────┬────┘      └────┬────┘     └────┬────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Storage Manager     │
              │  (In-Memory + File)   │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Results Endpoint    │
              │  (JSON Response API)  │
              └───────────────────────┘
```

### Key Components

1. **FastAPI Server**: Handles HTTP requests with async support for high concurrency
2. **Document Processor**: Validates and extracts text from PDF/TXT files
3. **Orchestrator**: Coordinates parallel agent execution with timeout and error handling
4. **Three Specialized Agents**:
   - **Summarizer**: Generates concise summaries with key points
   - **Entity Extractor**: Identifies people, organizations, dates, locations, and monetary values
   - **Sentiment Analyzer**: Analyzes emotional tone, formality, urgency, and objectivity
5. **Storage Manager**: Maintains document and job state in-memory
6. **Background Task Service**: Manages async job processing

### Parallelization Model

The system uses **asyncio-based parallel execution** with `asyncio.gather()` to run all three agents simultaneously on the same document. This approach:
- Reduces total processing time compared to sequential execution
- Maintains independent agent state for fault isolation
- Provides real-time status updates as each agent completes
- Handles partial failures gracefully

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API Key
- Docker (optional, for containerized deployment)

### Local Setup (Development)

1. **Clone the repository**
```bash
git clone https://github.com/dhruvjhamb1/multiagent_summarizer.git
cd multiagent_summarizer
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create .env file in project root
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=True
HOST=0.0.0.0
PORT=8000
MAX_FILE_SIZE_MB=10
AGENT_TIMEOUT_SECONDS=30
STORAGE_PATH=./storage
CREWAI_TRACING_ENABLED=False
EOF
```

5. **Create required directories**
```bash
mkdir -p storage/uploads
```

6. **Run the server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker Setup (Production)

1. **Configure environment**
```bash
# Create .env file with production values
cp .env.example .env
# Edit .env with your OpenAI API key
```

2. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

3. **Access the application**
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8000/dashboard`
- API Docs: `http://localhost:8000/docs`

### Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","version":"1.0.0"}
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints Overview

| Method | Endpoint | Description | Status Codes |
|--------|----------|-------------|--------------|
| GET | `/` | Welcome message | 200 |
| GET | `/health` | Health check | 200 |
| GET | `/dashboard` | Job queue dashboard UI | 200 |
| POST | `/upload` | Upload document | 201, 400, 413 |
| POST | `/analyze/{document_id}` | Start analysis | 202, 200, 404 |
| GET | `/status/{job_id}` | Get job status | 200, 404 |
| GET | `/results/{job_id}` | Get analysis results | 200, 202, 206, 404 |
| GET | `/jobs` | List all jobs | 200 |

---

### 1. Upload Document

Upload a PDF or TXT file for analysis.

**Endpoint:** `POST /upload`

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

**Response (201 Created):**
```json
{
  "document_id": "doc_a1b2c3d4e5f6",
  "filename": "document.pdf",
  "size_bytes": 102400,
  "upload_timestamp": "2025-11-24T10:00:00.000000Z",
  "status": "uploaded",
  "message": "Document uploaded successfully"
}
```

**Validation Rules:**
- File types: `.pdf`, `.txt`
- Max file size: 10MB (configurable)
- Non-empty files only

**Error Responses:**
```bash
# Invalid file type
HTTP 400: {"detail": "Invalid file type. Only PDF and TXT files are allowed."}

# File too large
HTTP 413: {"detail": "File too large. Maximum size is 10MB."}

# Empty file
HTTP 400: {"detail": "Uploaded file is empty."}
```

---

### 2. Start Analysis

Initiate multi-agent analysis on an uploaded document.

**Endpoint:** `POST /analyze/{document_id}`

**Request:**
```bash
curl -X POST "http://localhost:8000/analyze/doc_a1b2c3d4e5f6"
```

**Response (202 Accepted):**
```json
{
  "job_id": "job_z9y8x7w6v5u4",
  "document_id": "doc_a1b2c3d4e5f6",
  "status": "pending",
  "message": "Analysis started"
}
```

**Idempotent Behavior:**
If analysis already exists for the document:
```json
{
  "job_id": "job_z9y8x7w6v5u4",
  "document_id": "doc_a1b2c3d4e5f6",
  "status": "completed",
  "message": "Analysis already completed"
}
```

**Error Responses:**
```bash
# Document not found
HTTP 404: {"detail": "Document not found."}

# Server error
HTTP 500: {"detail": "Failed to start analysis."}
```

---

### 3. Check Analysis Status

Monitor real-time progress of analysis job.

**Endpoint:** `GET /status/{job_id}`

**Request:**
```bash
curl -X GET "http://localhost:8000/status/job_z9y8x7w6v5u4"
```

**Response (200 OK):**
```json
{
  "job_id": "job_z9y8x7w6v5u4",
  "document_id": "doc_a1b2c3d4e5f6",
  "status": "processing",
  "agents_status": {
    "summarizer": "completed",
    "entity_extractor": "processing",
    "sentiment_analyzer": "pending"
  },
  "progress_percentage": 33.33,
  "start_time": "2025-11-24T10:05:00.000000Z"
}
```

**Status Values:**
- `pending`: Job queued, not started
- `processing`: Agents actively working
- `completed`: All agents succeeded
- `partial`: Some agents succeeded, others failed
- `failed`: All agents failed

---

### 4. Get Analysis Results

Retrieve final analysis results with detailed insights.

**Endpoint:** `GET /results/{job_id}`

**Request:**
```bash
curl -X GET "http://localhost:8000/results/job_z9y8x7w6v5u4"
```

**Response (200 OK - Complete):**
```json
{
  "job_id": "job_z9y8x7w6v5u4",
  "document_id": "doc_a1b2c3d4e5f6",
  "document_name": "quarterly_report.pdf",
  "status": "completed",
  "results": {
    "summary": {
      "text": "The quarterly report shows strong revenue growth of 25% YoY...",
      "key_points": [
        "Revenue increased 25% year-over-year",
        "Customer base expanded to 10,000 active users",
        "New product line launched in Q3"
      ],
      "confidence": 0.92,
      "processing_time": 1.8
    },
    "entities": {
      "people": ["John Smith", "Jane Doe"],
      "organizations": ["Acme Corp", "TechVentures Inc"],
      "dates": ["Q3 2025", "November 15, 2025"],
      "locations": ["San Francisco", "New York"],
      "monetary_values": ["$5.2M", "$12.8M revenue"],
      "processing_time": 1.6
    },
    "sentiment": {
      "overall": "positive",
      "confidence": 0.88,
      "tone": {
        "formality": "formal",
        "urgency": "medium",
        "objectivity": "balanced"
      },
      "emotional_indicators": {
        "optimistic": 0.7,
        "confident": 0.65
      },
      "key_phrases": [
        {
          "text": "strong momentum",
          "sentiment": "positive"
        },
        {
          "text": "exceeded expectations",
          "sentiment": "positive"
        }
      ],
      "processing_time": 1.2
    }
  },
  "metadata": {
    "total_processing_time_seconds": 4.6,
    "parallel_execution": true,
    "agents_completed": 3,
    "agents_failed": 0,
    "timestamp": "2025-11-24T10:11:45.000000Z",
    "warning": null,
    "failed_agents": []
  }
}
```

**Response (206 Partial Content - Partial Success):**
```json
{
  "job_id": "job_z9y8x7w6v5u4",
  "document_id": "doc_a1b2c3d4e5f6",
  "document_name": "research_article.pdf",
  "status": "partial",
  "results": {
    "summary": {
      "text": "Research findings indicate...",
      "key_points": ["Finding 1", "Finding 2"],
      "confidence": 0.85,
      "processing_time": 2.1
    },
    "entities": {
      "error": "Entity extractor timed out"
    },
    "sentiment": {
      "overall": "neutral",
      "confidence": 0.72,
      "tone": {
        "formality": "formal",
        "urgency": "low",
        "objectivity": "objective"
      },
      "emotional_indicators": {},
      "key_phrases": [],
      "processing_time": 1.5
    }
  },
  "metadata": {
    "total_processing_time_seconds": 30.8,
    "parallel_execution": true,
    "agents_completed": 2,
    "agents_failed": 1,
    "timestamp": "2025-11-24T10:12:30.000000Z",
    "warning": "Some agents failed to complete",
    "failed_agents": ["entity_extractor"]
  }
}
```

**Response (202 Accepted - Still Processing):**
```json
{
  "job_id": "job_z9y8x7w6v5u4",
  "document_id": "doc_a1b2c3d4e5f6",
  "status": "processing",
  "message": "Analysis in progress"
}
```

---

### 5. List All Jobs

Retrieve all analysis jobs with current status.

**Endpoint:** `GET /jobs`

**Request:**
```bash
curl -X GET "http://localhost:8000/jobs"
```

**Response (200 OK):**
```json
{
  "jobs": [
    {
      "job_id": "job_z9y8x7w6v5u4",
      "document_id": "doc_a1b2c3d4e5f6",
      "document_name": "quarterly_report.pdf",
      "status": "completed",
      "progress_percentage": 100.0,
      "start_time": "2025-11-24T10:05:00.000000Z",
      "end_time": "2025-11-24T10:11:45.000000Z"
    },
    {
      "job_id": "job_abc123def456",
      "document_id": "doc_xyz789uvw012",
      "document_name": "research_article.pdf",
      "status": "processing",
      "progress_percentage": 66.67,
      "start_time": "2025-11-24T10:15:00.000000Z",
      "end_time": null
    }
  ],
  "total_count": 2
}
```

---


### Postman Collection

Import the following JSON to Postman for ready-to-use API calls:

```json
{
  "info": {
    "name": "Multi-Agent Document Analysis API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Upload Document",
      "request": {
        "method": "POST",
        "header": [],
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "file",
              "type": "file",
              "src": "/path/to/document.pdf"
            }
          ]
        },
        "url": "{{base_url}}/upload"
      }
    },
    {
      "name": "Start Analysis",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/analyze/{{document_id}}"
      }
    },
    {
      "name": "Get Status",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/status/{{job_id}}"
      }
    },
    {
      "name": "Get Results",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/results/{{job_id}}"
      }
    },
    {
      "name": "List Jobs",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/jobs"
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

## 🎯 Design Decisions

### Parallelization Approach

**Decision:** Used `asyncio.gather()` for concurrent agent execution instead of threading or multiprocessing.


- **Async/Await Pattern**: Agents primarily perform I/O-bound operations (API calls to OpenAI), making async ideal for efficient resource utilization without thread overhead.
- **Shared State Safety**: Asyncio's single-threaded event loop eliminates race conditions, simplifying storage manager implementation.
- **Resource Efficiency**: Can handle number of concurrent jobs on modest hardware compared to thread-per-request models.
- **Error Isolation**: Each agent runs in an independent coroutine, so one failure doesn't cascade to others.

**Trade-offs:** For CPU-intensive preprocessing (e.g., OCR on scanned PDFs), would need to offload to thread pool executors.

### Agent Failure Handling

**Strategy:** Graceful degradation with partial results.

**Implementation:**
1. **Timeout Guards**: Each agent has a 30-second timeout (configurable) using `asyncio.wait_for()`.
2. **Try-Catch Per Agent**: Individual exception handling prevents one agent's failure from breaking the entire pipeline.
3. **Status Tracking**: Real-time updates as each agent completes, stored in `agents_status` dict.
4. **Partial Results**: System returns `partial` status if ≥1 agent succeeds, with clear indication of which agents failed.
5. **Error Messages**: Failed agents return structured error responses with `error_type` (timeout vs exception) and detailed messages.

**Benefit:** Users get usable results even when 1-2 agents fail, dramatically improving reliability.

### Performance Optimizations

1. **Parallel Execution**: All 3 agents run simultaneously, reducing latency.
2. **In-Memory Storage**: Current implementation uses in-memory dictionaries for sub-millisecond lookups.
3. **Text Extraction Caching**: Extracted document text is stored once and reused by all agents.
4. **Connection Pooling**: FastAPI's async HTTP client reuses OpenAI API connections.
5. **Gunicorn Workers**: Docker deployment uses 4 Uvicorn workers for horizontal scaling.

### Known Limitations & Future Improvements

**Current Limitations:**
1. **No Persistence**: In-memory storage loses state on restart (migrate to PostgreSQL + Redis).
2. **Single Node**: Cannot scale horizontally without job queue.
3. **LLM Dependency**: 100% reliant on OpenAI API availability.
4. **File Size Limit**: 10MB cap may be insufficient for large reports (implement chunking for 100MB+ files).
5. **No Authentication**: API is completely open (add OAuth2/JWT tokens).

**Roadmap:**
- [ ] Add WebSocket support for real-time progress streaming
- [ ] Implement result caching to avoid re-analyzing identical documents
- [ ] Support for additional file formats (HTML, Markdown, etc.)
- [ ] Prometheus metrics and Grafana dashboards for observability
- [ ] Multi-tenant support with per-user quotas
- [ ] Add vector database (Pinecone/Weaviate) for semantic document search
- [ ] Implement custom CrewAI agents for domain-specific analysis (legal, medical, financial)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 Project Structure

```
multiagent_summarizer/
├── app/
│   ├── main.py                 # FastAPI application & routes
│   ├── config.py               # Environment configuration
│   ├── agents/
│   │   ├── base_agent.py       # Abstract base class with timeout handling
│   │   ├── summarizer.py       # Summary generation agent
│   │   ├── entity_extractor.py # Named entity recognition agent
│   │   └── sentiment_analyzer.py # Sentiment analysis agent
│   ├── models/
│   │   ├── schemas.py          # Pydantic response models
│   │   └── storage.py          # In-memory storage manager
│   ├── services/
│   │   ├── orchestrator.py     # Agent coordination logic
│   │   └── background_tasks.py # Async job processing
│   └── utils/
│       ├── file_processor.py   # File validation & text extraction
│       └── helpers.py          # Utility functions
├── static/
│   └── dashboard.html          # Job queue visualization UI
├── storage/
│   └── uploads/                # Uploaded document storage
├── tests/
│   └── test_api.py            # API endpoint tests
├── sample_documents/           # Example files for testing
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image definition
├── docker-compose.yml          # Service orchestration
└── README.md                 
```

## 🔧 Configuration

Environment variables (`.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `DEBUG` | Enable debug logging | `True` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MAX_FILE_SIZE_MB` | Max upload size | `10` |
| `AGENT_TIMEOUT_SECONDS` | Per-agent timeout | `30` |
| `STORAGE_PATH` | File storage directory | `./storage` |
| `CREWAI_TRACING_ENABLED` | Enable CrewAI tracing | `False` |

## 🐛 Troubleshooting

### Issue: "OpenAI API key not found"
**Solution:** Ensure `.env` file exists with valid `OPENAI_API_KEY=sk-...`

### Issue: Analysis stuck at "processing"
**Solution:** Check logs for timeout errors. Increase `AGENT_TIMEOUT_SECONDS` or verify OpenAI API connectivity.

### Issue: "File too large" error
**Solution:** Increase `MAX_FILE_SIZE_MB` in `.env` or compress document.

### Issue: Docker container exits immediately
**Solution:** Run `docker-compose logs app` to view error messages. Usually missing `.env` file.

## 📧 Support

For issues and questions:
- GitHub Issues: [Create Issue](https://github.com/dhruvjhamb1/multiagent_summarizer/issues)
- Documentation: [API Docs](http://localhost:8000/docs)

---

**Built with ❤️ using FastAPI, CrewAI, and OpenAI GPT-4**
