# AI Microservice

Centralized AI processing service for the Statex microservices ecosystem. Provides intelligent analysis, content generation, and prototype creation services through multiple AI agents.

## Features

- ✅ **AI Orchestrator** - Central coordination for all AI operations
- ✅ **NLP Service** - Text analysis and business plan generation
- ✅ **ASR Service** - Speech-to-text conversion
- ✅ **Document AI** - File analysis and OCR
- ✅ **Prototype Generator** - Website and application prototype creation
- ✅ **Template Repository** - Template management
- ✅ **Free AI Service** - Free AI models integration
- ✅ **AI Workers** - Background AI processing
- ✅ **Gemini AI Service** - Google Gemini AI integration
- ✅ **Data Viz Service** - Data visualization
- ✅ **Shared Database** - All AI agents data stored in shared database-server
- ✅ **Centralized Logging** - All logs sent to logging-microservice
- ✅ **Blue/Green Deployment** - Zero-downtime deployments

## Technology Stack

- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL (via shared database-server)
- **Cache**: Redis (via shared database-server)
- **Logging**: External centralized logging microservice
- **Network**: nginx-network (shared Docker network)

## Port Configuration

**Port Range**: 338x (shared microservices)

| Service | Host Port | Container Port | .env Variable | Description |
|---------|-----------|----------------|---------------|-------------|
| **AI Orchestrator** | `${AI_ORCHESTRATOR_PORT:-3380}` | `${AI_ORCHESTRATOR_PORT:-3380}` | `AI_ORCHESTRATOR_PORT` | Central AI coordination |
| **NLP Service** | `${NLP_SERVICE_PORT:-3381}` | `${NLP_SERVICE_PORT:-3381}` | `NLP_SERVICE_PORT` | Text analysis and generation |
| **ASR Service** | `${ASR_SERVICE_PORT:-3382}` | `${ASR_SERVICE_PORT:-3382}` | `ASR_SERVICE_PORT` | Speech-to-text conversion |
| **Document AI** | `${DOCUMENT_AI_PORT:-3383}` | `${DOCUMENT_AI_PORT:-3383}` | `DOCUMENT_AI_PORT` | Document processing |
| **Prototype Generator** | `${PROTOTYPE_GENERATOR_PORT:-3384}` | `${PROTOTYPE_GENERATOR_PORT:-3384}` | `PROTOTYPE_GENERATOR_PORT` | Website prototype creation |
| **Template Repository** | `${TEMPLATE_REPOSITORY_PORT:-3385}` | `${TEMPLATE_REPOSITORY_PORT:-3385}` | `TEMPLATE_REPOSITORY_PORT` | Template management |
| **Free AI Service** | `${FREE_AI_SERVICE_PORT:-3386}` | `${FREE_AI_SERVICE_PORT:-3386}` | `FREE_AI_SERVICE_PORT` | Free AI operations |
| **AI Workers** | `${AI_WORKERS_PORT:-3387}` | `${AI_WORKERS_PORT:-3387}` | `AI_WORKERS_PORT` | Background AI processing |
| **Gemini AI Service** | `${GEMINI_AI_SERVICE_PORT:-3388}` | `${GEMINI_AI_SERVICE_PORT:-3388}` | `GEMINI_AI_SERVICE_PORT` | Google Gemini AI integration |
| **Data Viz Service** | `${DATA_VIZ_SERVICE_PORT:-3389}` | `${DATA_VIZ_SERVICE_PORT:-3389}` | `DATA_VIZ_SERVICE_PORT` | Data visualization |

**Note**: All ports are configured in `ai-microservice/.env`. The values shown are defaults.

## Access Methods

### Production Access (HTTPS)

```bash
# AI Orchestrator
curl https://ai.statex.cz/health
```

### Docker Network Access

```bash
# From within a container on nginx-network
curl http://ai-microservice:${AI_ORCHESTRATOR_PORT:-3380}/health
```

### SSH Access

```bash
# Connect to production server
ssh statex

# Access microservice directory
cd /home/statex/ai-microservice
```

## Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Service Domain - Used by nginx-microservice for auto-registry (required for correct domain detection)
DOMAIN=ai.statex.cz

# Service Name - Used for logging and service identification
SERVICE_NAME=ai-microservice

# Port Configuration
AI_ORCHESTRATOR_PORT=3380
NLP_SERVICE_PORT=3381
# ... etc

# Database (Shared)
DB_HOST=db-server-postgres
DB_PORT=5432
DB_USER=dbadmin
DB_PASSWORD=<password>
DB_NAME=statex_ai

# Redis (Shared)
REDIS_HOST=db-server-redis
REDIS_SERVER_PORT=6379

# Logging Service (Shared)
LOGGING_SERVICE_URL=https://logging.statex.cz

# Notification Service (Shared)
NOTIFICATION_SERVICE_URL=https://notifications.statex.cz

# AI API Keys
OPENROUTER_API_KEY=<key>
GEMINI_API_KEY=<key>
```

## Quick Start

### Start Services

```bash
cd ai-microservice
./scripts/start.sh
```

### Check Status

```bash
./scripts/status.sh
```

### Stop Services

```bash
./scripts/stop.sh
```

### View Logs

```bash
docker compose -f docker-compose.blue.yml logs -f
```

## API Endpoints

### AI Orchestrator

- `POST /api/process-submission` - Process user submission
- `GET /api/status/{submission_id}` - Check processing status
- `GET /api/results/{submission_id}` - Get final results
- `GET /health` - Health check

## Integration

Applications use the AI microservice via HTTP:

```python
# Production
AI_SERVICE_URL=https://ai.statex.cz

# Docker Network
AI_SERVICE_URL=http://ai-microservice:3380
```

## Shared Services

### Database

All AI agents data, workflows, submissions, and related information are stored in the shared database:

- **Database Server**: `db-server-postgres` (Docker network)
- **Database Name**: `statex_ai`
- **Connection**: All services connect to shared `db-server-postgres:5432`

### Logging

All services send logs to the centralized logging microservice:

- **Production URL**: `https://logging.statex.cz`
- **Docker Network URL**: `http://logging-microservice:3367`
- **API Endpoint**: `POST /api/logs`
- **Fallback**: Local log files if logging service unavailable

## Blue/Green Deployment

The microservice supports blue/green deployments:

- **Blue**: `docker-compose.blue.yml`
- **Green**: `docker-compose.green.yml`

Switch between deployments by updating nginx configuration.

## Documentation

- **Migration Plan**: See `AI_MICROSERVICE_MIGRATION_PLAN.md` in project root
- **Main README**: See main `README.md` for ecosystem overview

## Support

For issues or questions:

- Check service logs: `docker compose logs <service-name>`
- Verify network connectivity: `docker network inspect nginx-network`
- Check health endpoints: `curl https://ai.statex.cz/health`

---

**AI Microservice** - Intelligent business solutions powered by AI agents 🚀
