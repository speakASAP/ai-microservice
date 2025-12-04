# AI Workers Service

## Overview

The AI Workers Service consists of specialized AI processing agents that perform specific tasks such as text analysis, audio processing, document analysis, and multi-modal content processing. These workers are designed to be scalable, fault-tolerant, and easily extensible for new AI capabilities.

## Purpose

- **AI Processing**: Execute specific AI tasks and analysis
- **Scalable Workers**: Horizontal scaling of AI processing capacity
- **Task Specialization**: Specialized workers for different AI tasks
- **Result Generation**: Produce structured results for user consumption
- **Model Management**: Manage AI models and their lifecycle
- **Performance Optimization**: Optimize AI processing performance

## Architecture

### Technology Stack

- **Framework**: FastAPI with Python 3.13
- **AI Libraries**: Transformers, OpenAI, Anthropic, local models
- **Message Broker**: RabbitMQ or NATS for task communication
- **Database**: PostgreSQL for worker state and results
- **Cache**: Redis for model caching and temporary data

### Worker Types

- **Text Analysis Workers**: Sentiment analysis, summarization, translation
- **Audio Processing Workers**: Speech-to-text, audio analysis, transcription
- **Document Processing Workers**: PDF analysis, document summarization
- **Multi-modal Workers**: Combined text, audio, and document analysis
- **Custom Workers**: User-defined AI processing tasks

## Worker Categories

### Text Analysis Workers

#### Sentiment Analysis Worker

```python
class SentimentAnalysisWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.model = load_sentiment_model()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        text = task.input_data.get('text')
        result = self.model.predict(text)
        return {
            'sentiment': result.sentiment,
            'confidence': result.confidence,
            'emotions': result.emotions
        }
```

#### Text Summarization Worker

```python
class TextSummarizationWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.model = load_summarization_model()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        text = task.input_data.get('text')
        max_length = task.parameters.get('max_length', 150)
        summary = self.model.summarize(text, max_length=max_length)
        return {
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': len(summary) / len(text)
        }
```

#### Translation Worker

```python
class TranslationWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.model = load_translation_model()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        text = task.input_data.get('text')
        source_lang = task.parameters.get('source_lang', 'auto')
        target_lang = task.parameters.get('target_lang', 'en')
        translation = self.model.translate(text, source_lang, target_lang)
        return {
            'translation': translation,
            'source_language': source_lang,
            'target_language': target_lang,
            'confidence': translation.confidence
        }
```

### Audio Processing Workers

#### Speech-to-Text Worker

```python
class SpeechToTextWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.model = load_whisper_model()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        audio_file = task.input_data.get('audio_file')
        language = task.parameters.get('language', 'auto')
        transcription = self.model.transcribe(audio_file, language=language)
        return {
            'transcription': transcription.text,
            'language': transcription.language,
            'confidence': transcription.confidence,
            'segments': transcription.segments
        }
```

#### Audio Analysis Worker

```python
class AudioAnalysisWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.model = load_audio_analysis_model()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        audio_file = task.input_data.get('audio_file')
        analysis = self.model.analyze(audio_file)
        return {
            'duration': analysis.duration,
            'sample_rate': analysis.sample_rate,
            'channels': analysis.channels,
            'loudness': analysis.loudness,
            'spectral_features': analysis.spectral_features
        }
```

### Document Processing Workers

#### PDF Analysis Worker

```python
class PDFAnalysisWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.pdf_processor = PDFProcessor()
        self.text_analyzer = TextAnalyzer()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        pdf_file = task.input_data.get('pdf_file')
        pages = self.pdf_processor.extract_pages(pdf_file)
        text = self.pdf_processor.extract_text(pdf_file)
        analysis = self.text_analyzer.analyze(text)
        return {
            'page_count': len(pages),
            'text_length': len(text),
            'analysis': analysis,
            'metadata': self.pdf_processor.get_metadata(pdf_file)
        }
```

#### Document Summarization Worker

```python
class DocumentSummarizationWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.document_processor = DocumentProcessor()
        self.summarizer = DocumentSummarizer()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        document_file = task.input_data.get('document_file')
        document_type = task.parameters.get('document_type', 'auto')
        summary_length = task.parameters.get('summary_length', 'medium')
        
        document = self.document_processor.process(document_file, document_type)
        summary = self.summarizer.summarize(document, summary_length)
        
        return {
            'summary': summary.text,
            'key_points': summary.key_points,
            'document_type': document_type,
            'original_length': len(document.text),
            'summary_length': len(summary.text)
        }
```

### Multi-modal Workers

#### Multi-modal Analysis Worker

```python
class MultiModalAnalysisWorker(BaseWorker):
    def __init__(self):
        super().__init__()
        self.text_analyzer = TextAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.document_analyzer = DocumentAnalyzer()
        self.fusion_model = MultiModalFusionModel()
    
    async def process(self, task: Task) -> Dict[str, Any]:
        text_data = task.input_data.get('text')
        audio_data = task.input_data.get('audio')
        document_data = task.input_data.get('document')
        
        # Process each modality
        text_analysis = self.text_analyzer.analyze(text_data) if text_data else None
        audio_analysis = self.audio_analyzer.analyze(audio_data) if audio_data else None
        document_analysis = self.document_analyzer.analyze(document_data) if document_data else None
        
        # Fuse results
        fused_result = self.fusion_model.fuse(
            text_analysis, audio_analysis, document_analysis
        )
        
        return {
            'fused_analysis': fused_result,
            'modalities': {
                'text': text_analysis,
                'audio': audio_analysis,
                'document': document_analysis
            },
            'confidence': fused_result.confidence
        }
```

## Base Worker Framework

### Base Worker Class

```python
class BaseWorker:
    def __init__(self, worker_id: str, capabilities: List[str]):
        self.worker_id = worker_id
        self.capabilities = capabilities
        self.status = WorkerStatus.IDLE
        self.current_task = None
        self.performance_metrics = PerformanceMetrics()
    
    async def start(self):
        """Start the worker and begin listening for tasks"""
        self.status = WorkerStatus.RUNNING
        await self._register_with_orchestrator()
        await self._listen_for_tasks()
    
    async def stop(self):
        """Stop the worker gracefully"""
        self.status = WorkerStatus.STOPPING
        if self.current_task:
            await self._cancel_current_task()
        await self._unregister_from_orchestrator()
        self.status = WorkerStatus.STOPPED
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task and return results"""
        self.status = WorkerStatus.PROCESSING
        self.current_task = task
        
        try:
            start_time = time.time()
            result = await self.process(task)
            processing_time = time.time() - start_time
            
            self.performance_metrics.record_success(processing_time)
            return result
            
        except Exception as e:
            self.performance_metrics.record_error(str(e))
            raise
        finally:
            self.current_task = None
            self.status = WorkerStatus.IDLE
    
    async def process(self, task: Task) -> Dict[str, Any]:
        """Override this method in subclasses"""
        raise NotImplementedError
```

### Worker Registration

```python
class WorkerRegistry:
    def __init__(self, RABBITMQ_URL: str):
        self.broker = MessageBroker(RABBITMQ_URL)
        self.workers = {}
    
    async def register_worker(self, worker: BaseWorker):
        """Register a worker with the orchestrator"""
        registration_data = {
            'worker_id': worker.worker_id,
            'capabilities': worker.capabilities,
            'status': worker.status.value,
            'performance_metrics': worker.performance_metrics.to_dict()
        }
        await self.broker.publish('worker.registered', registration_data)
        self.workers[worker.worker_id] = worker
    
    async def unregister_worker(self, worker_id: str):
        """Unregister a worker from the orchestrator"""
        await self.broker.publish('worker.unregistered', {'worker_id': worker_id})
        if worker_id in self.workers:
            del self.workers[worker_id]
```

## Task Processing

### Task Model

```python
class Task(BaseModel):
    id: UUID
    job_id: UUID
    worker_id: UUID
    task_type: str
    status: TaskStatus
    input_data: Dict[str, Any]
    parameters: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
```

### Task Processing Pipeline

1. **Task Reception**: Receive task from orchestrator
2. **Validation**: Validate task data and parameters
3. **Preprocessing**: Prepare data for AI processing
4. **AI Processing**: Execute AI model inference
5. **Postprocessing**: Format and validate results
6. **Result Submission**: Send results back to orchestrator
7. **Cleanup**: Clean up temporary resources

## Model Management

### Model Loading

```python
class ModelManager:
    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = model_cache_dir
        self.loaded_models = {}
    
    async def load_model(self, model_name: str, model_type: str):
        """Load a model into memory"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = os.path.join(self.model_cache_dir, model_name)
        model = await self._load_model_from_path(model_path, model_type)
        self.loaded_models[model_name] = model
        return model
    
    async def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()  # Force garbage collection
```

### Model Caching

```python
class ModelCache:
    def __init__(self, redis_client, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    async def get_model(self, model_name: str):
        """Get model from cache"""
        cached_model = await self.redis.get(f"model:{model_name}")
        if cached_model:
            return pickle.loads(cached_model)
        return None
    
    async def set_model(self, model_name: str, model):
        """Cache model"""
        serialized_model = pickle.dumps(model)
        await self.redis.setex(
            f"model:{model_name}", 
            self.ttl, 
            serialized_model
        )
```

## Configuration

### Environment Variables

```bash
# Service Configuration
SERVICE_NAME=ai-workers
WORKER_ID=worker-001
WORKER_TYPE=text_analysis
LOG_LEVEL=INFO
ENVIRONMENT=production

# Message Broker
# Port configured in statex/.env: RABBITMQ_PORT (default: 5672)
RABBITMQ_URL=amqp://user:password@localhost:${RABBITMQ_PORT:-5672}
BROKER_TOPIC=ai-tasks
BROKER_RETRY_ATTEMPTS=3
BROKER_RETRY_DELAY=1000

# Database
# Port configured in database-server/.env: DB_SERVER_PORT (default: 5432)
DATABASE_URL=postgresql://user:password@localhost:${DB_SERVER_PORT:-5432}/statex_ai
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Cache
# Port configured in database-server/.env: REDIS_SERVER_PORT (default: 6379)
REDIS_URL=redis://localhost:${REDIS_SERVER_PORT:-6379}/2
REDIS_POOL_SIZE=5

# Model Management
MODEL_CACHE_DIR=/app/models
MODEL_DOWNLOAD_URL=https://models.statex.cz
MODEL_CACHE_TTL=3600
MAX_MODEL_SIZE=5000000000  # 5GB

# Performance
MAX_CONCURRENT_TASKS=5
TASK_TIMEOUT=300
MODEL_LOAD_TIMEOUT=60
RESULT_CACHE_TTL=1800

# Monitoring
HEALTH_CHECK_INTERVAL=30
METRICS_UPDATE_INTERVAL=10

# AI Models
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
HUGGINGFACE_TOKEN=your-huggingface-token
```

## Deployment

### Docker Configuration

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model cache directory
RUN mkdir -p /app/models && chmod 755 /app/models

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Start application
CMD ["python", "-m", "app.main"]
```


## Observability

### Health Checks

- **Liveness Probe**: `/health` - Basic service health
- **Readiness Probe**: `/health/ready` - Service ready to accept tasks
- **Dependency Checks**: Database, Redis, message broker connectivity

### Metrics

- **Task Metrics**: Task count, duration, success rate
- **Model Metrics**: Model load time, inference time, memory usage
- **Performance Metrics**: CPU usage, memory usage, GPU usage
- **Error Metrics**: Error rates, retry counts, failure types

### Logging

- **Structured Logging**: JSON-formatted logs
- **Task Logging**: Log all task processing events
- **Model Logging**: Log model loading and inference
- **Error Logging**: Log all errors and exceptions

### Tracing

- **Distributed Tracing**: OpenTelemetry integration
- **Task Tracing**: Trace task processing pipeline
- **Model Tracing**: Trace model inference
- **Performance Tracing**: Trace performance bottlenecks

## Performance Optimization

### Model Optimization

- **Model Quantization**: Reduce model size and memory usage
- **Model Pruning**: Remove unnecessary model parameters
- **Batch Processing**: Process multiple tasks in batches
- **Model Caching**: Cache frequently used models

### Resource Management

- **Memory Management**: Efficient memory usage and cleanup
- **GPU Utilization**: Optimize GPU usage for AI models
- **CPU Optimization**: Optimize CPU usage for non-GPU tasks
- **Resource Monitoring**: Monitor and optimize resource usage

### Caching Strategies

- **Result Caching**: Cache processing results
- **Model Caching**: Cache loaded models
- **Data Caching**: Cache frequently used data
- **Configuration Caching**: Cache configuration data

## Testing

### Test Categories

- **Unit Tests**: Test individual worker components
- **Integration Tests**: Test worker interactions
- **Model Tests**: Test AI model functionality
- **Performance Tests**: Test under load

### Test Coverage

- **Code Coverage**: Minimum 80% code coverage
- **Model Coverage**: Test all AI models
- **Task Coverage**: Test all task types
- **Error Handling**: Test error scenarios

### Test Data

- **Mock Tasks**: Test task data for different scenarios
- **Mock Models**: Mock AI model responses
- **Test Files**: Test files for different formats
- **Performance Data**: Performance test data

## Troubleshooting

### Common Issues

- **Model Loading**: Check model files and dependencies
- **Memory Issues**: Check memory usage and limits
- **Task Timeout**: Check task timeout settings
- **Model Errors**: Check model compatibility and versions

### Debug Mode

- **Verbose Logging**: Enable detailed logging
- **Task Debug**: Debug task processing
- **Model Debug**: Debug model inference
- **Performance Debug**: Debug performance issues

## Future Enhancements

### Planned Features

- **Custom Models**: User-defined AI models
- **Model Training**: On-the-fly model training
- **Advanced Caching**: Multi-level caching strategies
- **Auto-scaling**: Automatic worker scaling

### Technical Improvements

- **GPU Support**: Enhanced GPU support
- **Distributed Processing**: Distributed AI processing
- **Model Optimization**: Advanced model optimization
- **Performance Monitoring**: Advanced performance monitoring
