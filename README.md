# Entity-Level Sentiment Analysis API

A RESTful service that performs entity-level sentiment analysis on text, identifying specific sentiments associated with entities mentioned in the text.

## Features

- Entity recognition using spaCy
- Sentiment analysis using BERT
- Negation and contrasting sentiment handling
- Support for multi-paragraph text
- REST API with FastAPI
- Docker support with multi-architecture builds

## Installation

### Local Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Start the service:
```bash
uvicorn api:app --reload
```

3. Access the API documentation at:
```
http://localhost:8000/docs
```

### Docker Installation

#### Prerequisites
- Docker
- Docker Compose (optional, for easy deployment)

## Docker Usage

The service can be run using Docker:

```bash
# Build the Docker image
docker build -t entity-sentiment-api .

# Run the container (basic usage)
docker run -d -p 8000:8000 entity-sentiment-api

# Run with environment variables
docker run -d -p 8000:8000 \
  -e MODEL_CACHE_DIR=/app/models \
  -e LOG_LEVEL=INFO \
  -e MAX_BATCH_SIZE=50 \
  -v model-cache:/app/models \
  entity-sentiment-api
```

Note: Allow about 15-20 seconds for initial startup while the models are loaded.
The service will be available at `http://localhost:8000`

### Health Check

The container includes a health check endpoint. You can verify the service is running properly with:

```bash
curl http://localhost:8000/health
```

### Multi-architecture Build

To build for both Intel (amd64) and ARM (arm64) architectures:

1. Set up Docker BuildX (if not already configured):
```bash
docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap
```

2. Build and push to a registry:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/entity-sentiment-api:latest --push .
```

Or build locally without pushing:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t entity-sentiment-api:latest --load .
```

### Using Docker Compose

1. Start the service:
```bash
docker-compose up -d
```

2. View logs:
```bash
docker-compose logs -f
```

3. Stop the service:
```bash
docker-compose down
```

## Configuration

The API can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| MODEL_CACHE_DIR | Directory to cache models | /app/models |
| LOG_LEVEL | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| MAX_BATCH_SIZE | Maximum batch size for processing | 50 |
| WORKERS | Number of worker processes | 1 |
| HOST | Binding host | 0.0.0.0 |
| PORT | Binding port | 8000 |

## Deployment

### Kubernetes Deployment

1. Create a deployment:
```bash
kubectl apply -f k8s/deployment.yaml
```

2. Create a service:
```bash
kubectl apply -f k8s/service.yaml
```

3. Monitor deployment:
```bash
kubectl get pods -l app=entity-sentiment-api
```

### Cloud Deployments

#### AWS
Deploy with AWS ECS or EKS:
```bash
# Example AWS CLI command to create ECS service
aws ecs create-service --cluster your-cluster --service-name entity-sentiment-api --task-definition entity-sentiment:1 --desired-count 2
```

#### Google Cloud
Deploy with Google Cloud Run:
```bash
gcloud run deploy entity-sentiment-api --image gcr.io/your-project/entity-sentiment-api --platform managed
```

#### Azure
Deploy with Azure Container Instances:
```bash
az container create --resource-group myResourceGroup --name entity-sentiment-api --image yourusername/entity-sentiment-api:latest --ports 8000 --dns-name-label entity-sentiment-api
```

## API Usage

### Analyze Text

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Chris hates Android phones but loves iPhones."}'
```

Response:
```json
{
  "sentences": [
    {
      "text": "Chris hates Android phones but loves iPhones.",
      "entities": [
        {
          "entity": "Chris",
          "sentiment": "mixed",
          "score": 0.15,
          "details": "Subject of contrasting sentiment verbs"
        },
        {
          "entity": "Android phones",
          "sentiment": "very negative",
          "score": -0.25,
          "details": "Object of negative sentiment verb 'hates'"
        },
        {
          "entity": "iPhones",
          "sentiment": "very positive",
          "score": 0.97,
          "details": "Object of positive sentiment verb 'loves'"
        }
      ]
    }
  ]
}
```

### Batch Process

```bash
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Chris hates Android phones but loves iPhones.", "Microsoft Windows has some bugs but Apple MacOS isn't perfect either."]}'
```

## Performance Considerations

- The first request may be slow due to model loading (allow 15-20 seconds after startup)
- For production, use a persistent volume to cache models
- Consider increasing worker count for higher throughput

## License

This project is licensed under the Apache License 2.0. Copyright 2025 Chris Collins <chris@hitorro.com>. See the LICENSE file for details.

## Technical Stack

The service uses:
- Hugging Face's BERT model for sentiment analysis
- spaCy for named entity recognition
- FastAPI for the REST API framework
- PyTorch for the machine learning components
- Python 3.10 or higher
