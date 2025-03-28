#!/bin/bash
set -e

# Default values
REGISTRY="localhost:5000"
IMAGE_NAME="entity-sentiment-analyzer"
TAG="latest"
PLATFORMS="linux/amd64,linux/arm64"
RUN_TESTS=true
PUSH_TO_REGISTRY=false
LOCAL_DEPLOY=false

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -r, --registry REGISTRY   Docker registry to push to (default: $REGISTRY)"
    echo "  -i, --image IMAGE_NAME    Image name (default: $IMAGE_NAME)"
    echo "  -t, --tag TAG             Image tag (default: $TAG)"
    echo "  -p, --platforms PLATFORMS Comma-separated list of platforms to build for (default: $PLATFORMS)"
    echo "  --no-tests                Skip running tests before build"
    echo "  --push                    Push the image to the registry after build"
    echo "  --deploy                  Deploy the container locally after build"
    echo ""
    echo "Examples:"
    echo "  $0 --no-tests             Build without running tests"
    echo "  $0 --push                 Build and push to registry"
    echo "  $0 --deploy               Build and deploy locally"
    echo "  $0 --push --deploy        Build, push, and deploy locally"
    echo "  $0 -r myregistry.com -i myapp -t v1.0  Build with custom registry, image name, and tag"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift
            shift
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        -t|--tag)
            TAG="$2"
            shift
            shift
            ;;
        -p|--platforms)
            PLATFORMS="$2"
            shift
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --push)
            PUSH_TO_REGISTRY=true
            shift
            ;;
        --deploy)
            LOCAL_DEPLOY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in the PATH${NC}"
    exit 1
fi

# Check if docker buildx is available
if ! docker buildx version &> /dev/null; then
    echo -e "${YELLOW}Warning: Docker Buildx not available. Multi-architecture builds won't work.${NC}"
    echo -e "${YELLOW}Installing Docker Buildx...${NC}"
    docker buildx create --name multiarch-builder --driver docker-container --use
fi

# Full image name
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo -e "${GREEN}Building image: ${FULL_IMAGE_NAME}${NC}"
echo -e "${GREEN}Platforms: ${PLATFORMS}${NC}"

# Run tests if enabled
if [ "$RUN_TESTS" = true ]; then
    echo -e "${GREEN}Running tests before build...${NC}"
    # Check if pytest is installed
    if ! command -v pytest &> /dev/null; then
        echo -e "${YELLOW}Warning: pytest not found. Installing...${NC}"
        pip install pytest
    fi
    
    # Run the tests
    if pytest -xvs tests/; then
        echo -e "${GREEN}Tests passed!${NC}"
    else
        echo -e "${RED}Tests failed. Aborting build.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping tests as requested.${NC}"
fi

# Build the image
echo -e "${GREEN}Building Docker image...${NC}"
if [ "$PUSH_TO_REGISTRY" = true ]; then
    # If pushing, we need to use buildx with --push
    echo -e "${GREEN}Building and pushing multi-architecture image...${NC}"
    docker buildx build --platform ${PLATFORMS} \
                        --tag ${FULL_IMAGE_NAME} \
                        --push .
else
    # Local build only needs the local architecture
    echo -e "${GREEN}Building local image...${NC}"
    docker build --tag ${FULL_IMAGE_NAME} .
    
    # Save images for each platform if multiple platforms are specified
    if [[ "$PLATFORMS" == *","* ]] && [ "$LOCAL_DEPLOY" = true ]; then
        echo -e "${YELLOW}Multiple platforms specified for local build. Only building for local architecture.${NC}"
    fi
fi

# Deploy locally if requested
if [ "$LOCAL_DEPLOY" = true ]; then
    echo -e "${GREEN}Deploying container locally...${NC}"
    
    # Check if a container with the same name is already running
    if docker ps -a --format '{{.Names}}' | grep -q "^${IMAGE_NAME}$"; then
        echo -e "${YELLOW}Container with name ${IMAGE_NAME} already exists. Stopping and removing...${NC}"
        docker stop ${IMAGE_NAME} || true
        docker rm ${IMAGE_NAME} || true
    fi
    
    # Run the container
    echo -e "${GREEN}Starting container...${NC}"
    docker run -d \
        --name ${IMAGE_NAME} \
        -p 8000:8000 \
        -e "MODEL_PATH=/app/models" \
        -e "LOG_LEVEL=info" \
        ${FULL_IMAGE_NAME}
    
    echo -e "${GREEN}Container is running. API available at: http://localhost:8000${NC}"
    echo -e "${GREEN}API documentation at: http://localhost:8000/docs${NC}"
else
    echo -e "${GREEN}Skipping local deployment.${NC}"
fi

echo -e "${GREEN}Build process completed successfully!${NC}"

