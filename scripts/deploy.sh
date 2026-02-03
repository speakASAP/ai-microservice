#!/bin/bash

# AI Microservice Deployment Script
# Deploys the AI microservice using external nginx microservice blue-green deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NGINX_MICROSERVICE_DIR="$(cd "$PROJECT_DIR/../nginx-microservice" && pwd)"

cd "$PROJECT_DIR"

echo "🚀 Deploying AI Microservice"
echo "=================================="

# Check if nginx-microservice directory exists
if [ ! -d "$NGINX_MICROSERVICE_DIR" ]; then
  echo "❌ Error: nginx-microservice directory not found at $NGINX_MICROSERVICE_DIR"
  echo "Please ensure nginx-microservice is located at ../nginx-microservice"
  exit 1
fi

# Check if deployment script exists
if [ ! -f "$NGINX_MICROSERVICE_DIR/scripts/blue-green/deploy-smart.sh" ]; then
  echo "❌ Error: Deployment script not found at $NGINX_MICROSERVICE_DIR/scripts/blue-green/deploy-smart.sh"
  exit 1
fi

echo ""
# Deploy only code from repository: sync with remote (discard local changes on server)
if [ -d ".git" ]; then
    echo "Syncing with remote repository..."
    git fetch origin
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git reset --hard "origin/$BRANCH"
    echo "✓ Repository synced to origin/$BRANCH"
    echo ""
fi

echo "Deploying via nginx microservice..."
echo ""

# Deploy using nginx microservice blue-green deployment
cd "$NGINX_MICROSERVICE_DIR" && ./scripts/blue-green/deploy-smart.sh ai-microservice

echo ""
echo "✅ Deployment completed"
