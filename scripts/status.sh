#!/bin/bash

# AI Microservice Status Script
# Checks the status of all AI microservice containers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

echo "=========================================="
echo "AI Microservice Status"
echo "=========================================="

# Load ports from .env if available
if [ -f .env ]; then
  source .env
fi

AI_ORCHESTRATOR_PORT=${AI_ORCHESTRATOR_PORT:-3380}

# Check if containers are running
echo ""
echo "📋 Container Status:"
if docker ps --format '{{.Names}}' | grep -q "ai-microservice"; then
  echo "✅ AI Microservice containers are running"
  docker ps --filter "name=ai-microservice" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
  echo "❌ No AI Microservice containers are running"
fi

# Check network
echo ""
echo "🌐 Network Status:"
if docker network inspect nginx-network >/dev/null 2>&1; then
  if docker network inspect nginx-network | grep -q "ai-microservice"; then
    echo "✅ Connected to nginx-network"
  else
    echo "⚠️  Not connected to nginx-network"
  fi
else
  echo "⚠️  nginx-network not found"
fi

# Check AI Orchestrator health
echo ""
echo "🏥 AI Orchestrator Health Check:"
if docker ps --format '{{.Names}}' | grep -q "ai-microservice-orchestrator"; then
  ORCHESTRATOR_CONTAINER=$(docker ps --format '{{.Names}}' | grep "ai-microservice-orchestrator" | head -1)
  if docker exec "$ORCHESTRATOR_CONTAINER" curl -f "http://localhost:${AI_ORCHESTRATOR_PORT}/health" >/dev/null 2>&1; then
    echo "✅ AI Orchestrator health check passed"
    docker exec "$ORCHESTRATOR_CONTAINER" curl -s "http://localhost:${AI_ORCHESTRATOR_PORT}/health" | jq . 2>/dev/null || docker exec "$ORCHESTRATOR_CONTAINER" curl -s "http://localhost:${AI_ORCHESTRATOR_PORT}/health"
  else
    echo "⚠️  AI Orchestrator health check failed"
  fi
else
  echo "❌ AI Orchestrator container not running"
fi

# Show recent logs
echo ""
echo "📝 Recent Logs (AI Orchestrator, last 20 lines):"
if docker ps --format '{{.Names}}' | grep -q "ai-microservice-orchestrator"; then
  ORCHESTRATOR_CONTAINER=$(docker ps --format '{{.Names}}' | grep "ai-microservice-orchestrator" | head -1)
  docker logs --tail=20 "$ORCHESTRATOR_CONTAINER"
else
  echo "No logs available (container not running)"
fi

echo ""
echo "=========================================="

