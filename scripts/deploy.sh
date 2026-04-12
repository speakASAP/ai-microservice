#!/bin/bash
# AI Microservice Deployment Script
# Deploys the AI microservice using nginx-microservice blue/green deployment.
#
# Uses: nginx-microservice/scripts/blue-green/deploy-smart.sh
# The script automatically detects the nginx-microservice location and
# calls the deploy-smart.sh script to perform the deployment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load .env to determine environment
NODE_ENV=""
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/.env" 2>/dev/null || true
    set +a
    NODE_ENV="${NODE_ENV:-}"
fi

# Pull from remote in production; preserve local changes (stash uncommitted if any, then reapply).
# Only sync if NODE_ENV is set to "production"
if [ -d ".git" ]; then
    if [ "$NODE_ENV" = "production" ]; then
        echo -e "${BLUE}Production environment detected (NODE_ENV=production)${NC}"
        echo -e "${BLUE}Pulling from remote (local changes preserved)...${NC}"
        git fetch origin
        BRANCH=$(git rev-parse --abbrev-ref HEAD)
        STASHED=0
        if [ -n "$(git status --porcelain)" ]; then
            git stash push -u -m "deploy.sh: stash before pull"
            STASHED=1
        fi
        git pull origin "$BRANCH"
        if [ "$STASHED" = "1" ]; then
            git stash pop
        fi
        echo -e "${GREEN}✓ Repository updated from origin/$BRANCH (local changes preserved)${NC}"
        echo ""
    else
        echo -e "${YELLOW}Development environment detected (NODE_ENV=${NODE_ENV:-not set})${NC}"
        echo -e "${YELLOW}Skipping git sync - local changes will be preserved${NC}"
        echo ""
    fi
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          AI Microservice - Production Deployment           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

SERVICE_NAME="ai-microservice"

# Detect nginx-microservice path
NGINX_MICROSERVICE_PATH=""
if [ -d "/home/statex/nginx-microservice" ]; then
    NGINX_MICROSERVICE_PATH="/home/statex/nginx-microservice"
elif [ -d "/home/alfares/nginx-microservice" ]; then
    NGINX_MICROSERVICE_PATH="/home/alfares/nginx-microservice"
elif [ -d "/home/belunga/nginx-microservice" ]; then
    NGINX_MICROSERVICE_PATH="/home/belunga/nginx-microservice"
elif [ -d "$HOME/nginx-microservice" ]; then
    NGINX_MICROSERVICE_PATH="$HOME/nginx-microservice"
elif [ -d "$(dirname "$PROJECT_ROOT")/nginx-microservice" ]; then
    NGINX_MICROSERVICE_PATH="$(dirname "$PROJECT_ROOT")/nginx-microservice"
elif [ -d "$PROJECT_ROOT/../nginx-microservice" ]; then
    NGINX_MICROSERVICE_PATH="$(cd "$PROJECT_ROOT/../nginx-microservice" && pwd)"
fi

if [ -z "$NGINX_MICROSERVICE_PATH" ] || [ ! -d "$NGINX_MICROSERVICE_PATH" ]; then
    echo -e "${RED}❌ Error: nginx-microservice not found${NC}"
    echo ""
    echo "Please ensure nginx-microservice is installed in one of these locations:"
    echo "  - /home/statex/nginx-microservice"
    echo "  - /home/alfares/nginx-microservice"
    echo "  - /home/belunga/nginx-microservice"
    echo "  - $HOME/nginx-microservice"
    echo "  - $(dirname "$PROJECT_ROOT")/nginx-microservice (sibling directory)"
    echo ""
    echo "Or set NGINX_MICROSERVICE_PATH environment variable:"
    echo "  export NGINX_MICROSERVICE_PATH=/path/to/nginx-microservice"
    exit 1
fi

DEPLOY_SCRIPT="$NGINX_MICROSERVICE_PATH/scripts/blue-green/deploy-smart.sh"
if [ ! -f "$DEPLOY_SCRIPT" ]; then
    echo -e "${RED}❌ Error: deploy-smart.sh not found at $DEPLOY_SCRIPT${NC}"
    exit 1
fi

if [ ! -x "$DEPLOY_SCRIPT" ]; then
    echo -e "${YELLOW}⚠️  Making deploy-smart.sh executable...${NC}"
    chmod +x "$DEPLOY_SCRIPT"
fi

echo -e "${GREEN}✅ Found nginx-microservice at: $NGINX_MICROSERVICE_PATH${NC}"
echo -e "${GREEN}✅ Deploying service: $SERVICE_NAME${NC}"
echo ""

# Timing and phase summary
get_timestamp_seconds() { date +%s.%N; }
PHASE_TIMING_FILE=$(mktemp /tmp/deploy-phases-XXXXXX)
trap "rm -f $PHASE_TIMING_FILE" EXIT
start_phase() {
    local phase_name="$1"
    local timestamp=$(get_timestamp_seconds)
    echo "$phase_name|START|$timestamp" >> "$PHASE_TIMING_FILE"
    echo -e "${YELLOW}⏱️  PHASE START: $phase_name${NC}"
}
end_phase() {
    local phase_name="$1"
    local timestamp=$(get_timestamp_seconds)
    echo "$phase_name|END|$timestamp" >> "$PHASE_TIMING_FILE"
    local start_line=$(grep "^${phase_name}|START|" "$PHASE_TIMING_FILE" | tail -1)
    if [ -n "$start_line" ]; then
        local start_time=$(echo "$start_line" | cut -d'|' -f3)
        local duration=$(awk "BEGIN {printf \"%.2f\", $timestamp - $start_time}")
        echo -e "${GREEN}⏱️  PHASE END: $phase_name (duration: ${duration}s)${NC}"
    fi
}
print_phase_summary() {
    if [ ! -f "$PHASE_TIMING_FILE" ] || [ ! -s "$PHASE_TIMING_FILE" ]; then echo ""; echo -e "${YELLOW}⚠️  No phase timing data available${NC}"; echo ""; return; fi
    echo ""; echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📊 DEPLOYMENT PHASE TIMING SUMMARY${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    local current_phase="" start_time="" total_phase_time=0
    while IFS='|' read -r phase_name event timestamp; do
        if [ "$event" = "START" ]; then current_phase="$phase_name"; start_time="$timestamp"
        elif [ "$event" = "END" ] && [ -n "$start_time" ] && [ -n "$current_phase" ]; then
            local duration=$(awk "BEGIN {printf \"%.2f\", $timestamp - $start_time}")
            total_phase_time=$(awk "BEGIN {printf \"%.2f\", $total_phase_time + $duration}")
            printf "  ${GREEN}%-45s${NC} ${YELLOW}%10.2fs${NC}\n" "$phase_name:" "$duration"
            current_phase=""; start_time=""
        fi
    done < "$PHASE_TIMING_FILE"
    if [ "$(echo "$total_phase_time > 0" | bc 2>/dev/null || echo "0")" = "1" ]; then
        echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
        printf "  ${GREEN}%-45s${NC} ${YELLOW}%10.2fs${NC}\n" "Total (all phases):" "$total_phase_time"
    fi
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"; echo ""
}

start_phase "Pre-deployment Setup"
echo -e "${YELLOW}Starting blue/green deployment...${NC}"
echo ""

cd "$NGINX_MICROSERVICE_PATH"
end_phase "Pre-deployment Setup"

START_TIME=$(get_timestamp_seconds)
"$DEPLOY_SCRIPT" "$SERVICE_NAME" 2>&1 | {
    build_started=0; start_containers_started=0; health_check_started=0
    while IFS= read -r line; do
        echo "$line"
        if echo "$line" | grep -qE "Phase 0:.*Infrastructure"; then start_phase "Phase 0: Infrastructure Check"
        elif echo "$line" | grep -qE "Phase 0 completed|✅ Phase 0 completed"; then end_phase "Phase 0: Infrastructure Check"
        elif echo "$line" | grep -qE "Phase 1:.*Preparing|Phase 1:.*Prepare"; then start_phase "Phase 1: Prepare Green Deployment"
        elif echo "$line" | grep -qE "Phase 1 completed|✅ Phase 1 completed"; then end_phase "Phase 1: Prepare Green Deployment"
        elif echo "$line" | grep -qE "Phase 2:.*Switching|Phase 2:.*Switch"; then start_phase "Phase 2: Switch Traffic to Green"
        elif echo "$line" | grep -qE "Phase 2 completed|✅ Phase 2 completed"; then end_phase "Phase 2: Switch Traffic to Green"
        elif echo "$line" | grep -qE "Phase 3:.*Monitoring|Phase 3:.*Monitor"; then start_phase "Phase 3: Monitor Health"
        elif echo "$line" | grep -qE "Phase 3 completed|✅ Phase 3 completed"; then end_phase "Phase 3: Monitor Health"
        elif echo "$line" | grep -qE "Phase 4:.*Verifying|Phase 4:.*Verify"; then start_phase "Phase 4: Verify HTTPS"
        elif echo "$line" | grep -qE "Phase 4 completed|✅ Phase 4 completed"; then end_phase "Phase 4: Verify HTTPS"
        elif echo "$line" | grep -qE "Phase 5:.*Cleaning|Phase 5:.*Cleanup"; then start_phase "Phase 5: Cleanup"
        elif echo "$line" | grep -qE "Phase 5 completed|✅ Phase 5 completed"; then end_phase "Phase 5: Cleanup"
        elif echo "$line" | grep -qE "Building containers|Image.*Building" && [ "$build_started" -eq 0 ]; then start_phase "Build Containers"; build_started=1
        elif echo "$line" | grep -qE "All services built|✅ All services built" && [ "$build_started" -eq 1 ]; then end_phase "Build Containers"; build_started=2
        elif echo "$line" | grep -qE "Starting containers|Container.*Starting" && [ "$start_containers_started" -eq 0 ]; then start_phase "Start Containers"; start_containers_started=1
        elif echo "$line" | grep -qE "Container.*Started|Waiting.*services to start" && [ "$start_containers_started" -eq 1 ]; then end_phase "Start Containers"; start_containers_started=2
        elif echo "$line" | grep -qE "Checking.*health|Health check" && [ "$health_check_started" -eq 0 ]; then start_phase "Health Checks"; health_check_started=1
        elif echo "$line" | grep -qE "health check passed|✅.*health" && [ "$health_check_started" -eq 1 ]; then end_phase "Health Checks"; health_check_started=2
        fi
    done
}
DEPLOY_EXIT_CODE=${PIPESTATUS[0]:-1}
END_TIME=$(get_timestamp_seconds)
TOTAL_DURATION=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")

if [ "${DEPLOY_EXIT_CODE}" -eq 0 ]; then
    TOTAL_DURATION_FORMATTED=$(awk "BEGIN {printf \"%.2f\", $TOTAL_DURATION}")
    print_phase_summary 2>&1
    echo ""
    echo -e "${BLUE}Running post-deploy tests (health + email-triage full pipeline)...${NC}"
    cd "$PROJECT_ROOT"
    TEST_EXIT=0
    if [ -n "${DOMAIN:-}" ]; then
        export AI_ORCHESTRATOR_BASE_URL="https://${DOMAIN}"
        echo "  Using AI_ORCHESTRATOR_BASE_URL=$AI_ORCHESTRATOR_BASE_URL"
    fi
    if python3 scripts/test-ai-services.py; then
        echo -e "${GREEN}✓ All tests passed (public URL).${NC}"
    else
        # Public URL may return 403 (firewall/WAF). Verify service via localhost.
        LOCAL_URL="http://localhost:${AI_ORCHESTRATOR_PORT:-3380}"
        echo -e "${YELLOW}  Public URL failed; verifying orchestrator at $LOCAL_URL ...${NC}"
        export AI_ORCHESTRATOR_BASE_URL="$LOCAL_URL"
        if python3 scripts/test-ai-services.py; then
            echo -e "${GREEN}✓ All tests passed at $LOCAL_URL (service is healthy).${NC}"
            if [ -n "${DOMAIN:-}" ]; then
                echo -e "${YELLOW}  Note: Public URL https://${DOMAIN} returned an error (e.g. 403). Check firewall/WAF if external access is required.${NC}"
            fi
        else
            TEST_EXIT=1
            echo -e "${RED}✗ One or more tests failed (public URL and localhost).${NC}"
        fi
    fi
    # Verify free-ai-service is running and reachable from ai-orchestrator (same Docker network)
    echo -e "${BLUE}Verifying free-ai-service reachable from ai-orchestrator (FREE_AI_SERVICE_URL)...${NC}"
    ORCHESTRATOR_CONTAINER=""
    for c in ai-microservice-orchestrator-green ai-microservice-orchestrator-blue; do
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${c}$"; then
            ORCHESTRATOR_CONTAINER="$c"
            break
        fi
    done
    if [ -n "$ORCHESTRATOR_CONTAINER" ]; then
        FREE_AI_URL=$(docker exec "$ORCHESTRATOR_CONTAINER" printenv FREE_AI_SERVICE_URL 2>/dev/null || true)
        if [ -z "$FREE_AI_URL" ]; then
            echo -e "${YELLOW}  Warning: FREE_AI_SERVICE_URL not set in $ORCHESTRATOR_CONTAINER (set in .env or docker-compose for email-triage LLM).${NC}"
            FREE_AI_URL="http://ai-microservice-free-ai-service-green:3386"
        fi
        HEALTH_URL="${FREE_AI_URL%/}/health"
        if docker exec "$ORCHESTRATOR_CONTAINER" curl -sf --connect-timeout 5 --max-time 10 "$HEALTH_URL" 2>/dev/null | grep -q "healthy"; then
            echo -e "${GREEN}✓ Verified: $ORCHESTRATOR_CONTAINER can reach free-ai-service at $FREE_AI_URL${NC}"
        else
            echo -e "${YELLOW}  Warning: $ORCHESTRATOR_CONTAINER could not reach $HEALTH_URL (check free-ai-service container, OPENROUTER_API_KEY in .env, and network).${NC}"
        fi
    else
        echo -e "${YELLOW}  Skip: ai-microservice orchestrator container not found.${NC}"
    fi

    echo -e "${BLUE}Verifying LiteLLM reachable from ai-orchestrator (LITELLM_BASE_URL)...${NC}"
    if [ -n "${ORCHESTRATOR_CONTAINER:-}" ]; then
        LM_BASE=$(docker exec "$ORCHESTRATOR_CONTAINER" printenv LITELLM_BASE_URL 2>/dev/null || true)
        if [ -z "$LM_BASE" ]; then
            echo -e "${YELLOW}  Warning: LITELLM_BASE_URL not set in $ORCHESTRATOR_CONTAINER (LiteLLM path disabled).${NC}"
        else
            LM_CODE=$(docker exec "$ORCHESTRATOR_CONTAINER" sh -c 'curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 12 -H "Authorization: Bearer $(printenv LITELLM_MASTER_KEY)" "${LITELLM_BASE_URL%/}/health"' 2>/dev/null || echo "000")
            if [ "$LM_CODE" = "200" ]; then
                echo -e "${GREEN}✓ Verified: $ORCHESTRATOR_CONTAINER can reach LiteLLM (HTTP 200 /health with master key) at $LM_BASE${NC}"
            else
                echo -e "${YELLOW}  Warning: LiteLLM /health from $ORCHESTRATOR_CONTAINER returned HTTP ${LM_CODE} (expect 200; check litellm container and LITELLM_MASTER_KEY).${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}  Skip: LiteLLM check (orchestrator container not found).${NC}"
    fi

    # Verify same-server consumers (e.g. aeps.alfares.cz) can reach AI: internal URL used by agentic-email
    echo -e "${BLUE}Verifying AI reachable from same-server services (e.g. aeps.alfares.cz uses http://ai-microservice:3380)...${NC}"
    AEPS_CONTAINER=""
    for c in agentic-email-processing-system-green agentic-email-processing-system-blue agentic-email-processing-system; do
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${c}$"; then
            AEPS_CONTAINER="$c"
            break
        fi
    done
    if [ -n "$AEPS_CONTAINER" ]; then
        if docker exec "$AEPS_CONTAINER" wget -q -O- --timeout=5 "http://ai-microservice:3380/health" 2>/dev/null | grep -q "healthy"; then
            echo -e "${GREEN}✓ Verified: $AEPS_CONTAINER (aeps.alfares.cz backend) can reach AI at http://ai-microservice:3380${NC}"
        else
            echo -e "${YELLOW}  Warning: $AEPS_CONTAINER could not reach http://ai-microservice:3380 (container may still be starting).${NC}"
        fi
    else
        echo -e "${YELLOW}  Skip: agentic-email container not running; run agentic-email deploy to verify aeps.alfares.cz → AI.${NC}"
    fi
    # If public URL failed, check if it works from same-network container (simulates same-server caller)
    if [ -n "${DOMAIN:-}" ] && command -v docker >/dev/null 2>&1 && docker network inspect nginx-network >/dev/null 2>&1; then
        PUBLIC_CODE=$(docker run --rm --network nginx-network "${HEALTH_CHECK_IMAGE:-alpine/curl:latest}" curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 -k "https://${DOMAIN}/health" 2>/dev/null || echo "000")
        if [ "$PUBLIC_CODE" = "200" ]; then
            echo -e "${GREEN}✓ Public URL https://${DOMAIN} is reachable from same-network containers (requests from same server will work).${NC}"
        elif [ "$PUBLIC_CODE" != "000" ]; then
            echo -e "${YELLOW}  Public URL https://${DOMAIN} returned HTTP ${PUBLIC_CODE} from same network. Fix nginx/firewall if aeps.alfares.cz or other services call this URL.${NC}"
        fi
    fi
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    ✅ AI microservice deployment completed successfully!   ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${GREEN}Total deployment time: ${TOTAL_DURATION_FORMATTED}s${NC}"
    echo ""
    echo "Check status with:"
    echo "  cd $NGINX_MICROSERVICE_PATH"
    echo "  ./scripts/status-all-services.sh"
    exit "$TEST_EXIT"
else
    TOTAL_DURATION_FORMATTED=$(awk "BEGIN {printf \"%.2f\", $TOTAL_DURATION}")
    echo ""
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}   ❌ AI microservice deployment failed! Failed after: ${TOTAL_DURATION_FORMATTED}s${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    print_phase_summary
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║           ❌ AI microservice deployment failed!            ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Please check the error messages above and:"
    echo "  1. Verify nginx-microservice is properly configured"
    echo "  2. Check service registry: $NGINX_MICROSERVICE_PATH/service-registry/$SERVICE_NAME.json"
    echo "  3. Review deployment logs (and container logs if health check fails)"
    echo "  4. Check service health: cd $NGINX_MICROSERVICE_PATH && ./scripts/blue-green/health-check.sh $SERVICE_NAME"
    exit 1
fi
