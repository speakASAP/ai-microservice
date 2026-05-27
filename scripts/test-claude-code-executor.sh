#!/bin/bash

# test-claude-code-executor.sh
# Manual bash polling script for Claude Code Executor E2E tests
# Polls job status every 2 seconds up to 30 times (60 seconds total)

set -e

# Configuration
BASE_URL="${BASE_URL:-http://localhost:3380}"
ENDPOINT="/ai/claude-code-execute"
POLL_INTERVAL=2
MAX_POLLS=30

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

# Require jq for JSON parsing
if ! command -v jq &> /dev/null; then
  log_error "jq is required. Install with: apt-get install jq"
  exit 1
fi

# Function to extract jobId from JSON response
extract_job_id() {
  local json_response="$1"
  echo "$json_response" | jq -r '.jobId' 2>/dev/null
}

# Function to extract status from JSON response
extract_status() {
  local json_response="$1"
  echo "$json_response" | jq -r '.status' 2>/dev/null
}

# Function to submit a job
submit_job() {
  local repo_path="$1"
  local branch="${2:-main}"
  local instructions="${3:-List all files in src/ directory}"
  local expected_outcome="${4:-File listing complete}"

  log_info "Submitting job to $BASE_URL$ENDPOINT"
  log_info "  Repository: $repo_path"
  log_info "  Branch: $branch"
  log_info "  Instructions: $instructions"

  local task_id
  task_id="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || printf '%s-%s-%s-%s-%s' "$(head -c4 /dev/urandom | xxd -p)" "$(head -c2 /dev/urandom | xxd -p)" "$(head -c2 /dev/urandom | xxd -p)" "$(head -c2 /dev/urandom | xxd -p)" "$(head -c6 /dev/urandom | xxd -p)" 2>/dev/null || date +%s)"

  local payload=$(cat <<EOF
{
  "taskId": "$task_id",
  "repoPath": "$repo_path",
  "branch": "$branch",
  "instructions": "$instructions",
  "expectedOutcome": "$expected_outcome"
}
EOF
)

  local response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$payload" \
    "$BASE_URL$ENDPOINT")

  echo "$response"
}

# Function to poll job status
poll_job_status() {
  local job_id="$1"

  log_info "Polling job status for jobId: $job_id"
  log_info "Poll interval: ${POLL_INTERVAL}s, max polls: $MAX_POLLS"

  local poll_count=0
  local final_status=""

  while [ $poll_count -lt $MAX_POLLS ]; do
    poll_count=$((poll_count + 1))

    local status_response=$(curl -s -X GET "$BASE_URL$ENDPOINT/$job_id")
    local status=$(extract_status "$status_response")

    if [ -z "$status" ]; then
      log_error "Failed to extract status from response: $status_response"
      return 1
    fi

    log_info "Poll #$poll_count: status = $status"

    # Check if job is complete
    if [ "$status" = "success" ] || [ "$status" = "failed" ] || [ "$status" = "completed" ]; then
      final_status="$status"
      log_info "Job reached terminal state: $final_status"
      echo "$status_response"
      return 0
    fi

    # Wait before next poll
    if [ $poll_count -lt $MAX_POLLS ]; then
      sleep $POLL_INTERVAL
    fi
  done

  log_warn "Job did not reach terminal state after $((MAX_POLLS * POLL_INTERVAL)) seconds"
  log_warn "Last status: $(curl -s -X GET "$BASE_URL$ENDPOINT/$job_id" | extract_status)"
  return 1
}

# Main script
main() {
  log_info "Claude Code Executor E2E Test Script"
  log_info "========================================"

  # Use provided arguments or defaults
  local repo_path="${1:-/home/ssf/Documents/Github/beauty}"
  local branch="${2:-main}"
  local instructions="${3:-List all files in src/ directory}"
  local expected_outcome="${4:-File listing complete}"

  # Submit job
  log_info "Step 1: Submit job"
  local job_response=$(submit_job "$repo_path" "$branch" "$instructions" "$expected_outcome")

  if [ -z "$job_response" ]; then
    log_error "Failed to submit job"
    exit 1
  fi

  log_info "Response: $job_response"

  # Extract jobId
  local job_id=$(extract_job_id "$job_response")
  if [ -z "$job_id" ] || [ "$job_id" = "null" ]; then
    log_error "Failed to extract jobId from response"
    exit 1
  fi

  log_info "Successfully extracted jobId: $job_id"
  log_info ""

  # Poll job status
  log_info "Step 2: Poll job status"
  local final_response=$(poll_job_status "$job_id")

  if [ $? -eq 0 ]; then
    log_info "Job completed successfully"
    log_info "Final response: $final_response"
    exit 0
  else
    log_error "Job polling failed or timed out"
    exit 1
  fi
}

# Show usage if requested
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  cat <<EOF
Usage: $0 [repo_path] [branch] [instructions] [expected_outcome]

Arguments:
  repo_path           Repository path (default: /home/ssf/Documents/Github/beauty)
  branch              Git branch (default: main)
  instructions        Task instructions (default: List all files in src/ directory)
  expected_outcome    Expected outcome description (default: File listing complete)

Environment Variables:
  BASE_URL            API base URL (default: http://localhost:3380)

Examples:
  # Use all defaults
  $0

  # Specify repository
  $0 /home/ssf/Documents/Github/ai-microservice

  # Specify repository and branch
  $0 /home/ssf/Documents/Github/ai-microservice develop

  # Custom base URL
  BASE_URL=http://ai-microservice:3380 $0

EOF
  exit 0
fi

# Run main function
main "$@"
