#!/bin/bash

# Quick test script for your deployed API
# Usage: ./test_deployed_api.sh https://your-pod-8000.proxy.runpod.net

API_URL="$1"

if [ -z "$API_URL" ]; then
    echo "Usage: $0 <API_URL>"
    echo "Example: $0 https://abc123-8000.proxy.runpod.net"
    exit 1
fi

echo "🧪 Testing ACE-Step API at: $API_URL"

# Health check
echo "1. Health check..."
curl -s "$API_URL/health" | jq '.' 2>/dev/null || curl -s "$API_URL/health"

echo -e "\n2. Generating music..."
curl -X POST "$API_URL/runsync" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "upbeat electronic music",
      "duration": 10.0,
      "seed": 42
    }
  }' \
  -s | jq '.output.generation_time' 2>/dev/null || echo "Check RunPod logs for details"

echo -e "\nAPI test complete! ✅"
