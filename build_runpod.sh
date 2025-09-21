#!/bin/bash

# RunPod ACE-Step Deployment Script
# This script builds and pushes a Docker image for RunPod deployment

set -e

echo "🚀 ACE-Step RunPod Deployment Builder"
echo "====================================="

# Configuration
IMAGE_NAME="ace-step-runpod"
REGISTRY="ghcr.io"  # GitHub Container Registry (free)
VERSION="latest"

# Get GitHub username
echo "📝 Docker Registry Setup"
echo "We'll use GitHub Container Registry (free)"
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ GitHub username is required"
    exit 1
fi

FULL_IMAGE_NAME="${REGISTRY}/${GITHUB_USERNAME}/${IMAGE_NAME}:${VERSION}"

echo "🐳 Image will be: ${FULL_IMAGE_NAME}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Build the image
echo ""
echo "🔨 Building Docker image..."
echo "This may take 10-15 minutes for the first build..."

if docker build -t "${FULL_IMAGE_NAME}" .; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Check image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}" --format "table {{.Size}}" | tail -n 1)
echo "📦 Image size: ${IMAGE_SIZE}"

# Ask if user wants to push
echo ""
read -p "🚀 Push image to registry? (y/N): " PUSH_CHOICE

if [[ $PUSH_CHOICE =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔑 You need to login to GitHub Container Registry first:"
    echo "1. Create a Personal Access Token at: https://github.com/settings/tokens"
    echo "2. Select 'write:packages' scope"
    echo "3. Run: echo \$TOKEN | docker login ghcr.io -u ${GITHUB_USERNAME} --password-stdin"
    echo ""
    read -p "Press Enter after you've logged in..."
    
    echo "📤 Pushing image to registry..."
    if docker push "${FULL_IMAGE_NAME}"; then
        echo "✅ Image pushed successfully!"
        echo ""
        echo "🎉 Deployment Ready!"
        echo "==================="
        echo "Your Docker image: ${FULL_IMAGE_NAME}"
        echo ""
        echo "Next steps for RunPod:"
        echo "1. Go to runpod.io and create a pod"
        echo "2. Use Custom Docker Image: ${FULL_IMAGE_NAME}"
        echo "3. Expose port 8000/http"
        echo "4. Set Container Disk to 50GB"
        echo "5. Start the pod!"
        echo ""
        echo "📖 See RUNPOD_DEPLOY.md for detailed instructions"
    else
        echo "❌ Push failed. Check your login credentials."
        exit 1
    fi
else
    echo "📋 Image built locally. To deploy:"
    echo "1. Push the image: docker push ${FULL_IMAGE_NAME}"
    echo "2. Use it in RunPod with image: ${FULL_IMAGE_NAME}"
fi

echo ""
echo "🧪 Test locally first:"
echo "docker run -p 8000:8000 ${FULL_IMAGE_NAME}"
echo "Then visit: http://localhost:8000/health"
