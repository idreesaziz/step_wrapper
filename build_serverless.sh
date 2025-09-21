#!/bin/bash

echo "🐳 Building ACE-Step Serverless Docker Image"
echo "============================================="

# Check if Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Starting Docker..."
    sudo systemctl start docker
    sleep 3
fi

echo "✅ Docker is running"

# Build the image
echo "🔨 Building image (this takes 10-15 minutes)..."
echo "Image will be: ghcr.io/idreesaziz/ace-step-serverless:latest"

sudo docker build -f Dockerfile.serverless -t ghcr.io/idreesaziz/ace-step-serverless:latest . 

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Build completed successfully!"
    echo ""
    echo "📦 Image built: ghcr.io/idreesaziz/ace-step-serverless:latest"
    
    # Show image size
    echo "📊 Image details:"
    sudo docker images ghcr.io/idreesaziz/ace-step-serverless:latest
    
    echo ""
    echo "🔑 Next steps:"
    echo "1. Create GitHub token: https://github.com/settings/tokens"
    echo "2. Login: echo 'YOUR_TOKEN' | sudo docker login ghcr.io -u idreesaziz --password-stdin"
    echo "3. Push: sudo docker push ghcr.io/idreesaziz/ace-step-serverless:latest"
    echo "4. Make package public on GitHub"
    echo "5. Update RunPod release with the image"
    echo ""
    echo "📖 See GITHUB_TOKEN_SETUP.md for detailed instructions"
    
else
    echo "❌ Build failed!"
    echo "Check the error messages above for details"
    exit 1
fi
