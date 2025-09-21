# 🔑 GitHub Container Registry Setup Guide

## Step 1: Create Personal Access Token

1. Go to **GitHub Settings**: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. **Token name**: `RunPod Docker Push`
4. **Expiration**: 90 days (or longer if you prefer)
5. **Select scopes**:
   - ✅ `write:packages` (allows pushing to container registry)
   - ✅ `read:packages` (allows pulling from container registry)
   - ✅ `delete:packages` (optional - for cleanup)
6. Click **"Generate token"**
7. **IMPORTANT**: Copy the token immediately (you won't see it again!)

## Step 2: Login to GitHub Container Registry

Once your Docker build completes, run:

```bash
# Replace YOUR_TOKEN with the token you just created
echo "YOUR_TOKEN" | sudo docker login ghcr.io -u idreesaziz --password-stdin
```

Example:
```bash
echo "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" | sudo docker login ghcr.io -u idreesaziz --password-stdin
```

## Step 3: Push the Image

After successful login:

```bash
sudo docker push ghcr.io/idreesaziz/ace-step-serverless:latest
```

## Step 4: Make Package Public (Important!)

1. Go to your GitHub profile: https://github.com/idreesaziz
2. Click **"Packages"** tab
3. Find **"ace-step-serverless"** package
4. Click on it
5. Go to **"Package settings"** (right side)
6. Scroll to **"Danger Zone"**
7. Click **"Change visibility"** → **"Make public"**
8. Type the package name to confirm

This is required so RunPod can pull your image!

## Step 5: Configure RunPod

In your RunPod serverless dashboard:

1. Go to endpoint `9eb182ubs5j0jf`
2. **Releases** tab → **"Create New Release"**
3. **Docker Image**: `ghcr.io/idreesaziz/ace-step-serverless:latest`
4. **Container Disk**: 50GB
5. **Environment Variables**:
   ```
   HF_HUB_ENABLE_HF_TRANSFER=1
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```
6. Save the release

## Troubleshooting

### Authentication Failed
```bash
# Make sure you're using the right username
docker login ghcr.io -u idreesaziz

# Check if token has correct permissions
# Go back to GitHub → Settings → Personal Access Tokens
```

### Push Failed
```bash
# Make sure image was built successfully
sudo docker images | grep ace-step-serverless

# Check if you're logged in
sudo docker info | grep Username
```

### RunPod Can't Pull Image
- Make sure the package is **public** (Step 4 above)
- Verify the image name is exactly: `ghcr.io/idreesaziz/ace-step-serverless:latest`

## Quick Commands Summary

```bash
# 1. Create GitHub token (do this in browser)

# 2. Login to registry
echo "YOUR_GITHUB_TOKEN" | sudo docker login ghcr.io -u idreesaziz --password-stdin

# 3. Push image (after build completes)
sudo docker push ghcr.io/idreesaziz/ace-step-serverless:latest

# 4. Make package public (do this in browser)

# 5. Update RunPod release (do this in RunPod dashboard)
```

Your Docker build is running in the background. Once it completes, follow these steps to deploy! 🚀
