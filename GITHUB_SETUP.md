# 🔐 GitHub Container Registry Setup Guide

## Step 1: Create GitHub Personal Access Token

1. **Go to GitHub Settings**:
   - Visit: https://github.com/settings/tokens
   - Click **"Generate new token (classic)"**

2. **Configure Token**:
   - **Note**: "ACE-Step Docker Images"
   - **Expiration**: 90 days (or longer)
   - **Scopes**: Check these boxes:
     - ✅ `write:packages` (Upload packages)
     - ✅ `read:packages` (Download packages)
     - ✅ `delete:packages` (Manage packages)

3. **Generate and Copy**:
   - Click **"Generate token"**
   - **COPY THE TOKEN** (starts with `ghp_`)
   - ⚠️ Save it somewhere - you won't see it again!

## Step 2: Login to GitHub Container Registry

```bash
# Replace YOUR_TOKEN_HERE with your actual token
echo "YOUR_TOKEN_HERE" | sudo docker login ghcr.io -u idreesaziz --password-stdin
```

**Example**:
```bash
echo "ghp_1234567890abcdefghijklmnop" | sudo docker login ghcr.io -u idreesaziz --password-stdin
```

## Step 3: Push Your Docker Image

After the build completes:
```bash
sudo docker push ghcr.io/idreesaziz/ace-step-serverless:latest
```

## Step 4: Make Package Public

1. **Go to your packages**: https://github.com/idreesaziz?tab=packages
2. **Click** on "ace-step-serverless" package
3. **Package settings** (right sidebar)
4. **Change package visibility** → **Make public**
5. Type the repository name to confirm

## Step 5: Get RunPod API Key

1. **Go to RunPod**: https://runpod.io
2. **Settings** → **API Keys**  
3. **Create API Key**
4. **Copy the key** (starts with letters/numbers)

---

## 🚀 Ready for Deployment!

Once you complete these steps, you'll have:
- ✅ Docker image built and pushed
- ✅ Public package on GitHub
- ✅ RunPod API key ready
- ✅ Serverless endpoint configured

**Total time**: ~20 minutes to live API! 🎵

## Next Commands (run after build completes):

```bash
# 1. Login to GitHub (use your token)
echo "YOUR_GITHUB_TOKEN" | sudo docker login ghcr.io -u idreesaziz --password-stdin

# 2. Push image  
sudo docker push ghcr.io/idreesaziz/ace-step-serverless:latest

# 3. Test your API
python3 test_serverless_api.py
```
