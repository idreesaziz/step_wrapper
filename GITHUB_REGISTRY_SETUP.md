# 🔐 GitHub Container Registry Setup

## Step 1: Create GitHub Personal Access Token

1. **Go to GitHub Settings**:
   - Visit: https://github.com/settings/tokens
   - Or: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Create New Token**:
   - Click **"Generate new token (classic)"**
   - **Note**: "RunPod Docker Registry"
   - **Expiration**: 90 days (or No expiration for convenience)
   - **Scopes**: Check these boxes:
     - ✅ `write:packages` (Upload packages to GitHub Package Registry)
     - ✅ `read:packages` (Download packages from GitHub Package Registry)
     - ✅ `delete:packages` (Delete packages from GitHub Package Registry)

3. **Generate and Copy Token**:
   - Click **"Generate token"**
   - **IMPORTANT**: Copy the token immediately (you won't see it again!)
   - Save it somewhere safe

## Step 2: Login to GitHub Container Registry

```bash
# Replace YOUR_TOKEN with the token you just created
echo "YOUR_TOKEN" | sudo docker login ghcr.io -u idreesaziz --password-stdin
```

Example:
```bash
echo "ghp_xxxxxxxxxxxxxxxxxxxx" | sudo docker login ghcr.io -u idreesaziz --password-stdin
```

## Step 3: Push Your Image

After the Docker build completes:

```bash
sudo docker push ghcr.io/idreesaziz/ace-step-serverless:latest
```

## Step 4: Update RunPod Endpoint

In your RunPod dashboard:

1. **Go to your endpoint**: `9eb182ubs5j0jf`
2. **Releases tab** → **Create New Release**
3. **Docker Image**: `ghcr.io/idreesaziz/ace-step-serverless:latest`
4. **Container Disk**: 50GB
5. **Save & Deploy**

## Alternative: Docker Hub (If GitHub doesn't work)

1. **Create account at**: https://hub.docker.com
2. **Login locally**:
   ```bash
   sudo docker login
   # Enter your Docker Hub username and password
   ```
3. **Tag and push**:
   ```bash
   sudo docker tag ghcr.io/idreesaziz/ace-step-serverless:latest idreesaziz/ace-step-serverless:latest
   sudo docker push idreesaziz/ace-step-serverless:latest
   ```
4. **Use in RunPod**: `idreesaziz/ace-step-serverless:latest`

## 🚨 Security Notes

- **Never commit tokens to GitHub**
- **Use tokens with minimal required permissions**
- **Regenerate tokens periodically**
- **Consider using GitHub Actions for automatic builds**

---

**Next**: Wait for Docker build to complete, then follow the push instructions above!
