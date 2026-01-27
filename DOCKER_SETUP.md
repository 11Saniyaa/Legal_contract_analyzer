# 🐳 Docker Setup Guide for Weaviate

## Prerequisites
- Docker Desktop installed and running
- Check Docker is running: `docker --version`

## Step-by-Step Setup

### 1. Start Weaviate Container

**Windows PowerShell:**
```powershell
docker run -d -p 8080:8080 -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -e DEFAULT_VECTORIZER_MODULE=none semitechnologies/weaviate:latest
```

**Windows CMD:**
```cmd
docker run -d -p 8080:8080 -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -e DEFAULT_VECTORIZER_MODULE=none semitechnologies/weaviate:latest
```

**Linux/Mac:**
```bash
docker run -d -p 8080:8080 \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  semitechnologies/weaviate:latest
```

### 2. Verify Weaviate is Running

**Check container status:**
```powershell
docker ps
```

You should see a container named something like `weaviate` or `semitechnologies/weaviate:latest` running on port 8080.

**Test connection:**
```powershell
curl http://localhost:8080/v1/.well-known/ready
```

Or open in browser: http://localhost:8080/v1/.well-known/ready

Expected response: `{"ready":true}`

### 3. Check Container Logs (if needed)

```powershell
docker ps  # Get container ID
docker logs <container_id>
```

### 4. Stop Weaviate (when needed)

```powershell
docker ps  # Get container ID
docker stop <container_id>
```

### 5. Remove Container (if you want to start fresh)

```powershell
docker ps -a  # See all containers (including stopped)
docker rm <container_id>
```

## Troubleshooting

### Port 8080 Already in Use

If you get an error that port 8080 is already in use:

**Option 1: Use a different port**
```powershell
docker run -d -p 8081:8080 -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -e DEFAULT_VECTORIZER_MODULE=none semitechnologies/weaviate:latest
```
Then update `.env` file: `WEAVIATE_URL=http://localhost:8081`

**Option 2: Stop the process using port 8080**
```powershell
# Find what's using port 8080
netstat -ano | findstr :8080
# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Docker Not Running

Make sure Docker Desktop is running:
- Check system tray for Docker icon
- Open Docker Desktop application
- Wait for it to fully start

### Container Keeps Stopping

Check logs for errors:
```powershell
docker logs <container_id>
```

### Verify Connection from Python

Test in Python:
```python
import requests
response = requests.get("http://localhost:8080/v1/.well-known/ready")
print(response.json())  # Should print: {'ready': True}
```

## Quick Commands Reference

| Command | Description |
|---------|-------------|
| `docker ps` | List running containers |
| `docker ps -a` | List all containers (including stopped) |
| `docker stop <id>` | Stop a container |
| `docker start <id>` | Start a stopped container |
| `docker rm <id>` | Remove a container |
| `docker logs <id>` | View container logs |
| `docker pull semitechnologies/weaviate:latest` | Update Weaviate image |

## Your Current Setup

Based on your `.env` file, you have:
```
WEAVIATE_URL=http://localhost:8080
```

This means your application expects Weaviate to be running on `localhost:8080`.

## Next Steps

1. ✅ Start Weaviate container (see Step 1 above)
2. ✅ Verify it's running (`docker ps`)
3. ✅ Test connection (`curl http://localhost:8080/v1/.well-known/ready`)
4. ✅ Run your Streamlit app: `streamlit run app.py`
5. ✅ Check connection status in the app's "View Contracts" page

---

**Note:** Weaviate is optional. If you don't start it, the system will automatically use Neo4j for vector search (slower but works).

