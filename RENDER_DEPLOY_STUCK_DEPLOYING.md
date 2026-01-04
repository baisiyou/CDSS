# Render Deployment Stuck at "Deploying..." - Troubleshooting Guide

## Common Causes

### 1. Build Process Taking Time
- **Normal**: Building scikit-learn, numpy, pandas can take 5-15 minutes
- **Action**: Wait 10-15 minutes before taking action

### 2. Model Loading Timeout
- Models may take time to load during startup
- **Check**: Look at Render logs for "Loading models..." messages

### 3. Port Binding Issues
- Service may be starting but not binding to port correctly
- **Check**: Look for "Gunicorn binding to:" in logs

## Immediate Actions

### Step 1: Check Render Dashboard Logs
1. Go to Render Dashboard
2. Click on your service (cdss-api)
3. Click "Logs" tab
4. Look for:
   - Build progress messages
   - Error messages
   - "Starting gunicorn" messages
   - "Loading models..." messages

### Step 2: Check Current Status
Look for these indicators in logs:
- ✅ `Installing dependencies...` - Normal, wait
- ✅ `Building wheel for scikit-learn...` - Normal, can take 5-10 minutes
- ✅ `Loading models...` - Normal, wait
- ✅ `Gunicorn binding to: 0.0.0.0:XXXX` - Service starting
- ❌ `ERROR` or `FAILED` - Problem detected

### Step 3: Wait Times
- **Free Tier**: Can take 10-20 minutes for full deployment
- **First Deploy**: May take longer (15-25 minutes)
- **Subsequent Deploys**: Usually faster (5-10 minutes)

## If Stuck for More Than 20 Minutes

### Option 1: Cancel and Retry
1. In Render Dashboard
2. Find the current deployment
3. Click "Cancel" (if available)
4. Click "Manual Deploy" → "Deploy latest commit"

### Option 2: Check for Errors
1. Scroll through all logs
2. Look for:
   - `MemoryError` - Need to reduce model size
   - `ImportError` - Missing dependency
   - `Port already in use` - Service conflict
   - `Timeout` - Build taking too long

### Option 3: Simplify Build (Temporary)
If build keeps failing, you can temporarily simplify:

```yaml
# render.yaml - Simplified build command
buildCommand: pip install --upgrade pip && pip install -r requirements.txt
```

## Common Issues and Solutions

### Issue: Build Stuck at "Installing dependencies"
**Solution**: 
- Wait 10-15 minutes (scikit-learn compilation takes time)
- Check if Python 3.9.18 is correctly set

### Issue: "Module not found" after deployment
**Solution**:
- Check `requirements.txt` includes all dependencies
- Verify imports in `cdss_api.py` match installed packages

### Issue: Service starts but returns 503
**Solution**:
- Check if models are loading correctly
- Look for "Model loaded successfully" in logs
- Verify model files exist in `models/` directory

### Issue: Port binding errors
**Solution**:
- Verify `gunicorn_config.py` uses `0.0.0.0:$PORT`
- Check `startCommand` in `render.yaml` includes `--bind 0.0.0.0:$PORT`

## Quick Health Check

Once deployment completes, test:
```bash
curl https://cdss-kd6u.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "combination_analyzer_initialized": true
}
```

## Next Steps

1. **Wait 15 minutes** - First deployment can be slow
2. **Check logs** - Look for specific error messages
3. **Test health endpoint** - Once deployment shows "Live"
4. **Check frontend** - Try the analysis function

## If Still Stuck

1. Check Render status page: https://status.render.com
2. Try deploying from a different branch (create a test branch)
3. Contact Render support if issue persists

