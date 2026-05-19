# REVIA Troubleshooting Guide

## Issue: "Format not recognised" Error When Playing Audio

### Symptoms
```
[TTS] Playback error: Error opening 'C:\Users\USER\AppData\Local\Temp\tmpXXXXXXXX.wav': Format not recognised.
```

### Root Causes (in order of likelihood)

| Cause | Indicators | Solution |
|-------|-----------|----------|
| **Zero-byte WAV file** | File size is 0 bytes | Check Gradio API response; try different Qwen3 server |
| **Corrupted WAV from API** | Random failures, not repeatable | Restart Qwen3-TTS server; check server logs |
| **Unsupported audio codec** | Consistent with certain speakers | Try different voice/language combination |
| **Temp file permissions** | Happens only on certain users | Check `%temp%` folder permissions |
| **Soundfile library issue** | ImportError in logs | Reinstall soundfile: `pip install --upgrade soundfile` |

### Quick Fixes

**1. Check for zero-byte files:**
```bash
# Windows PowerShell
Get-ChildItem "$env:TEMP\tmp*.wav" | Where-Object {$_.Length -eq 0}

# If found, remove them:
Get-ChildItem "$env:TEMP\tmp*.wav" | Where-Object {$_.Length -eq 0} | Remove-Item
```

**2. Restart Qwen3-TTS server:**
```bash
# Stop the server
# Start fresh
python -m qwen_tts.cli.demo
```

**3. Try a different voice/language:**
```python
# Instead of current settings
tts.generate_custom_voice(
    "Hello world",
    language="English",  # was: "Auto"
    speaker="Dylan",     # was: "Ryan"
)
```

**4. Check temp folder permissions:**
```bash
# Run as Administrator, then:
icacls "%temp%" /grant "%username%:(F)"
```

**5. Verify soundfile installation:**
```bash
python -c "import soundfile; print(soundfile.__version__)"
# Should print version, not error
```

---

## Issue: Server Keeps Disconnecting

### Symptoms
```
[Revia] Server disconnected. Reconnection attempt 1, retrying in 3 seconds...
[Revia] Server disconnected. Reconnection attempt 2, retrying in 6 seconds...
[Revia] Server disconnected. Reconnection attempt 3, retrying in 12 seconds...
```

Keeps repeating indefinitely.

### Root Causes

| Cause | Indicators | Solution |
|-------|-----------|----------|
| **Server is down** | Core server process not running | Start REVIA Core: `python -m revia.core` |
| **Network connectivity** | Can't ping server IP | Check firewall, network connection |
| **Port mismatch** | Wrong port in env vars | Verify `REVIA_CORE_WS_URL`, `REVIA_CORE_URL` |
| **Firewall blocking** | Connection timeout immediately | Add firewall exception for port 8123-8124 |
| **Server overloaded** | Drops connection under load | Check server CPU/memory; reduce load |
| **Old client, new server** | Version mismatch errors | Rebuild controller from latest |

### Quick Fixes

**1. Verify server is running:**
```bash
# Check if process is running
tasklist | findstr "python"  # Windows
ps aux | grep "revia.core"  # Linux/Mac

# If not, start it
python -m revia.core
```

**2. Check connectivity:**
```bash
# Try to reach the server
curl http://127.0.0.1:8123/api/status
# Should return JSON, not "Connection refused"
```

**3. Verify environment variables:**
```bash
# Windows
set | findstr REVIA_CORE

# Should show:
# REVIA_CORE_URL=http://127.0.0.1:8123
# REVIA_CORE_WS_URL=ws://127.0.0.1:8124
```

**4. Check firewall (Windows):**
```bash
# PowerShell as Administrator
# Allow port 8123 (REST) and 8124 (WebSocket)
New-NetFirewallRule -DisplayName "REVIA Core" `
  -Direction Inbound -LocalPort 8123,8124 `
  -Protocol TCP -Action Allow
```

**5. Check server logs:**
```bash
tail -f logs/revia_core.log  # Watch for errors
# Look for:
# - "Address already in use" → Kill process on that port
# - "Cannot bind" → Port not available
# - "Connection reset" → Client disconnecting
```

**6. Restart core completely:**
```bash
# Kill any running instances
taskkill /IM python.exe /F  # Windows - use with caution!

# Or more safely:
pkill -f "revia.core"  # Linux/Mac

# Wait 5 seconds
sleep 5

# Start fresh
python -m revia.core
```

---

## Issue: Reconnection Takes Too Long

### Symptoms
```
[Revia] Server disconnected. Reconnection attempt 1, retrying in 3 seconds...
[Revia] Server disconnected. Reconnection attempt 2, retrying in 6 seconds...
[Revia] Server disconnected. Reconnection attempt 3, retrying in 12 seconds...
[Revia] Server disconnected. Reconnection attempt 4, retrying in 24 seconds...
[Revia] Server disconnected. Reconnection attempt 5, retrying in 30 seconds...
```

Waiting 30 seconds between retries feels slow.

### Root Cause
Exponential backoff is designed to prevent hammering a recovering server, but max interval is 30 seconds.

### Solutions

**Option 1: Restart the app (quickest)**
- Exit Revia controller
- Start REVIA Core server
- Restart Revia controller
- Should connect immediately

**Option 2: Modify backoff configuration (if you control the code)**

Edit `controller_client.py` line 56-57:
```python
# Current (safe defaults)
self._reconnect_base_ms = 3000      # Start at 3 seconds
self._reconnect_max_ms = 30000      # Cap at 30 seconds

# More aggressive (WARNING: might hammer a recovering server)
self._reconnect_base_ms = 1000      # Start at 1 second
self._reconnect_max_ms = 10000      # Cap at 10 seconds
```

**Option 3: Add manual "Reconnect Now" button**

Wire this in your UI:
```python
def reconnect_now(self):
    """Skip backoff and try connecting immediately."""
    self.controller_client.reconnect_timer.stop()
    self.controller_client.reconnect_timer.setInterval(100)  # Short delay
    self.controller_client._try_connect()
```

---

## Monitoring & Debugging

### Enable verbose logging
```python
# In your app startup
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Watch for specific error patterns
```bash
# TTS errors
grep "\[TTS\]" logs/revia.log | tail -20

# Connection errors  
grep "\[ControllerClient\]" logs/revia.log | tail -20

# All errors
grep "ERROR" logs/revia.log | tail -20
```

### Check system resources during disconnect
```bash
# Windows Task Manager
# - Look for CPU spikes on restart
# - Check memory usage
# - Look at network connections

# Linux/Mac
top -p $(pgrep -f "revia.core")
netstat -an | grep 8123
```

---

## Performance Tuning

### If getting random "Format not recognised" errors:

**1. Check Qwen3-TTS server capacity:**
```bash
# Monitor server during TTS synthesis
watch -n 1 'curl http://localhost:8000/api/status'
# Look for queue depth, response times
```

**2. Reduce synthesis load:**
```python
# Instead of synthesizing full paragraphs
long_text = "This is a very long paragraph..."

# Break into sentences
sentences = long_text.split(". ")
for sent in sentences:
    tts.generate_custom_voice(sent + ".")  # Synthesize each separately
    time.sleep(0.5)  # Add delay between requests
```

**3. Increase synthesis semaphore (if bottlenecked):**

Edit `tts_backend.py` line 118:
```python
# Current (max 3 parallel synthesis)
self._synth_semaphore = threading.Semaphore(3)

# Increase if your server can handle it (may cause timeouts if too high)
self._synth_semaphore = threading.Semaphore(6)
```

---

## Common Error Messages & Solutions

| Error Message | Cause | Fix |
|---------------|-------|-----|
| `Address already in use` | Port 8123/8124 in use | Kill existing process: `lsof -i :8123` |
| `Connection refused` | Server not running | Start server: `python -m revia.core` |
| `Timeout` | Server slow/unresponsive | Increase timeout in controller_client.py |
| `Format not recognised` | Zero-byte WAV file | Restart Qwen3-TTS or try different settings |
| `Cannot connect to Qwen3-TTS` | Wrong URL or server down | Check `REVIA_CORE_URL` env var |
| `api_name missing` | Wrong Gradio endpoint | Check Qwen3-TTS model variant |

---

## Getting Help

When reporting a bug, include:

1. **Error message (exact):**
   ```
   [copy the full error from logs]
   ```

2. **Environment:**
   ```
   - OS: Windows 11 / Ubuntu 22.04 / etc.
   - Python version: 3.10 / 3.11 / etc.
   - Qwen3-TTS variant: Base / CustomVoice / VoiceDesign
   ```

3. **Steps to reproduce:**
   ```
   1. Start server
   2. Click [X] button
   3. Error occurs
   ```

4. **Relevant logs:**
   ```bash
   # Last 50 lines of logs
   tail -50 logs/revia.log
   ```

5. **Configuration:**
   ```bash
   # Environment variables
   set | findstr REVIA_CORE  # Windows
   env | grep REVIA_CORE      # Linux/Mac
   ```

---

## Health Check Script

```python
#!/usr/bin/env python3
"""Quick health check for REVIA."""
import requests
import json
from pathlib import Path

def check_server():
    """Check if core server is reachable."""
    try:
        r = requests.get("http://127.0.0.1:8123/api/status", timeout=2)
        print(f"✓ Server responds: HTTP {r.status_code}")
        if r.ok:
            status = r.json()
            print(f"  State: {status.get('state', '?')}")
            print(f"  Connection: {status.get('llm_connection', '?')}")
    except Exception as exc:
        print(f"✗ Server unreachable: {exc}")

def check_temp_files():
    """Check for zero-byte WAV files."""
    import glob
    zero_bytes = []
    for f in glob.glob("/tmp/tmp*.wav"):
        if Path(f).stat().st_size == 0:
            zero_bytes.append(f)
    
    if zero_bytes:
        print(f"✗ Found {len(zero_bytes)} empty WAV files (cleanup recommended)")
    else:
        print(f"✓ No empty WAV files found")

def check_dependencies():
    """Check if required packages are installed."""
    deps = ["sounddevice", "soundfile", "pyttsx3", "gradio_client"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✓ {dep} installed")
        except ImportError:
            print(f"✗ {dep} missing: pip install {dep}")

if __name__ == "__main__":
    print("=== REVIA Health Check ===")
    print("\n[Server]")
    check_server()
    print("\n[Temp Files]")
    check_temp_files()
    print("\n[Dependencies]")
    check_dependencies()
    print("\n=== Done ===")
```

Save as `health_check.py` and run:
```bash
python health_check.py
```
