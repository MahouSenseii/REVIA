# REVIA Bug Fixes Applied - April 30, 2026

## Summary

Two critical bugs have been identified and fixed:

1. **TTS Playback Error** - "Format not recognised" crash
2. **Silent Server Disconnections** - No user feedback during reconnection

---

## Fix #1: TTS WAV Playback Validation

### Problem
When soundfile/sounddevice attempted to play a WAV, they would crash with:
```
[TTS] Playback error: Error opening 'C:\...\tmp.wav': Format not recognised.
```

**Root causes:**
- WAV file was 0 bytes (corrupted/incomplete from Gradio API)
- WAV file had invalid/unsupported codec
- File was deleted before playback started
- No validation before attempting to read

### Solution Applied

#### File: `tts_backend.py`

**1. Updated `play_wav()` method (line ~354)**
- Added file existence check before reading
- Added file size validation (reject 0-byte files)
- Added WAV format validation using `soundfile.info()`
- Better error categorization with exception type logging
- Graceful fallback with clear error messages

**2. Updated `_play_wav_blocking()` method (line ~497)**
- Same validations as `play_wav()`
- Synchronous version now prevents crash before attempting playback

**3. Updated `_extract_wav()` method (line ~1011)**
- Validates extracted WAV is not empty
- Validates copied WAV succeeded with data
- Deletes corrupted files automatically
- Logs file sizes for debugging

### Code Added

```python
# Pre-flight validation
wav_path_obj = Path(wav_path)

if not wav_path_obj.exists():
    _log.error("[TTS] WAV file does not exist: %s", wav_path)
    self.error_occurred.emit(f"Audio file not found: {wav_path}")
    return

# Check file size
file_size = wav_path_obj.stat().st_size
if file_size == 0:
    _log.error("[TTS] WAV file is empty (0 bytes): %s", wav_path)
    self.error_occurred.emit("Generated audio is empty (0 bytes)")
    return

# Validate WAV format
try:
    info = sf.info(str(wav_path))
    _log.debug("[TTS] WAV format valid: %d Hz, %d channel(s), %d frames",
              info.samplerate, info.channels, info.frames)
except Exception as format_err:
    _log.error("[TTS] WAV format invalid or corrupted: %s", format_err)
    self.error_occurred.emit(f"Invalid audio format: corrupted WAV file")
    return
```

### Testing

To verify the fix works:

```python
# Test 1: Create a 0-byte WAV file
Path("/tmp/empty.wav").touch()
tts.play_wav("/tmp/empty.wav")
# Expected: Error message logged, no crash

# Test 2: Play a corrupted WAV
# Expected: Format error caught, fallback to pyttsx3

# Test 3: Normal WAV playback
tts.play_wav("valid_audio.wav")
# Expected: Plays normally
```

---

## Fix #2: Server Disconnection Notifications

### Problem
When Revia's server connection dropped:
- User received **no feedback** about the disconnection
- User didn't know if reconnection was happening
- Exponential backoff existed but was silent (3s → 6s → 12s → 30s)
- Users thought the app was frozen

### Solution Applied

#### File: `event_bus.py`

**Added two new signals:**

```python
# Connection retry feedback signals
connection_retry_attempt = Signal(int)     # attempt number (1, 2, 3, ...)
connection_retry_waiting = Signal(int)     # milliseconds to wait before retry
```

These signals allow the UI to notify users of reconnection attempts.

#### File: `controller_client.py`

**1. Updated `_on_ws_connection_timeout()` (line ~294)**

Now emits signals and logs when reconnection is scheduled:

```python
# Log reconnection attempt with wait time
wait_seconds = new_interval // 1000
msg = (
    f"[Revia] Server disconnected. "
    f"Reconnection attempt {self._reconnect_attempt}, "
    f"retrying in {wait_seconds} seconds..."
)
_log.warning("[ControllerClient] %s (interval=%dms)", msg, new_interval)

# Emit signals for UI notification
self.event_bus.log_entry.emit(msg)
self.event_bus.connection_retry_attempt.emit(self._reconnect_attempt)
self.event_bus.connection_retry_waiting.emit(new_interval)
```

**2. Updated `_on_ws_connected()` (line ~271)**

Now logs successful reconnection:

```python
if self._reconnect_attempt > 0:
    msg = f"[Revia] Server reconnected after {self._reconnect_attempt} attempt(s)"
    _log.info("[ControllerClient] %s", msg)
    try:
        self.event_bus.log_entry.emit(msg)
    except Exception:
        pass
```

### What Users Will See Now

**Before (silent):**
```
[no output]
[app appears frozen for 3-12 seconds]
[connection restored, user confused about what happened]
```

**After (with notifications):**
```
[User chat input]
[Revia] Server disconnected. Reconnection attempt 1, retrying in 3 seconds...
[User sees timeout feedback in logs]
[Revia] Server reconnected after 1 attempt(s)
[Chat continues normally]
```

If server is down longer:
```
[Revia] Server disconnected. Reconnection attempt 1, retrying in 3 seconds...
[wait 3s]
[Revia] Server disconnected. Reconnection attempt 2, retrying in 6 seconds...
[wait 6s]
[Revia] Server disconnected. Reconnection attempt 3, retrying in 12 seconds...
[wait 12s]
[Revia] Server reconnected after 3 attempt(s)
```

### How to Wire Up UI Notifications (Optional)

If you want to add audio or visual feedback in your main UI file:

```python
def setup_connection_handlers(self):
    """Wire up reconnection notifications in main window."""
    self.controller_client.event_bus.connection_retry_attempt.connect(
        self.on_connection_lost
    )
    self.controller_client.event_bus.connection_retry_waiting.connect(
        self.on_waiting_for_retry
    )

def on_connection_lost(self, attempt_num):
    """Called when server disconnection detected."""
    # Option 1: Show visual toast notification
    self.statusBar().showMessage(
        f"⚠ Connection lost. Retrying... (attempt {attempt_num})"
    )
    
    # Option 2: Play a subtle alert sound
    # self.tts_backend.play_notification_sound("disconnect.wav")
    
    # Option 3: Change UI state
    # self.chat_input.setEnabled(False)

def on_waiting_for_retry(self, wait_ms):
    """Called when scheduling next retry."""
    wait_seconds = wait_ms // 1000
    self.statusBar().showMessage(
        f"Reconnecting in {wait_seconds} seconds..."
    )
```

---

## Files Modified

1. **`revia_controller_py/app/tts_backend.py`**
   - Added WAV validation in `play_wav()` (~45 lines added)
   - Added WAV validation in `_play_wav_blocking()` (~45 lines added)
   - Enhanced `_extract_wav()` with file validation (~25 lines added)
   - **Total changes: ~115 lines**

2. **`revia_controller_py/app/event_bus.py`**
   - Added 2 new signals: `connection_retry_attempt`, `connection_retry_waiting`
   - Updated docstring with signal descriptions
   - **Total changes: ~10 lines**

3. **`revia_controller_py/app/controller_client.py`**
   - Enhanced `_on_ws_connection_timeout()` with notifications (~25 lines added)
   - Enhanced `_on_ws_connected()` with success logging (~15 lines added)
   - **Total changes: ~40 lines**

**Total lines added: ~165**  
**Complexity: Low** (no architectural changes, pure defensive programming)  
**Risk: Minimal** (all changes are error handling improvements)

---

## Verification Checklist

- [ ] TTS playback with valid WAV files works normally
- [ ] TTS playback with 0-byte WAV files shows error, no crash
- [ ] TTS playback with corrupted WAV shows "Invalid audio format" message
- [ ] Server disconnect shows "[Revia] Server disconnected" in logs
- [ ] Retry attempts log "Reconnection attempt N, retrying in Xs"
- [ ] Successful reconnection logs "[Revia] Server reconnected after N attempt(s)"
- [ ] Exponential backoff intervals are correct: 3s → 6s → 12s → 24s → 30s (capped)
- [ ] No crashes or hangs during reconnection cycles

---

## Next Steps (Optional Enhancements)

1. **Add visual/audio notification on disconnect**
   - Wire the new signals in your main UI file
   - Show toast notification or play sound

2. **Add manual reconnect button**
   - Emit a signal to trigger `_try_connect()` immediately
   - Allows users to skip the backoff wait if desired

3. **Add connection timeout configuration**
   - Make backoff times user-configurable
   - Currently: 3s → 6s → 12s → ... → 30s cap

4. **Add network status indicator**
   - Show current connection status in status bar
   - Green = connected, Red = disconnected, Yellow = retrying

5. **Add detailed error categories**
   - Distinguish between connection errors and API errors
   - Different UI treatment for "server down" vs "network timeout"

---

## Debugging Tips

### View WAV validation logs:
```bash
grep "\[TTS\].*WAV" logs/revia.log
```

### View connection retry logs:
```bash
grep "\[ControllerClient\].*Reconnect" logs/revia.log
```

### Simulate a disconnect (for testing):
```bash
# Kill the REVIA core server while app is running
# Watch logs for "Reconnection attempt" messages
```

### Monitor playback errors:
```bash
grep "\[TTS\].*Playback error" logs/revia.log
```

---

## Summary of Changes

| Issue | Cause | Fix | Impact |
|-------|-------|-----|--------|
| WAV Format Error | 0-byte/corrupted files | File validation before playback | Prevents crash, clearer errors |
| Silent Disconnects | No user feedback | Emit signals + log messages | Users know what's happening |
| Confusing Errors | Generic exception messages | Categorized error types | Better debugging |

All changes are **backward compatible** and **non-breaking**.
