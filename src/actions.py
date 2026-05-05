"""
Gesture → macOS media action mapping.

Volume and play/pause use CoreGraphics HID events (pyobjc-framework-Quartz).
Track skipping uses AppleScript (osascript) targeting Music.app directly —
NX_KEYTYPE_NEXT/PREVIOUS HID events are not reliably delivered to Apple Music
on macOS Tahoe.

Dependency for volume/play:  pip install pyobjc-framework-Quartz

Public API:
    try_fire(gesture)   → str | None   fire action, return label, or None
    cooldown_fraction() → float        0.0 = just fired, 1.0 = ready
    hold_fraction()     → float        0.0 = hold just started, 1.0 = threshold met
    reset_hold()        → None         call when the active gesture disappears
"""

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

try:
    import Quartz
    _QUARTZ_OK = True
except ImportError:
    _QUARTZ_OK = False
    print(
        "Warning: pyobjc-framework-Quartz not found — media actions disabled.\n"
        "  Install with:  pip install pyobjc-framework-Quartz"
    )

# ── NX media-key constants (hidsystem/ev_keymap.h) ────────────────────────────
_NX_KEYTYPE_SOUND_UP   = 0
_NX_KEYTYPE_SOUND_DOWN = 1
_NX_KEYTYPE_PLAY       = 16   # play / pause toggle

# NSEvent type for system-defined (HID) events (NSEvent.h)
_NS_SYSTEM_DEFINED = 14


def _send_media_key(key_type: int) -> None:
    """
    Post a media-key press-then-release pair via CoreGraphics.

    Uses NSEvent.otherEventWithType:… to build an NSSystemDefined event and
    CGEventPost to inject it into the HID event stream — identical to what a
    physical media key press produces.
    """
    if not _QUARTZ_OK:
        return
    for down in (True, False):
        flags = 0xa00 if down else 0xb00
        data1 = (key_type << 16) | ((0xa if down else 0xb) << 8)
        ev = Quartz.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            _NS_SYSTEM_DEFINED,   # type
            (0, 0),               # location
            flags,                # modifierFlags
            0,                    # timestamp
            0,                    # windowNumber
            0,                    # context
            8,                    # subtype  (8 = key event)
            data1,                # data1    encodes key + direction
            -1,                   # data2
        )
        Quartz.CGEventPost(0, ev.CGEvent())


def _media_key(key_type: int) -> Callable[[], None]:
    """Return a handler that posts a Quartz HID media-key event."""
    return lambda: _send_media_key(key_type)


def _applescript(script: str) -> Callable[[], None]:
    """
    Return a non-blocking handler that runs an osascript one-liner.
    Popen (fire-and-forget) avoids stalling the camera loop.
    """
    def _run() -> None:
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return _run


# ── action registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    label:        str                  # shown in the HUD overlay (keep ≤ ~10 chars)
    handler:      Callable[[], None]   # called when the gesture fires
    cooldown:     float                # seconds to lock out after firing
    hold_seconds: float                # gesture must be held this long before firing (0 = instant)


ACTIONS: dict[str, Action] = {
    "thumbs_up":      Action("Vol +",      _media_key(_NX_KEYTYPE_SOUND_UP),   cooldown=1.0, hold_seconds=0.0),
    "thumbs_down":    Action("Vol -",      _media_key(_NX_KEYTYPE_SOUND_DOWN), cooldown=1.0, hold_seconds=0.0),
    "fist":           Action("Play/Pause", _media_key(_NX_KEYTYPE_PLAY),       cooldown=3.0, hold_seconds=0.35),
    # HID track-skip events (17/18) are silently dropped by Apple Music on Tahoe;
    # AppleScript targets Music.app directly and works regardless of focus.
    "pointing_right": Action("Prev Track", _applescript('tell application "Music" to previous track'), cooldown=2.0, hold_seconds=0.0),
    "pointing_left":  Action("Next Track", _applescript('tell application "Music" to next track'),     cooldown=2.0, hold_seconds=0.0),
}

# ── cooldown state ─────────────────────────────────────────────────────────────
_last_fired:    float = -999.0   # far enough in the past to fire immediately
_last_cooldown: float = 1.0      # cooldown duration of the last fired action

# ── hold state ────────────────────────────────────────────────────────────────
_hold_gesture: str   = ""    # gesture currently being held
_hold_since:   float = 0.0   # when the current hold started


# ── public API ────────────────────────────────────────────────────────────────

def cooldown_fraction() -> float:
    """
    Progress through the post-fire cooldown window.
      0.0 — just fired, fully locked out
      1.0 — cooldown elapsed, ready to fire again
    """
    return min((time.monotonic() - _last_fired) / _last_cooldown, 1.0)


def hold_fraction() -> float:
    """
    Progress through the hold-duration requirement for the current gesture.
      0.0 — hold just started
      1.0 — threshold met (or gesture has no hold requirement)
    Useful for drawing a hold-progress indicator in the HUD.
    """
    if _hold_gesture not in ACTIONS:
        return 0.0
    required = ACTIONS[_hold_gesture].hold_seconds
    if required <= 0:
        return 1.0
    return min((time.monotonic() - _hold_since) / required, 1.0)


def reset_hold() -> None:
    """
    Reset the hold timer. Call whenever the active gesture disappears —
    e.g. confidence drops, hand leaves frame — so a partial hold does not
    carry over when the gesture reappears.
    """
    global _hold_gesture
    _hold_gesture = ""


def try_fire(gesture: str) -> str | None:
    """
    Per-frame call from the recognition loop.

    • Tracks how long `gesture` has been continuously detected.
    • Returns None until the gesture's hold_seconds threshold is met.
    • Then fires once the post-fire cooldown has also elapsed.
    • Resets the hold timer automatically when the gesture changes.
    """
    global _last_fired, _last_cooldown, _hold_gesture, _hold_since

    _dbg = gesture == "pointing_left"

    if gesture not in ACTIONS:
        if _dbg:
            print(f"[DBG] pointing_left: not in ACTIONS", flush=True)
        _hold_gesture = ""   # gesture switched to something unmapped
        return None

    # Start (or continue) hold tracking
    now = time.monotonic()
    if gesture != _hold_gesture:
        _hold_gesture = gesture
        _hold_since   = now

    action = ACTIONS[gesture]

    # Gate 1 — hold duration not yet met
    if action.hold_seconds > 0 and (now - _hold_since) < action.hold_seconds:
        if _dbg:
            print(f"[DBG] pointing_left: gate1 hold {now - _hold_since:.2f}s / {action.hold_seconds}s", flush=True)
        return None

    # Gate 2 — post-fire cooldown still active
    cd = cooldown_fraction()
    if cd < 1.0:
        if _dbg:
            print(f"[DBG] pointing_left: gate2 cooldown {cd:.2f}", flush=True)
        return None

    if _dbg:
        print(f"[DBG] pointing_left: firing handler", flush=True)
    action.handler()
    _last_fired    = now
    _last_cooldown = action.cooldown
    return action.label
