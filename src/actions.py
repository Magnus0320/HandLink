"""
Gesture → macOS media action mapping.

Media controls are implemented via CoreGraphics HID events (pyobjc-framework-Quartz).
NX_KEYTYPE_* constants are posted as NSSystemDefined events through CGEventPost,
which is identical to a physical media key press and requires no special permissions.

Dependency:  pip install pyobjc-framework-Quartz

Public API:
    try_fire(gesture)   → str | None   fire action, return label, or None
    cooldown_fraction() → float        0.0 = just fired, 1.0 = ready
    hold_fraction()     → float        0.0 = hold just started, 1.0 = threshold met
    reset_hold()        → None         call when the active gesture disappears
"""

import time
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


# ── action registry ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    label:        str    # shown in the HUD overlay (keep ≤ ~10 chars)
    key_type:     int    # NX_KEYTYPE_* constant
    cooldown:     float  # seconds to lock out after firing
    hold_seconds: float  # gesture must be held this long before firing (0 = instant)


ACTIONS: dict[str, Action] = {
    "thumbs_up":   Action("Vol +",      _NX_KEYTYPE_SOUND_UP,   cooldown=1.0, hold_seconds=0.0),
    "thumbs_down": Action("Vol -",      _NX_KEYTYPE_SOUND_DOWN, cooldown=1.0, hold_seconds=0.0),
    "fist":        Action("Play/Pause", _NX_KEYTYPE_PLAY,       cooldown=3.0, hold_seconds=0.35),
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

    if gesture not in ACTIONS:
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
        return None

    # Gate 2 — post-fire cooldown still active
    if cooldown_fraction() < 1.0:
        return None

    _send_media_key(action.key_type)
    _last_fired    = now
    _last_cooldown = action.cooldown
    return action.label
