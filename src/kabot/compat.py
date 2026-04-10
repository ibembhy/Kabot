from __future__ import annotations

from datetime import timezone

try:
    from datetime import UTC as UTC
except ImportError:  # pragma: no cover
    UTC = timezone.utc
