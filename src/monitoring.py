"""Minimal in-process monitoring: request counts and latency per endpoint.

Exposed through ``GET /stats``. Infrastructure endpoints (health checks, docs, the stats
page itself) are not counted, so platform health pings do not drown out real traffic.
The numbers live in memory and reset when the service restarts.
"""

import logging
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

IGNORED_PATHS = frozenset({"/health", "/stats", "/docs", "/redoc", "/openapi.json", "/favicon.ico"})
UNMATCHED = "unmatched"  # requests that hit no route (404s), grouped to keep the output small


def get_request_logger() -> logging.Logger:
    """Logger that prints one line per request (visible in ``docker logs`` and on Render)."""
    log = logging.getLogger("api.requests")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


@dataclass
class _EndpointStats:
    count: int = 0
    errors: int = 0
    total_latency_s: float = 0.0
    max_latency_s: float = 0.0


class RequestStats:
    """Thread-safe request counter with average and maximum latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_endpoint: dict[str, _EndpointStats] = defaultdict(_EndpointStats)
        self._started = time.time()

    def record(self, endpoint: str, status_code: int, duration_s: float) -> None:
        """Record one finished request. Status 400 and above counts as an error."""
        with self._lock:
            entry = self._by_endpoint[endpoint]
            entry.count += 1
            entry.errors += int(status_code >= 400)
            entry.total_latency_s += duration_s
            entry.max_latency_s = max(entry.max_latency_s, duration_s)

    def snapshot(self) -> dict:
        """Return totals and a per-endpoint breakdown (latencies in milliseconds)."""
        with self._lock:
            entries = {name: _EndpointStats(**vars(e)) for name, e in self._by_endpoint.items()}
        total = sum(e.count for e in entries.values())
        errors = sum(e.errors for e in entries.values())
        latency = sum(e.total_latency_s for e in entries.values())
        return {
            "uptime_seconds": round(time.time() - self._started, 1),
            "total_requests": total,
            "error_requests": errors,
            "avg_latency_ms": round(1000 * latency / total, 2) if total else 0.0,
            "max_latency_ms": round(
                1000 * max((e.max_latency_s for e in entries.values()), default=0.0), 2
            ),
            "by_endpoint": {
                name: {
                    "count": e.count,
                    "errors": e.errors,
                    "avg_latency_ms": round(1000 * e.total_latency_s / e.count, 2),
                }
                for name, e in sorted(entries.items())
            },
            "note": "Health checks, /stats and /docs are not counted. Resets on restart.",
        }
