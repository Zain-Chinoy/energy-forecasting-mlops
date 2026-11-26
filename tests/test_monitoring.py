import threading

import pytest

from src.monitoring import RequestStats


def test_empty_stats_are_zero():
    snap = RequestStats().snapshot()
    assert snap["total_requests"] == 0
    assert snap["avg_latency_ms"] == 0.0
    assert snap["by_endpoint"] == {}


def test_counts_and_average_latency():
    stats = RequestStats()
    stats.record("/predict", 200, 0.010)
    stats.record("/predict", 200, 0.030)
    stats.record("/explain", 200, 0.200)
    snap = stats.snapshot()
    assert snap["total_requests"] == 3
    assert snap["avg_latency_ms"] == pytest.approx(80.0)
    assert snap["max_latency_ms"] == pytest.approx(200.0)
    assert snap["by_endpoint"]["/predict"]["count"] == 2
    assert snap["by_endpoint"]["/predict"]["avg_latency_ms"] == pytest.approx(20.0)


def test_errors_are_counted_from_status_400():
    stats = RequestStats()
    for code in (200, 399, 400, 422, 500):
        stats.record("/predict", code, 0.001)
    snap = stats.snapshot()
    assert snap["error_requests"] == 3
    assert snap["by_endpoint"]["/predict"]["errors"] == 3


def test_recording_from_many_threads_loses_no_counts():
    stats = RequestStats()

    def worker():
        for _ in range(200):
            stats.record("/predict", 200, 0.001)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert stats.snapshot()["total_requests"] == 1600
