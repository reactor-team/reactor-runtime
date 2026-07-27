from __future__ import annotations

from prometheus_client import Counter

from reactor_runtime.metrics import CONTENT_TYPE, RuntimeMetrics


def test_publishes_the_identity_of_the_process() -> None:
    metrics = RuntimeMetrics(version="1.4.2", model="pipeline:Brightness")

    rendered = metrics.render().decode()

    assert 'runtime_info{model="pipeline:Brightness",version="1.4.2"} 1.0' in rendered


def test_renders_in_the_prometheus_text_format() -> None:
    metrics = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")

    assert CONTENT_TYPE.startswith("text/plain")
    assert metrics.render().startswith(b"# HELP runtime_info")


def test_each_holder_owns_its_own_registry() -> None:
    # Two holders in one process must not see each other's numbers. The registry
    # is a field of the holder rather than a module global, so a component that
    # observes on one leaves the other empty. Every test in this suite relies on
    # that: they build a holder each and read it back.
    first = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")
    second = RuntimeMetrics(version="0.0.0", model="pipeline:Echo")

    Counter("runtime_test_total", "A counter one holder carries.", registry=first.registry).inc()

    assert "runtime_test_total" in first.render().decode()
    assert "runtime_test_total" not in second.render().decode()


def test_the_registry_carries_only_the_runtime_instruments() -> None:
    # A dedicated registry, not the default one: the process reports what the
    # runtime declares, and the scraper adds what it knows about the machine.
    rendered = RuntimeMetrics(version="0.0.0", model="pipeline:Echo").render().decode()

    assert "python_gc_objects_collected_total" not in rendered
    assert "process_virtual_memory_bytes" not in rendered
