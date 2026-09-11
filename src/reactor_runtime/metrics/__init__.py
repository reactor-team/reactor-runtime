# Copyright (c) 2026 Reactor Technologies, Inc. All rights reserved.
"""Expose the shared registry and runtime metric groups."""

from prometheus_client import CONTENT_TYPE_LATEST

from reactor_runtime.metrics.command import UNKNOWN_COMMAND, CommandMetrics
from reactor_runtime.metrics.model import ModelMetrics
from reactor_runtime.metrics.registry import RuntimeMetrics
from reactor_runtime.metrics.session import MetricsRecorder

CONTENT_TYPE = CONTENT_TYPE_LATEST
"""The media type of a registry rendered in Prometheus text format."""

__all__ = [
    "CONTENT_TYPE",
    "UNKNOWN_COMMAND",
    "CommandMetrics",
    "MetricsRecorder",
    "ModelMetrics",
    "RuntimeMetrics",
]
