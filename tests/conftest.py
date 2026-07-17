"""Shared pytest fixtures for the test suite."""

import pycares
import pytest


@pytest.fixture(scope="session", autouse=True)
def _prewarm_pycares_shutdown_thread() -> None:
    """
    Start pycares' background channel-shutdown thread before any test runs.

    pycares >=4.9 (the fix for CVE-2025-48945, a Channel use-after-free) destroys
    DNS resolver Channels on a lazily-started, process-wide daemon thread
    ("_run_safe_shutdown_loop"). If that thread first spins up mid-suite (e.g.
    when a real aiohttp.TCPConnector/aiodns resolver is created in
    tests/test_connection_pool.py), pytest-homeassistant-custom-component's
    per-test thread-leak check flags it as a lingering thread and fails an
    unrelated test. Starting it here, once, before the first per-test
    before/after thread snapshot is taken, makes it invisible to that diff.
    """
    pycares.Channel()
