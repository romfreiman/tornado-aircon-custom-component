"""Test connection pooling functionality."""

from __future__ import annotations

import asyncio
import socket
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from custom_components.tornado.aux_cloud import AuxCloudAPI

# Constants
CONNECTION_POOL_LIMIT = 10


@pytest.fixture(autouse=True)
async def cleanup_shared_resources() -> None:
    """Reset shared resources before and after each test."""
    # Reset before test
    AuxCloudAPI._shared_connector = None
    AuxCloudAPI._shared_session = None

    # Clear any shared locks
    AuxCloudAPI._shared_connector_lock = asyncio.Lock()
    AuxCloudAPI._shared_session_lock = asyncio.Lock()

    yield

    # Clean up after test
    await AuxCloudAPI.cleanup_shared_resources()
    AuxCloudAPI._shared_connector = None
    AuxCloudAPI._shared_session = None

    # Allow event loop to process any pending tasks
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_shared_connector_singleton() -> None:
    """Test that shared connector is created only once."""
    # Create multiple API instances
    api1 = AuxCloudAPI("test1@example.com", "password1")
    api2 = AuxCloudAPI("test2@example.com", "password2")

    # Get connector from both instances
    connector1 = await api1.get_shared_connector()
    connector2 = await api2.get_shared_connector()

    # Verify both instances share the same connector
    assert connector1 is connector2
    assert isinstance(connector1, aiohttp.TCPConnector)

    # Verify connector settings
    assert connector1.limit == CONNECTION_POOL_LIMIT
    assert connector1.use_dns_cache is True  # Changed from ttl_dns_cache
    assert connector1.family == socket.AF_INET

    # Cleanup
    await api1.cleanup()
    await api2.cleanup()


@pytest.mark.asyncio
async def test_connector_reuse_across_sessions() -> None:
    """Test that sessions reuse the shared connector."""
    api1 = AuxCloudAPI("test1@example.com", "password1")
    api2 = AuxCloudAPI("test2@example.com", "password2")

    # Get sessions from both instances
    session1 = await api1._get_session()
    session2 = await api2._get_session()

    # Verify both sessions use the same connector
    assert session1.connector is session2.connector

    # Cleanup
    await api1.cleanup()
    await api2.cleanup()


@pytest.mark.asyncio
async def test_connection_pool_limits() -> None:
    """Test that connection pool respects limits."""
    api = AuxCloudAPI("test@example.com", "password")

    # Track connections
    active_connections = 0
    connection_lock = asyncio.Lock()
    connection_event = asyncio.Event()

    async def mock_create_connection(*_: Any, **__: Any) -> tuple[MagicMock, MagicMock]:
        nonlocal active_connections
        async with connection_lock:
            active_connections += 1
            if active_connections >= CONNECTION_POOL_LIMIT:
                connection_event.set()

        try:
            await asyncio.sleep(0.1)  # Simulate network delay
            return MagicMock(), MagicMock()
        finally:
            async with connection_lock:
                active_connections -= 1

    with patch(
        "aiohttp.connector.TCPConnector._create_direct_connection",
        new_callable=lambda: mock_create_connection,
    ):

        async def make_request() -> None:
            session = await api._get_session()
            try:
                async with session.get("http://example.com"):
                    await asyncio.sleep(0.1)
            except aiohttp.ClientError:
                pass

        # Create requests
        tasks = []
        for _ in range(CONNECTION_POOL_LIMIT + 5):
            task = asyncio.create_task(make_request())
            tasks.append(task)
            await asyncio.sleep(0)  # Allow other tasks to run

        try:
            # Wait for pool to reach limit
            await asyncio.wait_for(connection_event.wait(), timeout=5.0)

            # Verify connection count
            async with connection_lock:
                assert active_connections <= CONNECTION_POOL_LIMIT, (
                    f"Connection limit exceeded: "
                    f"{active_connections} > {CONNECTION_POOL_LIMIT}"
                )
        finally:
            # Clean up
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await api.cleanup()
            await AuxCloudAPI.cleanup_shared_resources()

        # Verify cleanup
        assert api.session is None
        assert api._cleaned_up is True


@pytest.mark.asyncio
async def test_connection_cleanup() -> None:
    """Test that connections are properly cleaned up."""
    # Reset the shared connector before the test
    AuxCloudAPI._shared_connector = None

    api = AuxCloudAPI("test@example.com", "password")
    session = await api._get_session()

    # Mock close method to track if it's called
    close_called = False

    async def mock_close() -> None:
        nonlocal close_called
        close_called = True
        await session.connector.close()

    # Store original close method
    original_close = session.close

    # Replace close method with our mock
    session.close = mock_close

    try:
        # Cleanup instance
        await api.cleanup()

        # Verify instance cleanup behavior
        assert not close_called, (
            "Session close should not be called during instance cleanup"
        )
        assert api.session is None, "Session reference should be cleared"
        assert api._cleaned_up is True, "API should be marked as cleaned up"

        # Now test shared resources cleanup
        await AuxCloudAPI.cleanup_shared_resources()

        # Verify shared resources cleanup behavior
        assert close_called, (
            "Session close should be called during shared resources cleanup"
        )
    finally:
        # Restore original method
        session.close = original_close
        # Ensure we clean up the connector
        if not session.connector.closed:
            await session.connector.close()
        # Reset the shared connector after the test
        AuxCloudAPI._shared_connector = None


@pytest.mark.asyncio
async def test_session_reuse() -> None:
    """Test that the same session is reused for multiple requests."""
    api = AuxCloudAPI("test@example.com", "password")

    # Get session multiple times
    session1 = await api._get_session()
    session2 = await api._get_session()
    session3 = await api._get_session()

    # Verify same session instance is returned
    assert session1 is session2 is session3

    # Cleanup
    await api.cleanup()


@pytest.mark.asyncio
async def test_dns_cache() -> None:
    """Test that DNS cache is working."""
    # Reset the shared connector before the test
    AuxCloudAPI._shared_connector = None

    api = AuxCloudAPI("test@example.com", "password")

    # Get the shared connector
    connector = await api.get_shared_connector()

    # Counter for DNS lookups
    dns_lookups = 0

    # Create a mock resolver
    original_resolve_method = connector._resolver.resolve

    async def mock_resolver_resolve(host: str, port: int, *, family: int = 0) -> Any:
        nonlocal dns_lookups
        dns_lookups += 1
        # Return the original result to maintain compatibility
        return await original_resolve_method(host, port, family=family)

    try:
        # Patch the resolve method directly on the connector's resolver instance
        connector._resolver.resolve = mock_resolver_resolve

        # Test resolution with the same host and port
        host = "example.com"
        port = 80

        # First resolution should trigger a DNS lookup
        await connector._resolve_host(host, port)
        assert dns_lookups == 1, "First request should trigger DNS lookup"

        # Subsequent resolutions should use the cache
        await connector._resolve_host(host, port)
        await connector._resolve_host(host, port)
        assert dns_lookups == 1, "DNS cache not working - multiple lookups performed"
    finally:
        # Restore original method
        connector._resolver.resolve = original_resolve_method
        # Cleanup
        await api.cleanup()
        # Reset the shared connector after the test
        AuxCloudAPI._shared_connector = None
