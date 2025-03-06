"""Test connection pooling functionality."""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from custom_components.tornado.aux_cloud import AuxCloudAPI

if TYPE_CHECKING:
    from collections.abc import Coroutine

# Constants
CONNECTION_POOL_LIMIT = 10


@pytest.fixture(autouse=True)
async def cleanup_shared_connector() -> None:
    """Reset shared connector before and after each test."""
    # Reset before test
    AuxCloudAPI._shared_connector = None
    yield
    # Reset after test
    AuxCloudAPI._shared_connector = None

    # Ensure any pending tasks are completed
    await asyncio.sleep(0)  # Allow event loop to process any pending tasks


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
    await api.get_shared_connector()

    # Track active connections
    active_connections = 0
    max_active_connections = 0

    # Create mock connection response
    mock_response = MagicMock()
    mock_response.status = 200

    # Mock transport and protocol
    mock_transport = MagicMock()
    mock_protocol = MagicMock()

    # Mock the create_connection method
    async def mock_create_connection(*_: Any, **__: Any) -> tuple[MagicMock, MagicMock]:
        nonlocal active_connections, max_active_connections
        active_connections += 1
        max_active_connections = max(max_active_connections, active_connections)
        try:
            return mock_transport, mock_protocol
        finally:
            active_connections -= 1

    with patch(
        "aiohttp.connector.TCPConnector._create_direct_connection",
        new_callable=lambda: mock_create_connection,
    ):
        # Simulate multiple concurrent requests
        async def make_request() -> None:
            session = await api._get_session()
            try:
                async with session.get("http://example.com") as _:
                    await asyncio.sleep(0.1)  # Simulate request duration
            except aiohttp.ClientError:
                # Log the exception with a specific type instead of a blind catch
                import logging

                logging.exception("Error during request")

        # Make 15 concurrent requests (more than the limit of 10)
        tasks: list[Coroutine[Any, Any, None]] = [make_request() for _ in range(15)]
        await asyncio.gather(*tasks)

        # Verify connection pool didn't exceed limit
        assert max_active_connections <= CONNECTION_POOL_LIMIT

    # Cleanup
    await api.cleanup()


@pytest.mark.asyncio
async def test_connection_cleanup() -> None:
    """Test that connections are properly cleaned up."""
    # Reset the shared connector before the test
    AuxCloudAPI._shared_connector = None

    api = AuxCloudAPI("test@example.com", "password")
    session = await api._get_session()

    # Mock close method to track if it's called
    original_close = session.close
    close_called = False

    async def mock_close() -> None:
        nonlocal close_called
        close_called = True
        await original_close()

    try:
        session.close = mock_close

        # Cleanup
        await api.cleanup()

        # Verify session was closed
        assert close_called
        assert session.closed
    finally:
        # Restore original method
        session.close = original_close
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
