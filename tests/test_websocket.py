"""Tests for the Tornado AC WebSocket component."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import aiohttp
import pytest

from custom_components.tornado.aux_cloud.aux_cloud_ws import (
    AuxCloudWebSocket,
    WEBSOCKET_SERVER_URL_EU,
    WEBSOCKET_SERVER_URL_USA,
    WEBSOCKET_SERVER_URL_CN,
)

# Mock data for testing
MOCK_HEADERS = {
    "Authorization": "Bearer test_token",
    "Content-Type": "application/json",
}
MOCK_LOGIN_SESSION = "test_login_session"
MOCK_USER_ID = "test_user_id"


@pytest.fixture
def mock_websocket_response():
    """Create a mock WebSocket response."""
    mock_ws = MagicMock()
    mock_ws.closed = False
    mock_ws.close = AsyncMock()
    mock_ws.send_str = AsyncMock()
    return mock_ws


@pytest.fixture
def mock_client_session():
    """Create a mock aiohttp ClientSession."""
    mock_session = MagicMock()
    mock_session.ws_connect = AsyncMock()
    return mock_session


class TestAuxCloudWebSocketInitialization:
    """Test WebSocket initialization and configuration."""

    def test_websocket_initialization_eu_region(self):
        """Test WebSocket initialization with EU region."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        assert ws.websocket_url == WEBSOCKET_SERVER_URL_EU
        assert ws.headers == MOCK_HEADERS
        assert ws.loginsession == MOCK_LOGIN_SESSION
        assert ws.userid == MOCK_USER_ID
        assert ws.websocket is None
        assert ws._listeners == []
        assert ws._reconnect_task is None
        assert ws.api_initialized is False

    def test_websocket_initialization_usa_region(self):
        """Test WebSocket initialization with USA region."""
        ws = AuxCloudWebSocket("usa", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        assert ws.websocket_url == WEBSOCKET_SERVER_URL_USA

    def test_websocket_initialization_cn_region(self):
        """Test WebSocket initialization with China region."""
        ws = AuxCloudWebSocket("cn", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        assert ws.websocket_url == WEBSOCKET_SERVER_URL_CN

    def test_websocket_initialization_unknown_region(self):
        """Test WebSocket initialization with unknown region defaults to CN."""
        ws = AuxCloudWebSocket("unknown", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        assert ws.websocket_url == WEBSOCKET_SERVER_URL_CN


class TestAuxCloudWebSocketConnection:
    """Test WebSocket connection establishment and management."""

    @patch("aiohttp.ClientSession")
    async def test_initialize_websocket_success(
        self, mock_session_class, mock_websocket_response
    ):
        """Test successful WebSocket initialization."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_websocket_response)
        mock_session_class.return_value = mock_session
        
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Mock send_data to avoid actual sending
        ws.send_data = AsyncMock()
        
        try:
            # Patch asyncio.create_task to return a completed mock task
            def mock_create_task(coro):
                """Create a mock task that doesn't leave unawaited coroutines."""
                # Close the coroutine to prevent warnings
                coro.close()
                # Return a mock task
                mock_task = AsyncMock()
                mock_task.done.return_value = True
                return mock_task
            
            with patch("asyncio.create_task", side_effect=mock_create_task):
                await ws.initialize_websocket()
                
                # Verify connection was established
                expected_url = f"{WEBSOCKET_SERVER_URL_EU}/appsync/apprelay/relayconnect"
                mock_session.ws_connect.assert_called_once_with(
                    expected_url, headers=MOCK_HEADERS
                )
                
                # Verify WebSocket is stored
                assert ws.websocket == mock_websocket_response
                
                # Verify initialization message was sent
                ws.send_data.assert_called_once()
            
        finally:
            # Clean up the websocket to prevent warnings
            await ws.close_websocket()

    @patch("aiohttp.ClientSession")
    async def test_initialize_websocket_connection_failure(self, mock_session_class):
        """Test WebSocket initialization failure handling."""
        # Setup mock to raise exception
        mock_session = MagicMock()
        mock_session.ws_connect = AsyncMock(side_effect=ConnectionError("Connection failed"))
        mock_session_class.return_value = mock_session
        
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Mock _schedule_reconnect to avoid actual reconnection
        ws._schedule_reconnect = AsyncMock()
        
        await ws.initialize_websocket()
        
        # Verify reconnection was scheduled
        ws._schedule_reconnect.assert_called_once()

    async def test_send_data_success(self, mock_websocket_response):
        """Test successful data sending."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        
        test_data = {"msgtype": "test", "data": "test_value"}
        
        await ws.send_data(test_data)
        
        expected_json = json.dumps(test_data)
        mock_websocket_response.send_str.assert_called_once_with(expected_json)

    async def test_send_data_websocket_not_connected(self):
        """Test sending data when WebSocket is not connected."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        with pytest.raises(ConnectionError, match="WebSocket is not connected"):
            await ws.send_data({"test": "data"})

    async def test_send_data_websocket_closed(self, mock_websocket_response):
        """Test sending data when WebSocket is closed."""
        mock_websocket_response.closed = True
        
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        
        with pytest.raises(ConnectionError, match="WebSocket is not connected"):
            await ws.send_data({"test": "data"})

    async def test_send_data_websocket_send_failure(self, mock_websocket_response):
        """Test handling WebSocket send failure."""
        mock_websocket_response.send_str = AsyncMock(side_effect=Exception("Send failed"))
        
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        
        with pytest.raises(Exception, match="Send failed"):
            await ws.send_data({"test": "data"})


class TestAuxCloudWebSocketMessageHandling:
    """Test WebSocket message handling and processing."""

    async def test_listen_to_websocket_init_message(self):
        """Test handling of initialization acknowledgment message."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Mock WebSocket message
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = json.dumps({
            "status": 0,
            "msgtype": "initk",
            "data": "initialization_acknowledged"
        })
        
        # Create async iterator that yields the message
        class MockAsyncIterator:
            def __init__(self, messages):
                self.messages = messages
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.messages):
                    raise StopAsyncIteration
                msg = self.messages[self.index]
                self.index += 1
                return msg
        
        # Mock the websocket
        ws.websocket = MockAsyncIterator([mock_msg])
        ws._schedule_reconnect = AsyncMock()
        
        await ws._listen_to_websocket()
        
        # Verify API was marked as initialized
        assert ws.api_initialized is True

    async def test_listen_to_websocket_ping_response(self):
        """Test handling of ping response message."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Mock WebSocket message
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = json.dumps({
            "status": 0,
            "msgtype": "pingk",
            "data": "pong"
        })
        
        # Create async iterator that yields the message
        async def mock_iterator():
            yield mock_msg
        
        # Mock the websocket
        mock_websocket = MagicMock()
        mock_websocket.__aiter__ = mock_iterator
        ws.websocket = mock_websocket
        ws._schedule_reconnect = AsyncMock()
        
        await ws._listen_to_websocket()
        
        # Verify reconnection was scheduled (normal flow after message processing)
        ws._schedule_reconnect.assert_called_once()

    async def test_listen_to_websocket_error_status(self):
        """Test handling of error status in messages."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Mock WebSocket message with error status
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = json.dumps({
            "status": 1,  # Error status
            "msgtype": "initk",
            "data": "error"
        })
        
        # Create async iterator that yields the message
        class MockAsyncIterator:
            def __init__(self, messages):
                self.messages = messages
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.messages):
                    raise StopAsyncIteration
                msg = self.messages[self.index]
                self.index += 1
                return msg
        
        # Mock the websocket
        ws.websocket = MockAsyncIterator([mock_msg])
        ws.close_websocket = AsyncMock()
        ws._schedule_reconnect = AsyncMock()
        
        await ws._listen_to_websocket()
        
        # Verify connection was closed and reconnection scheduled
        # Note: _schedule_reconnect is called once for the error status and once in finally block
        ws.close_websocket.assert_called_once()
        assert ws._schedule_reconnect.call_count == 2

    async def test_listen_to_websocket_regular_message(self):
        """Test handling of regular data messages."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Add a mock listener
        mock_listener = AsyncMock()
        ws.add_websocket_listener(mock_listener)
        
        # Mock WebSocket message
        test_message_data = {
            "status": 0,
            "msgtype": "data",
            "payload": {"device_id": "test123", "temperature": 25}
        }
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.TEXT
        mock_msg.data = json.dumps(test_message_data)
        
        # Create async iterator that yields the message
        class MockAsyncIterator:
            def __init__(self, messages):
                self.messages = messages
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.messages):
                    raise StopAsyncIteration
                msg = self.messages[self.index]
                self.index += 1
                return msg
        
        # Mock the websocket
        ws.websocket = MockAsyncIterator([mock_msg])
        ws._schedule_reconnect = AsyncMock()
        
        await ws._listen_to_websocket()
        
        # Verify listener was called with the message data
        mock_listener.assert_called_once_with(test_message_data)

    async def test_listen_to_websocket_websocket_error_message(self):
        """Test handling of WebSocket error messages."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Mock WebSocket error message
        mock_msg = MagicMock()
        mock_msg.type = aiohttp.WSMsgType.ERROR
        mock_msg.data = "WebSocket error occurred"
        
        # Create a mock async iterator
        class MockAsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item
        
        # Mock the websocket using MockAsyncIterator
        ws.websocket = MockAsyncIterator([mock_msg])
        ws._schedule_reconnect = AsyncMock()
        
        await ws._listen_to_websocket()
        
        # Verify reconnection was scheduled
        ws._schedule_reconnect.assert_called_once()

    async def test_listen_to_websocket_exception_handling(self):
        """Test handling of exceptions during message listening."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Create a mock async iterator that raises an exception
        class MockAsyncIterator:
            def __init__(self, exception):
                self.exception = exception
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                raise self.exception
        
        # Mock the websocket to raise an exception
        ws.websocket = MockAsyncIterator(Exception("Connection lost"))
        ws._schedule_reconnect = AsyncMock()
        
        await ws._listen_to_websocket()
        
        # Verify reconnection was scheduled
        ws._schedule_reconnect.assert_called_once()


class TestAuxCloudWebSocketListeners:
    """Test WebSocket listener functionality."""

    def test_add_websocket_listener(self):
        """Test adding a WebSocket listener."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        mock_listener = AsyncMock()
        ws.add_websocket_listener(mock_listener)
        
        assert mock_listener in ws._listeners

    async def test_notify_listeners_success(self):
        """Test successful notification of listeners."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Add multiple mock listeners
        mock_listener1 = AsyncMock()
        mock_listener2 = AsyncMock()
        ws.add_websocket_listener(mock_listener1)
        ws.add_websocket_listener(mock_listener2)
        
        test_message = {"msgtype": "test", "data": "test_data"}
        
        await ws._notify_listeners(test_message)
        
        # Verify both listeners were called with the message
        mock_listener1.assert_called_once_with(test_message)
        mock_listener2.assert_called_once_with(test_message)

    async def test_notify_listeners_with_exception(self):
        """Test listener notification when one listener raises an exception."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Add listeners - one that fails and one that succeeds
        mock_listener1 = AsyncMock(side_effect=Exception("Listener error"))
        mock_listener2 = AsyncMock()
        ws.add_websocket_listener(mock_listener1)
        ws.add_websocket_listener(mock_listener2)
        
        test_message = {"msgtype": "test", "data": "test_data"}
        
        # Should not raise exception despite listener1 failing
        await ws._notify_listeners(test_message)
        
        # Verify both listeners were called
        mock_listener1.assert_called_once_with(test_message)
        mock_listener2.assert_called_once_with(test_message)


class TestAuxCloudWebSocketKeepalive:
    """Test WebSocket keepalive functionality."""

    async def test_keepalive_websocket_success(self, mock_websocket_response):
        """Test successful keepalive message sending."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        ws.send_data = AsyncMock()
        
        await ws._keepalive_websocket()
        
        # Verify keepalive message was sent
        ws.send_data.assert_called_once()
        call_args = ws.send_data.call_args[0][0]
        assert call_args["msgtype"] == "ping"
        assert "messageid" in call_args

    async def test_keepalive_websocket_closed(self):
        """Test keepalive when WebSocket is closed."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        mock_websocket = MagicMock()
        mock_websocket.closed = True
        ws.websocket = mock_websocket
        ws.send_data = AsyncMock()
        
        await ws._keepalive_websocket()
        
        # Verify no message was sent for closed WebSocket
        ws.send_data.assert_not_called()

    async def test_keepalive_websocket_send_failure(self, mock_websocket_response):
        """Test keepalive when sending fails."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        ws.send_data = AsyncMock(side_effect=Exception("Send failed"))
        ws._schedule_reconnect = AsyncMock()
        
        await ws._keepalive_websocket()
        
        # Verify reconnection was scheduled due to send failure
        ws._schedule_reconnect.assert_called_once()

    @patch("asyncio.sleep")
    async def test_keepalive_loop(self, mock_sleep, mock_websocket_response):
        """Test the keepalive loop."""
        mock_websocket_response.closed = False
        
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        ws._keepalive_websocket = AsyncMock()
        
        # Make the websocket appear closed after first iteration
        def side_effect(duration):
            mock_websocket_response.closed = True
        
        mock_sleep.side_effect = side_effect
        
        await ws._keepalive_loop()
        
        # Verify keepalive was called once and sleep was called
        ws._keepalive_websocket.assert_called_once()
        mock_sleep.assert_called_once_with(10)


class TestAuxCloudWebSocketReconnection:
    """Test WebSocket reconnection functionality."""

    async def test_schedule_reconnect_first_time(self):
        """Test scheduling reconnection for the first time."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Create a mock task creator that properly handles coroutines
        def mock_create_task(coro):
            """Create a mock task that doesn't leave unawaited coroutines."""
            # Close the coroutine to prevent warnings
            coro.close()
            # Return a mock task
            mock_task = MagicMock()
            mock_task.done.return_value = True
            mock_task.cancel.return_value = None  # cancel() should not return a coroutine
            return mock_task
        
        with patch("asyncio.create_task", side_effect=mock_create_task) as mock_create_task_patch:
            try:
                await ws._schedule_reconnect()
                
                # Verify reconnect task was created
                mock_create_task_patch.assert_called_once()
                assert ws._reconnect_task is not None
                assert not ws._stop_reconnect.is_set()
                
            finally:
                # Clean up the task to prevent warnings
                if ws._reconnect_task:
                    ws._reconnect_task.cancel()
                    ws._reconnect_task = None

    async def test_schedule_reconnect_already_scheduled(self):
        """Test that reconnection is not scheduled if already in progress."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws._reconnect_task = MagicMock()  # Simulate existing task
        
        with patch("asyncio.create_task") as mock_create_task:
            await ws._schedule_reconnect()
            
            # Verify no new task was created
            mock_create_task.assert_not_called()

    @patch("asyncio.sleep")
    async def test_reconnect_success(self, mock_sleep):
        """Test successful reconnection."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.initialize_websocket = AsyncMock()
        
        await ws._reconnect()
        
        # Verify initialization was called and task was cleared
        ws.initialize_websocket.assert_called_once()
        assert ws._reconnect_task is None

    @patch("asyncio.sleep")
    async def test_reconnect_retry_on_failure(self, mock_sleep):
        """Test reconnection retry on failure."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Make first attempt fail, second succeed
        ws.initialize_websocket = AsyncMock(
            side_effect=[ConnectionError("Failed"), None]
        )
        
        await ws._reconnect()
        
        # Verify initialization was called twice
        assert ws.initialize_websocket.call_count == 2
        # Verify sleep was called once (after first failure)
        mock_sleep.assert_called_once_with(10)

    @patch("asyncio.sleep")
    async def test_reconnect_stop_event(self, mock_sleep):
        """Test reconnection stops when stop event is set."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.initialize_websocket = AsyncMock(side_effect=ConnectionError("Failed"))
        
        # Set stop event after first sleep
        def stop_after_sleep(duration):
            ws._stop_reconnect.set()
        
        mock_sleep.side_effect = stop_after_sleep
        
        await ws._reconnect()
        
        # Verify only one attempt was made
        ws.initialize_websocket.assert_called_once()
        mock_sleep.assert_called_once_with(10)


class TestAuxCloudWebSocketCleanup:
    """Test WebSocket cleanup and closure functionality."""

    async def test_close_websocket_success(self, mock_websocket_response):
        """Test successful WebSocket closure."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        
        # Mock reconnect task
        mock_task = MagicMock()
        ws._reconnect_task = mock_task
        
        await ws.close_websocket()
        
        # Verify stop event was set
        assert ws._stop_reconnect.is_set()
        
        # Verify reconnect task was cancelled
        mock_task.cancel.assert_called_once()
        assert ws._reconnect_task is None
        
        # Verify websocket was closed
        mock_websocket_response.close.assert_called_once()
        assert ws.websocket is None

    async def test_close_websocket_no_existing_connection(self):
        """Test closing WebSocket when no connection exists."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Should not raise exception
        await ws.close_websocket()
        
        # Verify stop event was set
        assert ws._stop_reconnect.is_set()

    async def test_close_websocket_no_reconnect_task(self, mock_websocket_response):
        """Test closing WebSocket when no reconnect task exists."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        
        await ws.close_websocket()
        
        # Verify websocket was still closed
        mock_websocket_response.close.assert_called_once()
        assert ws.websocket is None


class TestAuxCloudWebSocketIntegration:
    """Test WebSocket integration scenarios."""

    @patch("aiohttp.ClientSession")
    @patch("asyncio.sleep")
    async def test_full_websocket_lifecycle(
        self, mock_sleep, mock_session_class, mock_websocket_response
    ):
        """Test complete WebSocket lifecycle from initialization to cleanup."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session.ws_connect = AsyncMock(return_value=mock_websocket_response)
        mock_session_class.return_value = mock_session
        
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Add a listener
        mock_listener = AsyncMock()
        ws.add_websocket_listener(mock_listener)
        
        try:
            # Patch asyncio.create_task to return completed mock tasks
            def mock_create_task(coro):
                """Create a mock task that doesn't leave unawaited coroutines."""
                # Close the coroutine to prevent warnings
                coro.close()
                # Return a mock task
                mock_task = AsyncMock()
                mock_task.done.return_value = True
                return mock_task
            
            with patch("asyncio.create_task", side_effect=mock_create_task):
                # Initialize WebSocket (will call send_data internally)
                with patch.object(ws, 'send_data', new_callable=AsyncMock) as mock_send_data:
                    await ws.initialize_websocket()
                    
                    # Verify initialization
                    assert ws.websocket == mock_websocket_response
                    assert mock_listener in ws._listeners
                    
                    # Verify initialization message was sent
                    mock_send_data.assert_called_once()
                
                # Test sending data manually
                test_data = {"msgtype": "test"}
                await ws.send_data(test_data)
                
                # Verify data was sent
                expected_json = json.dumps(test_data)
                mock_websocket_response.send_str.assert_called_with(expected_json)
            
        finally:
            # Test cleanup
            await ws.close_websocket()
            
            # Verify cleanup
            assert ws.websocket is None
            assert ws._stop_reconnect.is_set()

    async def test_message_flow_with_listeners(self):
        """Test complete message flow from WebSocket to listeners."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        
        # Add multiple listeners
        received_messages = []
        
        async def listener1(message):
            received_messages.append(("listener1", message))
            
        async def listener2(message):
            received_messages.append(("listener2", message))
        
        ws.add_websocket_listener(listener1)
        ws.add_websocket_listener(listener2)
        
        # Simulate receiving a message
        test_message = {
            "msgtype": "device_update",
            "payload": {"device_id": "test123", "status": "online"}
        }
        
        await ws._notify_listeners(test_message)
        
        # Verify both listeners received the message
        assert len(received_messages) == 2
        assert ("listener1", test_message) in received_messages
        assert ("listener2", test_message) in received_messages

    @patch("time.time", return_value=1234567890)
    async def test_message_id_generation(self, mock_time, mock_websocket_response):
        """Test that message IDs are generated correctly."""
        ws = AuxCloudWebSocket("eu", MOCK_HEADERS, MOCK_LOGIN_SESSION, MOCK_USER_ID)
        ws.websocket = mock_websocket_response
        
        await ws._keepalive_websocket()
        
        # Verify message was sent with correct timestamp-based ID
        expected_message_id = "1234567890000"
        ws.websocket.send_str.assert_called_once()
        
        # Parse the sent message to verify message ID
        sent_data = json.loads(ws.websocket.send_str.call_args[0][0])
        assert sent_data["messageid"] == expected_message_id
        assert sent_data["msgtype"] == "ping"
