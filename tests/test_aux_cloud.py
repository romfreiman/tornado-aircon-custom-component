"""Tests for AuxCloud API client."""

import json
import logging
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.tornado.aux_cloud import (
    AuxCloudAPI,
    AuxCloudApiError,
    AuxCloudAuthError,
)

_LOGGER = logging.getLogger(__name__)

MAX_LOGIN_RETRIES = 3  # Maximum number of login retry attempts
MIN_HOMES_COUNT = 2  # Minimum number of homes in test data


@pytest.fixture
def mock_response() -> MagicMock:
    """Mock aiohttp response."""
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock()
    response.text = AsyncMock()
    return response


@pytest.fixture
def mock_session() -> MagicMock:
    """Mock aiohttp ClientSession."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    session.closed = False
    return session


@pytest.fixture
async def api(mock_session: MagicMock) -> AsyncGenerator[AuxCloudAPI, None]:
    """Create API instance with mocked login and session."""
    try:
        api = AuxCloudAPI(
            "test@example.com", "password", session=mock_session, region="eu"
        )
        # Set required attributes that would normally be set during login
        api.loginsession = "test_session"
        api.userid = "test_user"
        yield api
    except Exception:
        _LOGGER.exception("Error in api fixture")
        raise
    finally:
        _LOGGER.info("Cleaning up API fixture resources")
        await api.cleanup()


@pytest.mark.asyncio
async def test_login_success(mock_session: MagicMock, mock_response: MagicMock) -> None:
    """Test successful login."""
    mock_session.post.return_value = mock_response
    api = AuxCloudAPI("test@example.com", "password", session=mock_session, region="eu")
    mock_response.text.return_value = json.dumps(
        {"status": 0, "loginsession": "session123", "userid": "user123"}
    )

    result = await api.login()

    assert result is True
    assert api.loginsession == "session123"
    assert api.userid == "user123"


@pytest.mark.asyncio
async def test_login_error_handling(
    mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test failed login using a fake client session that simulates a failed login."""
    mock_session.post.return_value = mock_response
    api = AuxCloudAPI(
        "test@example.com", "wrongpassword", session=mock_session, region="eu"
    )

    class FakeResponse:
        async def __aenter__(self) -> "FakeResponse":
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            pass

        async def text(self) -> str:
            # Return a JSON string simulating a failed login.
            return json.dumps({"status": 1, "msg": "Invalid credentials"})

    class FakeClientSession:
        async def __aenter__(self) -> "FakeClientSession":
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            pass

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            # Ignore url and kwargs as they're not used in the test implementation
            _ = url, kwargs
            return FakeResponse()

    # Patch aiohttp.ClientSession to use FakeClientSession.
    with (
        patch("aiohttp.ClientSession", FakeClientSession),
        pytest.raises(AuxCloudAuthError, match="Login failed: Invalid credentials"),
    ):
        # When login is called, the fake session/response chain will yield a failure.
        await api.login()


@pytest.mark.asyncio
async def test_list_families_success(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test successful family list retrieval."""
    mock_response.text.return_value = json.dumps(
        {
            "status": 0,
            "data": {
                "familyList": [
                    {"familyid": "test1", "name": "Home 1"},
                    {"familyid": "test2", "name": "Home 2"},
                ]
            },
        }
    )
    mock_session.post.return_value = mock_response

    # First call - should hit the API
    result1 = await api.list_families()
    assert len(result1) == MIN_HOMES_COUNT
    assert result1[0]["familyid"] == "test1"
    assert result1[1]["familyid"] == "test2"
    assert mock_session.post.call_count == 1

    # Second call within cache TTL - should return cached result
    result2 = await api.list_families()
    assert result2 == result1
    assert mock_session.post.call_count == 1  # No additional API calls

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_retry_success(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families succeeds after retry on login validation failure."""
    # First call fails with login validation error, second succeeds
    mock_response.text = AsyncMock(
        side_effect=[
            json.dumps({"status": api.LOGIN_VALIDATION_FAILED}),
            json.dumps(
                {
                    "status": 0,
                    "data": {
                        "familyList": [
                            {"familyid": "test1", "name": "Home 1"},
                            {"familyid": "test2", "name": "Home 2"},
                        ]
                    },
                }
            ),
        ]
    )
    mock_session.post.return_value = mock_response

    # Mock successful login
    api.login = AsyncMock(return_value=True)

    # Clear the cache before test
    api.list_families.cache_clear()

    result = await api.list_families()

    assert len(result) == MIN_HOMES_COUNT
    assert result[0]["familyid"] == "test1"
    assert result[1]["familyid"] == "test2"
    api.login.assert_called_once()

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_max_retries_exceeded(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families fails after exceeding max retries."""
    mock_session.post.return_value.__aenter__.return_value = mock_response
    # All calls fail with login validation error
    mock_response.text.return_value = json.dumps(
        {"status": api.LOGIN_VALIDATION_FAILED}
    )

    # Mock successful login but requests still fail
    api.login = AsyncMock(return_value=True)

    # Clear the cache before test
    api.list_families.cache_clear()

    with pytest.raises(
        AuxCloudAuthError, match="Login validation failed after retries"
    ):
        await api.list_families()

    assert api.login.call_count == MAX_LOGIN_RETRIES

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_empty_response(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families handles empty family list."""
    mock_response.text.return_value = json.dumps(
        {"status": 0, "data": {"familyList": []}}
    )
    mock_session.post.return_value = mock_response

    # Clear the cache before test
    api.list_families.cache_clear()

    result = await api.list_families()

    assert isinstance(result, list)
    assert len(result) == 0
    assert api.data == {}

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_network_error(
    api: AuxCloudAPI, mock_session: MagicMock
) -> None:
    """Test list_families handles network errors."""
    # Clear the cache before test
    api.list_families.cache_clear()

    mock_session.post.side_effect = TimeoutError("Connection timeout")

    with pytest.raises(TimeoutError, match="Connection timeout"):
        await api.list_families()

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_invalid_json(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families handles invalid JSON response."""
    # Clear the cache before test
    api.list_families.cache_clear()

    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_response.text.return_value = "Invalid JSON response"

    with pytest.raises(AuxCloudApiError):
        await api.list_families()

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_missing_login_session(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families handles missing login session."""
    # Clear the cache before test
    api.list_families.cache_clear()

    api.loginsession = None
    mock_response.text.return_value = json.dumps(
        {"status": 0, "data": {"familyList": [{"familyid": "test1", "name": "Home 1"}]}}
    )
    mock_session.post.return_value = mock_response

    # Mock successful login
    api.login = AsyncMock(return_value=True)

    result = await api.list_families()

    assert len(result) == 1
    assert result[0]["familyid"] == "test1"
    api.login.assert_called_once()

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_failure(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test failed family list retrieval."""
    # Clear the cache before test
    api.list_families.cache_clear()

    # Set up response mock
    mock_response.text = AsyncMock(
        return_value=json.dumps({"status": -1, "msg": "API Error"})
    )
    # Set up session mock
    mock_session.post.return_value = mock_response
    # Make login raise an exception
    api.loginsession = None
    api.userid = None
    api.login = AsyncMock(side_effect=AuxCloudAuthError("Login failed for test"))

    with pytest.raises(AuxCloudAuthError, match="Login failed for test"):
        await api.list_families()

    # Clear the cache for subsequent tests
    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_get_devices(api: AuxCloudAPI) -> None:
    """Test getting all devices."""
    families = [{"familyid": "abc123def456ghi789jkl012mno345p"}]
    devices = [
        {
            "endpointId": "00000000000000000000000000000001",
            "friendlyName": "Living Room",
            "mac": "aa:bb:cc:dd:ee:ff",
            "gatewayId": "",
            "productId": "000000000000000000000000c0620000",
            "icon": (
                "/staticfilesys/openlimit/queryfile"
                "?mtag=appmanage&mkey=1234567890abcdef"
            ),
            "roomId": "2000000000000000000",
            "order": 1,
            "cookie": (
                "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5Ijog"
                "ImtleTEifX0="
            ),
            "vGroup": "",
            "irData": "",
            "extend": "",
            "userId": "abc123def456ghi789jkl012mno345p",
            "familyid": "abc123def456ghi789jkl012mno345p",
            "v1moduleid": "00000000000000000000000000000001",
            "devicetypeFlag": 0,
            "devSession": "abcdef123456789",
            "createTime": "2024-01-01 00:00:00",
            "state": 1,
            "params": {
                "temp": 280,
                "ac_vdir": 0,
                "ac_hdir": 0,
                "ac_mark": 2,
                "ac_mode": 1,
                "ac_slp": 0,
                "pwr": 0,
                "ac_astheat": 0,
                "ecomode": 0,
                "ac_clean": 0,
                "ac_health": 0,
                "scrdisp": 1,
                "mldprf": 0,
                "pwrlimitswitch": 0,
                "pwrlimit": 0,
                "comfwind": 0,
                "sleepdiy": 1,
                "childlock": 0,
                "ac_tempconvert": 0,
                "tempunit": 1,
            },
        }
    ]
    device_state = {
        "event": {
            "payload": {
                "status": 0,
                "data": [
                    {
                        "properties": {
                            "power": "on",
                            "temp": 280,
                            "ac_mode": 1,
                            "ac_mark": 2,
                        }
                    }
                ],
            }
        }
    }
    device_params = {"temp": 280, "ac_mode": 1, "ac_mark": 2, "pwr": 0}
    ambient_mode = {"envtemp": 220}  # 22.0 degrees

    temp_constant = 280  # Define constant for temperature
    with (
        patch.object(api, "list_families", AsyncMock(return_value=families)),
        patch.object(api, "list_devices", AsyncMock(side_effect=[devices, []])),
        patch.object(
            api,
            "query_device_state",
            AsyncMock(return_value=device_state["event"]["payload"]),
        ),
        patch.object(
            api,
            "get_device_params",
            AsyncMock(side_effect=[device_params, ambient_mode]),
        ),
    ):
        result = await api.get_devices()

        assert len(result) == 1
        assert result[0]["endpointId"] == "00000000000000000000000000000001"
        assert result[0]["params"]["temp"] == temp_constant
        assert result[0]["icon"] == (
            "/staticfilesys/openlimit/queryfile?mtag=appmanage&mkey=1234567890abcdef"
        )


@pytest.mark.asyncio
async def test_query_device_state(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test querying device state."""
    mock_response.text.return_value = json.dumps(
        {"event": {"payload": {"status": 0, "data": [{"state": "on"}]}}}
    )
    mock_session.post.return_value = mock_response

    result = await api.query_device_state("dev1", "sess1")

    assert result["status"] == 0
    assert result["data"][0]["state"] == "on"


@pytest.mark.asyncio
async def test_set_device_params(api: AuxCloudAPI) -> None:
    """Test setting device parameters."""
    device = {
        "endpointId": "dev1",
        "productId": "prod1",
        "mac": "00:11:22:33:44:55",
        "devicetypeFlag": "flag1",
        "devSession": "sess1",
        "cookie": (
            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5IjogImtleTEifX0="
        ),
    }

    with patch.object(api, "_act_device_params", AsyncMock(return_value={"temp": 25})):
        result = await api.set_device_params(device, {"temp": 25})

        assert result == {"temp": 25}


@pytest.mark.asyncio
async def test_get_headers() -> None:
    """Test header generation."""
    mock_session = MagicMock()
    api = AuxCloudAPI("test@example.com", "password", session=mock_session, region="eu")
    headers = api._get_headers(custom="value")
    assert headers["custom"] == "value"
    assert headers["Content-Type"] == "application/x-java-serialized-object"


@pytest.mark.asyncio
async def test_act_device_params_get(api: AuxCloudAPI) -> None:
    """Test getting device parameters."""
    device = {
        "endpointId": "dev1",
        "productId": "prod1",
        "mac": "00:11:22:33:44:55",
        "devicetypeFlag": "flag1",
        "devSession": "sess1",
        "cookie": (
            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5IjogImtleTEifX0="
        ),
    }

    with patch.object(api, "_act_device_params", AsyncMock(return_value={"temp": 25})):
        result = await api.get_device_params(device)
        assert result == {"temp": 25}


@pytest.mark.asyncio
async def test_act_device_params_set(api: AuxCloudAPI) -> None:
    """Test setting device parameters."""
    device = {
        "endpointId": "dev1",
        "productId": "prod1",
        "mac": "00:11:22:33:44:55",
        "devicetypeFlag": "flag1",
        "devSession": "sess1",
        "cookie": (
            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5IjogImtleTEifX0="
        ),
    }

    with patch.object(api, "_act_device_params", AsyncMock(return_value={"temp": 25})):
        result = await api.set_device_params(device, {"temp": 25})
        assert result == {"temp": 25}


@pytest.mark.asyncio
async def test_is_ambient_mode(api: AuxCloudAPI) -> None:
    """Test ambient mode detection."""
    assert api._is_ambient_mode(["mode"]) is True
    assert api._is_ambient_mode(["temp"]) is False


@pytest.mark.asyncio
async def test_get_directive_header() -> None:
    """Test directive header generation."""
    api = AuxCloudAPI("test@example.com", "password", region="eu")
    directive = api._get_directive_header(
        namespace="DNA.TemperatureSensor",
        name="ReportState",
        message_id_prefix="test_user",
    )
    assert "namespace" in directive
    assert directive["namespace"] == "DNA.TemperatureSensor"
    assert "name" in directive
    assert directive["name"] == "ReportState"
    assert "messageId" in directive


@pytest.mark.asyncio
async def test_query_device_temperature_success(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test successful device temperature query."""
    mock_response.text.return_value = json.dumps(
        {"event": {"payload": {"status": 0, "temperature": 23.5}}}
    )
    mock_session.post.return_value = mock_response

    result = await api.query_device_temperature("dev1", "sess1")

    assert result == {"status": 0, "temperature": 23.5}
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_query_device_temperature_failure(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test device temperature query failure."""
    # Set up the mock to work with async context manager
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_response.text.return_value = json.dumps(
        {"event": {"payload": {"status": -1, "msg": "Temperature query failed"}}}
    )

    with pytest.raises(AuxCloudApiError, match="Failed to query device temperature"):
        await api.query_device_temperature("dev1", "sess1")


@pytest.mark.asyncio
async def test_act_device_params_mismatch(api: AuxCloudAPI) -> None:
    """Test act_device_params raising ValueError when params and vals len mismatch."""
    device = {
        "endpointId": "dev1",
        "productId": "prod1",
        "mac": "00:11:22:33:44:55",
        "devicetypeFlag": "flag1",
        "devSession": "sess1",
        # Using a valid cookie string as expected by the implementation.
        "cookie": (
            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5IjogImtleTEifX0="
        ),
    }
    with pytest.raises(ValueError, match="Params and Vals must have the same length"):
        # Directly call _act_device_params with mismatching params/vals.
        await api._act_device_params(device, "set", ["temp"], [])


@pytest.mark.asyncio
async def test_refresh_success(api: AuxCloudAPI) -> None:
    """Test refresh function when family and device fetching succeed."""
    test_families = [{"familyid": "fam1"}]
    # Patch list_families and list_devices to simulate successful fetches.
    with (
        patch.object(api, "list_families", AsyncMock(return_value=test_families)),
        patch.object(api, "list_devices", AsyncMock(return_value=[])),
    ):
        await api.refresh()  # Should complete without raising an exception.


@pytest.mark.asyncio
async def test_refresh_failure(api: AuxCloudAPI) -> None:
    """Test refresh function when list_families fails."""
    with (
        patch.object(
            api,
            "list_families",
            AsyncMock(side_effect=Exception("Failed to fetch families")),
        ),
        pytest.raises(Exception, match="Failed to fetch families"),
    ):
        await api.refresh()


@pytest.mark.asyncio
async def test_cleanup_owned_session(mock_session: MagicMock) -> None:
    """Test cleanup with owned session."""
    api = AuxCloudAPI(
        "test@example.com", "password", region="eu"
    )  # No session provided
    api.session = mock_session
    api._session_owner = True
    mock_session.close = AsyncMock()  # Make close() awaitable
    await api.cleanup()
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_external_session(mock_session: MagicMock) -> None:
    """Test cleanup with external session."""
    api = AuxCloudAPI("test@example.com", "password", session=mock_session, region="eu")
    mock_session.close = AsyncMock()  # Make close() awaitable
    await api.cleanup()
    mock_session.close.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_closed_session(mock_session: MagicMock) -> None:
    """Test cleanup with already closed session."""
    api = AuxCloudAPI("test@example.com", "password", region="eu")
    api.session = mock_session
    api._session_owner = True
    mock_session.closed = True
    mock_session.close = AsyncMock()
    await api.cleanup()
    mock_session.close.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_none_session() -> None:
    """Test cleanup with None session."""
    api = AuxCloudAPI("test@example.com", "password", region="eu")
    api.session = None
    await api.cleanup()  # Should not raise any exception
