"""Tests for AuxCloud API client."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import tenacity

from custom_components.tornado.aux_cloud import (
    AuxCloudAPI,
    AuxCloudAuthError,
)

_LOGGER = logging.getLogger(__name__)

CONNECTION_POOL_LIMIT = 10  # Connection pool limit

MAX_LOGIN_RETRIES = 3  # Maximum number of login retry attempts
NUM_OF_API_RETRIES = 3  # Number of retries for network errors
MIN_HOMES_COUNT = 2  # Minimum number of homes in test data

TEST_TEMPERATURE = 25
TEST_AMBIENT_TEMP = 22
TEST_SHARED_DEVICES_CALL_COUNT = 3
TEST_MIN_DEVICE_COUNT = 2


@pytest.fixture
def mock_response() -> MagicMock:
    """Mock aiohttp response."""
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock()
    response.text = AsyncMock()
    return response


@pytest.fixture
def mock_connector() -> MagicMock:
    """Mock TCP connector."""
    connector = MagicMock()
    connector.closed = False
    connector.close = AsyncMock()  # Make close() awaitable
    connector._acquired = set()  # Mock the _acquired set
    connector.size = 100
    connector.acquired = 0
    return connector


@pytest.fixture
def mock_session(mock_connector: MagicMock) -> MagicMock:
    """Mock aiohttp ClientSession."""
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock()
    session.closed = False
    session.close = AsyncMock()  # Make close() awaitable
    session.connector = mock_connector
    return session


@pytest.fixture
async def api(mock_session: MagicMock) -> AsyncGenerator[AuxCloudAPI, None]:
    """Create API instance with mocked session."""
    try:
        with patch.object(
            AuxCloudAPI, "get_shared_session", AsyncMock(return_value=mock_session)
        ):
            api = AuxCloudAPI(
                "test@example.com", "password", session=mock_session, region="eu"
            )
            api.loginsession = "test_session"
            api.userid = "test_user"
            yield api
    finally:
        await api.cleanup()


@pytest.mark.asyncio
async def test_login_success(mock_session: MagicMock, mock_response: MagicMock) -> None:
    """Test successful login."""
    mock_response.text.return_value = json.dumps(
        {"status": 0, "loginsession": "session123", "userid": "user123"}
    )
    mock_session.post.return_value = mock_response

    with patch.object(
        AuxCloudAPI, "get_shared_session", AsyncMock(return_value=mock_session)
    ):
        api = AuxCloudAPI(
            "test@example.com", "password", session=mock_session, region="eu"
        )

        result = await api.login()

        assert result is True
        assert api.loginsession == "session123"
        assert api.userid == "user123"

        mock_session.post.assert_called_once()
        await api.cleanup()


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
            return json.dumps({"status": 1, "msg": "Invalid credentials"})

    class FakeClientSession:
        def __init__(self, **kwargs: Any) -> None:
            pass

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
            _ = url, kwargs
            return FakeResponse()

    with (
        patch("aiohttp.ClientSession", FakeClientSession),
        pytest.raises(AuxCloudAuthError, match="Login failed: Invalid credentials"),
    ):
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

    result1 = await api.list_families()
    assert len(result1) == MIN_HOMES_COUNT
    assert result1[0]["familyid"] == "test1"
    assert result1[1]["familyid"] == "test2"
    assert mock_session.post.call_count == 1

    result2 = await api.list_families()
    assert result2 == result1
    assert mock_session.post.call_count == 1

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_retry_success(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families succeeds after retry on login validation failure."""
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

    api.login = AsyncMock(return_value=True)

    api.list_families.cache_clear()

    result = await api.list_families()

    assert len(result) == MIN_HOMES_COUNT
    assert result[0]["familyid"] == "test1"
    assert result[1]["familyid"] == "test2"
    api.login.assert_called_once()

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_max_retries_exceeded(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families fails after exceeding max retries."""
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_response.text.return_value = json.dumps(
        {"status": api.LOGIN_VALIDATION_FAILED}
    )

    api.login = AsyncMock(return_value=True)

    api.list_families.cache_clear()

    with pytest.raises(
        AuxCloudAuthError, match="Login validation failed after retries"
    ):
        await api.list_families()

    assert api.login.call_count == MAX_LOGIN_RETRIES

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

    api.list_families.cache_clear()

    result = await api.list_families()

    assert isinstance(result, list)
    assert len(result) == 0
    assert api.data == {}

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_network_error(
    api: AuxCloudAPI, mock_session: MagicMock
) -> None:
    """Test list_families handles network errors with retry logic."""
    try:
        api.list_families.cache_clear()

        mock_session.post.side_effect = TimeoutError("Connection timeout")

        with pytest.raises(tenacity.RetryError) as exc_info:
            await api.list_families()

        assert isinstance(exc_info.value.last_attempt.exception(), TimeoutError)
        assert str(exc_info.value.last_attempt.exception()) == "Connection timeout"

        assert mock_session.post.call_count == NUM_OF_API_RETRIES

    finally:
        api.list_families.cache_clear()
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_list_families_invalid_json(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families handles invalid JSON response with retries."""
    api.list_families.cache_clear()

    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_response.text.return_value = "Invalid JSON response"

    with pytest.raises(tenacity.RetryError):
        await api.list_families()

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_missing_login_session(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families handles missing login session."""
    api.list_families.cache_clear()

    api.loginsession = None
    mock_response.text.return_value = json.dumps(
        {"status": 0, "data": {"familyList": [{"familyid": "test1", "name": "Home 1"}]}}
    )
    mock_session.post.return_value = mock_response

    api.login = AsyncMock(return_value=True)

    result = await api.list_families()

    assert len(result) == 1
    assert result[0]["familyid"] == "test1"
    api.login.assert_called_once()

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_failure(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test failed family list retrieval."""
    api.list_families.cache_clear()

    mock_response.text = AsyncMock(
        return_value=json.dumps({"status": -1, "msg": "API Error"})
    )
    mock_session.post.return_value = mock_response
    api.loginsession = None
    api.userid = None
    api.login = AsyncMock(side_effect=AuxCloudAuthError("Login failed for test"))

    with pytest.raises(AuxCloudAuthError, match="Login failed for test"):
        await api.list_families()

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_get_devices(api: AuxCloudAPI) -> None:
    """Test getting all devices."""
    try:
        api._has_shared_devices.cache_clear()
        api.list_families.cache_clear()

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
        ambient_mode = {"envtemp": 220}

        temp_constant = 280
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

    finally:
        api._has_shared_devices.cache_clear()
        api.list_families.cache_clear()
        await asyncio.sleep(0)


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
    """Test device temperature query failure with retries."""
    mock_session.post.return_value.__aenter__.return_value = mock_response
    mock_response.text.return_value = json.dumps(
        {"event": {"payload": {"status": -1, "msg": "Temperature query failed"}}}
    )

    with pytest.raises(tenacity.RetryError):
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
        "cookie": (
            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5IjogImtleTEifX0="
        ),
    }
    with pytest.raises(ValueError, match="Params and Vals must have the same length"):
        await api._act_device_params(device, "set", ["temp"], [])


@pytest.mark.asyncio
async def test_refresh_success(api: AuxCloudAPI) -> None:
    """Test refresh function when family and device fetching succeed."""
    test_families = [{"familyid": "fam1"}]
    with (
        patch.object(api, "list_families", AsyncMock(return_value=test_families)),
        patch.object(api, "list_devices", AsyncMock(return_value=[])),
    ):
        await api.refresh()


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
async def test_cleanup_shared_resources(mock_session: MagicMock) -> None:
    """Test cleanup of shared resources."""
    connector = MagicMock()
    connector.close = AsyncMock()
    connector.closed = False

    mock_session.closed = False
    mock_session.close = AsyncMock()

    with patch.object(
        AuxCloudAPI, "get_shared_connector", AsyncMock(return_value=connector)
    ) and patch.object(
        AuxCloudAPI, "get_shared_session", AsyncMock(return_value=mock_session)
    ):
        AuxCloudAPI._shared_connector = connector
        AuxCloudAPI._shared_session = mock_session

        await AuxCloudAPI.cleanup_shared_resources()

        assert AuxCloudAPI._shared_connector is None
        assert AuxCloudAPI._shared_session is None
        mock_session.close.assert_called_once()
        connector.close.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_owned_session(mock_session: MagicMock) -> None:
    """Test cleanup with owned session (shared session)."""
    api = AuxCloudAPI("test@example.com", "password", region="eu")
    api.session = mock_session
    api._session_owner = True  # Using shared session

    mock_session.close = AsyncMock()

    await api.cleanup()

    assert api.session is None
    assert api._cleaned_up is True
    mock_session.close.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_external_session(mock_session: MagicMock) -> None:
    """Test cleanup with external session."""
    api = AuxCloudAPI("test@example.com", "password", session=mock_session, region="eu")
    assert api._session_owner is False

    mock_session.close = AsyncMock()

    await api.cleanup()

    assert api.session is None
    assert api._cleaned_up is True
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

    assert api.session is None
    assert api._cleaned_up is True
    mock_session.close.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_none_session() -> None:
    """Test cleanup with None session."""
    api = AuxCloudAPI("test@example.com", "password", region="eu")
    api.session = None
    await api.cleanup()


@pytest.mark.asyncio
async def test_list_families_cache_hit(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families cache hit scenario."""
    mock_response.text.return_value = json.dumps(
        {
            "status": 0,
            "data": {
                "familyList": [
                    {"familyid": "test1", "name": "Home 1"},
                ]
            },
        }
    )
    mock_session.post.return_value = mock_response

    api.list_families.cache_clear()

    result1 = await api.list_families()
    assert mock_session.post.call_count == 1
    assert len(result1) == 1
    assert result1[0]["familyid"] == "test1"

    result2 = await api.list_families()
    assert mock_session.post.call_count == 1
    assert result2 == result1

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_cache_clear(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families cache clearing behavior."""
    mock_response.text = AsyncMock(
        side_effect=[
            json.dumps(
                {
                    "status": 0,
                    "data": {
                        "familyList": [
                            {"familyid": "test1", "name": "Home 1"},
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "status": 0,
                    "data": {
                        "familyList": [
                            {"familyid": "test2", "name": "Home 2"},
                        ]
                    },
                }
            ),
        ]
    )
    mock_session.post.return_value = mock_response

    api.list_families.cache_clear()

    result1 = await api.list_families()
    assert mock_session.post.call_count == 1
    assert result1[0]["familyid"] == "test1"

    api.list_families.cache_clear()

    result2 = await api.list_families()
    assert mock_session.post.call_count == MIN_HOMES_COUNT
    assert result2[0]["familyid"] == "test2"
    assert result1 != result2

    api.list_families.cache_clear()


@pytest.mark.asyncio
async def test_list_families_cache_error(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test list_families cache behavior with errors with retry logic."""
    try:
        api.list_families.cache_clear()

        mock_session.post.side_effect = TimeoutError("Connection timeout")

        with pytest.raises(tenacity.RetryError) as exc_info:
            await api.list_families()

        assert isinstance(exc_info.value.last_attempt.exception(), TimeoutError)
        assert str(exc_info.value.last_attempt.exception()) == "Connection timeout"

        mock_session.post.side_effect = None
        mock_session.post.return_value = mock_response
        mock_response.text.return_value = json.dumps(
            {
                "status": 0,
                "data": {
                    "familyList": [
                        {"familyid": "test1", "name": "Home 1"},
                    ]
                },
            }
        )

        result = await api.list_families()
        assert len(result) == 1
        assert result[0]["familyid"] == "test1"

    finally:
        api.list_families.cache_clear()
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_shared_devices_caching_complete(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test complete shared devices caching behavior."""
    api._has_shared_devices.cache_clear()
    api.list_families.cache_clear()

    regular_response = {
        "status": 0,
        "data": {
            "endpoints": [
                {
                    "endpointId": "device1",
                    "devSession": "session1",
                    "productId": "prod1",
                    "mac": "00:11:22:33:44:55",
                    "devicetypeFlag": "1",
                    "cookie": (
                        "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5"
                        "IjogImtleTEifX0="
                    ),
                }
            ]
        },
    }

    shared_response = {
        "status": 0,
        "data": {
            "shareFromOther": [
                {
                    "devinfo": {
                        "endpointId": "shared1",
                        "devSession": "session2",
                        "productId": "prod1",
                        "mac": "00:11:22:33:44:55",
                        "devicetypeFlag": "1",
                        "cookie": (
                            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5"
                            "IjogImtleTEifX0="
                        ),
                    }
                }
            ]
        },
    }

    mock_response.text = AsyncMock(
        side_effect=[
            json.dumps(regular_response),
            json.dumps(shared_response),
            json.dumps(regular_response),
            json.dumps(shared_response),
        ]
    )
    mock_session.post.return_value = mock_response

    state_response = {
        "status": 0,
        "data": [{"state": "on"}],
    }
    params_response = {"temp": TEST_TEMPERATURE}
    ambient_response = {"envtemp": TEST_AMBIENT_TEMP}

    with (
        patch.object(api, "query_device_state", AsyncMock(return_value=state_response)),
        patch.object(
            api,
            "get_device_params",
            AsyncMock(side_effect=[params_response, ambient_response] * 4),
        ),
    ):
        regular_devices = await api.list_devices("family1", shared=False)
        assert len(regular_devices) == 1
        assert regular_devices[0]["endpointId"] == "device1"

        shared_devices = await api.list_devices("family1", shared=True)
        assert len(shared_devices) == 1
        assert shared_devices[0]["endpointId"] == "shared1"

        assert "family1" in api.data
        cached_devices = api.data["family1"]["devices"]
        assert len(cached_devices) == TEST_MIN_DEVICE_COUNT
        device_ids = {dev["endpointId"] for dev in cached_devices}
        assert "device1" in device_ids
        assert "shared1" in device_ids

        for device in cached_devices:
            assert "state" in device
            assert "params" in device
            assert device["params"]["temp"] == TEST_TEMPERATURE
            assert device["params"]["envtemp"] == TEST_AMBIENT_TEMP


@pytest.mark.asyncio
async def test_shared_devices_caching_no_shared_devices(
    api: AuxCloudAPI, mock_session: MagicMock, mock_response: MagicMock
) -> None:
    """Test shared devices caching behavior when no shared devices exist."""
    try:
        api._has_shared_devices.cache_clear()
        api.list_families.cache_clear()

        regular_response = {
            "status": 0,
            "data": {
                "endpoints": [
                    {
                        "endpointId": "device1",
                        "devSession": "session1",
                        "productId": "prod1",
                        "mac": "00:11:22:33:44:55",
                        "devicetypeFlag": "1",
                        "cookie": (
                            "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5"
                            "IjogImtleTEifX0="
                        ),
                    }
                ]
            },
        }

        original_list_devices = api.list_devices

        async def mock_list_devices(
            family_id: str, *, shared: bool = False
        ) -> list[dict[str, Any]]:
            if shared:
                return []
            return await original_list_devices(family_id, shared=False)

        with patch.object(api, "list_devices", side_effect=mock_list_devices):
            mock_response.text = AsyncMock(return_value=json.dumps(regular_response))
            mock_session.post.return_value = mock_response

            state_response = {
                "status": 0,
                "data": [{"state": "on"}],
            }
            params_response = {"temp": TEST_TEMPERATURE}
            ambient_response = {"envtemp": TEST_AMBIENT_TEMP}

            with (
                patch.object(
                    api, "query_device_state", AsyncMock(return_value=state_response)
                ),
                patch.object(
                    api,
                    "get_device_params",
                    AsyncMock(side_effect=[params_response, ambient_response] * 2),
                ),
            ):
                regular_devices = await api.list_devices("family1", shared=False)
                assert len(regular_devices) == 1
                assert regular_devices[0]["endpointId"] == "device1"

                shared_devices = await api.list_devices("family1", shared=True)
                assert len(shared_devices) == 0

                assert "family1" in api.data
                cached_devices = api.data["family1"]["devices"]
                assert len(cached_devices) == 1
                assert cached_devices[0]["endpointId"] == "device1"

                device = cached_devices[0]
                assert "state" in device
                assert "params" in device
                assert device["params"]["temp"] == TEST_TEMPERATURE
                assert device["params"]["envtemp"] == TEST_AMBIENT_TEMP

                has_shared = await api._has_shared_devices("family1")
                assert not has_shared

                second_check = await api._has_shared_devices("family1")
                assert not second_check

    finally:
        api._has_shared_devices.cache_clear()
        api.list_families.cache_clear()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_shared_devices_caching_no_shared_devices_call_count(
    api: AuxCloudAPI,
) -> None:
    """Test shared devices caching behavior when no shared devices exist."""
    try:
        api._has_shared_devices.cache_clear()
        api.list_families.cache_clear()

        list_devices_mock = AsyncMock()

        async def side_effect(
            family_id: str,  # noqa: ARG001
            *,
            shared: bool = False,
        ) -> list[dict[str, Any]]:
            if shared:
                return []
            device = {
                "endpointId": "device1",
                "devSession": "session1",
                "productId": "prod1",
                "mac": "00:11:22:33:44:55",
                "devicetypeFlag": "1",
                "cookie": (
                    "eyJkZXZpY2UiOiB7InRlcm1pbmFsaWQiOiAidGVybTEiLCAiYWVza2V5"
                    "IjogImtleTEifX0="
                ),
                "state": "on",
                "params": {"temp": TEST_TEMPERATURE, "envtemp": TEST_AMBIENT_TEMP},
            }
            return [device]

        list_devices_mock.side_effect = side_effect

        with patch.object(api, "list_devices", list_devices_mock):
            regular_devices = await api.list_devices("family1", shared=False)
            assert len(regular_devices) == 1
            assert regular_devices[0]["endpointId"] == "device1"

            shared_devices = await api.list_devices("family1", shared=True)
            assert len(shared_devices) == 0

            api.data = {"family1": {"devices": regular_devices}}

            device = regular_devices[0]
            assert "state" in device
            assert "params" in device
            assert device["params"]["temp"] == TEST_TEMPERATURE
            assert device["params"]["envtemp"] == TEST_AMBIENT_TEMP

            has_shared = await api._has_shared_devices("family1")
            assert not has_shared

            second_check = await api._has_shared_devices("family1")
            assert not second_check

            assert list_devices_mock.call_count == TEST_SHARED_DEVICES_CALL_COUNT

            calls = list_devices_mock.call_args_list

            first_call = calls[0]
            assert first_call.args[0] == "family1"
            assert first_call.kwargs.get("shared") is False

            second_call = calls[1]
            assert second_call.args[0] == "family1"
            assert second_call.kwargs.get("shared") is True

    finally:
        api._has_shared_devices.cache_clear()
        api.list_families.cache_clear()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_get_shared_connector() -> None:
    """Test get_shared_connector creates and reuses connector."""
    AuxCloudAPI._shared_connector = None

    connector1 = await AuxCloudAPI.get_shared_connector()
    assert isinstance(connector1, aiohttp.TCPConnector)
    assert connector1.limit == CONNECTION_POOL_LIMIT
    assert not connector1.closed

    connector2 = await AuxCloudAPI.get_shared_connector()
    assert connector2 is connector1

    await connector1.close()
    AuxCloudAPI._shared_connector = None


@pytest.mark.asyncio
async def test_get_shared_session() -> None:
    """Test get_shared_session creates and reuses session."""
    AuxCloudAPI._shared_session = None
    AuxCloudAPI._shared_connector = None

    session1 = await AuxCloudAPI.get_shared_session()
    assert isinstance(session1, aiohttp.ClientSession)
    assert not session1.closed

    session2 = await AuxCloudAPI.get_shared_session()
    assert session2 is session1

    await session1.close()
    if session1.connector:
        await session1.connector.close()
    AuxCloudAPI._shared_session = None
    AuxCloudAPI._shared_connector = None
