"""Tests for AuxCloud API client."""

import json
from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.tornado.aux_cloud import (
    AuxCloudAPI,
    AuxCloudApiError,
    AuxCloudAuthError,
)


@pytest.fixture
def mock_response() -> MagicMock:
    """Mock aiohttp response."""
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock()
    response.text = AsyncMock()
    return response


@pytest.fixture
async def api() -> AuxCloudAPI:
    """Create API instance with mocked login."""
    api = AuxCloudAPI("test@example.com", "password", "eu")
    # Set required attributes that would normally be set during login
    api.loginsession = "test_session"
    api.userid = "test_user"
    return api


@pytest.mark.asyncio
async def test_login_success(mock_response: MagicMock) -> None:
    """Test successful login."""
    api = AuxCloudAPI("test@example.com", "password", "eu")
    mock_response.text.return_value = json.dumps(
        {"status": 0, "loginsession": "session123", "userid": "user123"}
    )

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        result = await api.login()

        assert result is True
        assert api.loginsession == "session123"
        assert api.userid == "user123"


@pytest.mark.asyncio
async def test_login_error_handling() -> None:
    """Test failed login using a fake client session that simulates a failed login."""
    api = AuxCloudAPI("test@example.com", "wrongpassword", "eu")

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
    api: AuxCloudAPI, mock_response: MagicMock
) -> None:
    """Test successful family list retrieval."""
    mock_response.text.return_value = json.dumps(
        {
            "status": 0,
            "data": {
                "familyList": [
                    {
                        "familyid": "abc123def456ghi789jkl012mno345p",
                        "userid": "",
                        "name": "Test Home",
                        "icon": "",
                        "description": "",
                        "countryCode": "USA",
                        "familylimit": 0,
                        "provinceCode": "1",
                        "cityCode": "10",
                        "orgname": "",
                        "createTime": "2024-01-01 00:00:00",
                        "version": "2024-01-01 00:00:00",
                        "createUser": "abc123def456ghi789jkl012mno345p",
                        "grouptype": "",
                        "master": "abc123def456ghi789jkl012mno345p",
                        "extend": (
                            '{"weather":{"city":"10","country":"USA","province":"1"}}'
                        ),
                        "zoneInfo": "",
                        "spaceId": "",
                        "companyId": "",
                    }
                ]
            },
        }
    )

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
        result = await api.list_families()

        assert len(result) == 1
        assert result[0]["familyid"] == "abc123def456ghi789jkl012mno345p"
        assert result[0]["name"] == "Test Home"
        assert result[0]["countryCode"] == "USA"
        assert result[0]["createTime"] == "2024-01-01 00:00:00"
        assert result[0]["version"] == "2024-01-01 00:00:00"


@pytest.mark.asyncio
async def test_list_families_failure(
    api: AuxCloudAPI, mock_response: MagicMock
) -> None:
    """Test failed family list retrieval."""
    # Set up response mock
    mock_response.text = AsyncMock(
        return_value=json.dumps({"status": -1, "msg": "Error"})
    )
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__.return_value = None

    # Set up session mock
    session_mock = MagicMock()
    session_mock.post.return_value = mock_response
    session_mock.__aenter__.return_value = session_mock
    session_mock.__aexit__.return_value = None

    with (
        patch("aiohttp.ClientSession", return_value=session_mock),
        pytest.raises(AuxCloudApiError, match="Failed to get families list:"),
    ):
        await api.list_families()


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
async def test_query_device_state(api: AuxCloudAPI, mock_response: MagicMock) -> None:
    """Test querying device state."""
    mock_response.text.return_value = json.dumps(
        {"event": {"payload": {"status": 0, "data": [{"state": "on"}]}}}
    )

    with patch("aiohttp.ClientSession.post", return_value=mock_response):
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
    api = AuxCloudAPI("test@example.com", "password", "eu")
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
    """Test _get_directive_header to ensure correct header generation."""
    api = AuxCloudAPI("test@example.com", "password", "eu")
    extra_kwarg = {"custom": "value"}
    header = api._get_directive_header(
        "TestNamespace", "TestName", "prefix", **extra_kwarg
    )
    # Check that expected keys are present
    assert header["namespace"] == "TestNamespace"
    assert header["name"] == "TestName"
    assert header["interfaceVersion"] == "2"
    assert header["senderId"] == "sdk"
    # Check messageId starts with given prefix and a hyphen separator
    assert header["messageId"].startswith("prefix-")
    # Check extra kwargs are merged into header
    assert header["custom"] == "value"


@pytest.mark.asyncio
async def test_query_device_temperature_success() -> None:
    """Test successful device temperature query."""
    api = AuxCloudAPI("test@example.com", "password", "eu")
    # Set a test userid since it's used in the query data
    api.userid = "test_user"
    fake_payload = {"event": {"payload": {"status": 0, "temperature": 23.5}}}

    query_temp_constant = 23.5  # Define constant for query temperature

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
            return json.dumps(fake_payload)

    class FakeSession:
        async def __aenter__(self) -> "FakeSession":
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

    with patch("aiohttp.ClientSession", return_value=FakeSession()):
        result = await api.query_device_temperature("dev1", "sess1")
        assert "temperature" in result
        assert result["temperature"] == query_temp_constant


@pytest.mark.asyncio
async def test_query_device_temperature_failure() -> None:
    """Test failed device temperature query."""
    api = AuxCloudAPI("test@example.com", "password", "eu")
    # Set a test userid since it's used in the query data
    api.userid = "test_user"

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
            # Simulate a failure response
            return json.dumps(
                {"event": {"payload": {"status": 1, "msg": "Temperature query failed"}}}
            )

    class FakeSession:
        async def __aenter__(self) -> "FakeSession":
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

    with (
        patch("aiohttp.ClientSession", return_value=FakeSession()),
        pytest.raises(AuxCloudApiError, match="Failed to query device temperature"),
    ):
        await api.query_device_temperature("dev1", "sess1")


@pytest.mark.asyncio
async def test_build_temperature_query_data() -> None:
    """Test _build_temperature_query_data method structure."""
    api = AuxCloudAPI("test@example.com", "password", "eu")
    # Set a test userid since it's used as message_id_prefix
    api.userid = "test_user"
    device_id = "dev123"
    dev_session = "sess123"
    data = api._build_temperature_query_data(device_id, dev_session)
    assert "directive" in data
    directive = data["directive"]
    # Check that header is properly built
    assert "header" in directive
    header = directive["header"]
    assert header["namespace"] == "DNA.TemperatureSensor"
    assert header["name"] == "ReportState"
    # Verify that endpoint info was added correctly
    assert "endpoint" in directive
    endpoint = directive["endpoint"]
    assert endpoint["endpointId"] == device_id


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
