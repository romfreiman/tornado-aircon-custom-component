import pathlib
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.tornado.config_flow import (
    TornadoConfigFlow,
    TornadoOptionsFlow,
)
from custom_components.tornado.const import (
    CONF_EMAIL,
    CONF_PASSWORD,
    CONF_REGION,
)


# A minimal mock config entry for testing options flow
class MockConfigEntry:
    """Mock config entry for testing."""

    def __init__(self, data) -> None:
        self.entry_id = "test"
        self.data = data
        self.options = {}
        self.title = data.get(CONF_EMAIL, "unknown")

    @property
    def config_entry_id(self):
        return self.entry_id


@pytest.fixture
async def hass(tmp_path: pathlib.Path) -> HomeAssistant:
    """Fixture to provide a test instance of Home Assistant."""
    hass = HomeAssistant(config_dir=str(tmp_path))
    # Start Home Assistant
    await hass.async_start()
    yield hass
    # Stop Home Assistant
    await hass.async_stop()


# Fixture for valid user input for the config flow
@pytest.fixture
def valid_user_input():
    return {
        CONF_EMAIL: "test@example.com",
        CONF_PASSWORD: "secret",
        CONF_REGION: "eu",
    }


# Test the async_step_user when login is successful
@pytest.mark.asyncio
async def test_async_step_user_success(hass, valid_user_input) -> None:
    """Test successful user flow."""
    flow = TornadoConfigFlow()
    flow.hass = hass

    # First call without user input to get the form
    result = await flow.async_step_user()
    assert result["type"] == "form"
    assert result["step_id"] == "user"

    # Mock the unique ID configuration and API
    with (
        patch("custom_components.tornado.config_flow.AuxCloudAPI") as mock_aux,
        patch(
            "homeassistant.config_entries.ConfigFlow.async_set_unique_id"
        ) as mock_set_unique_id,
    ):
        instance = AsyncMock()
        instance.login.return_value = None  # simulate successful login
        mock_aux.return_value = instance
        mock_set_unique_id.return_value = None

        # Now call with user input
        result = await flow.async_step_user(user_input=valid_user_input)

    assert result["type"] == "create_entry"
    assert result["title"] == valid_user_input[CONF_EMAIL]
    assert result["data"] == valid_user_input


# Test the async_step_user when login fails
@pytest.mark.asyncio
async def test_async_step_user_cannot_connect(hass, valid_user_input) -> None:
    flow = TornadoConfigFlow()
    flow.hass = hass

    with patch("custom_components.tornado.config_flow.AuxCloudAPI") as mock_aux:
        instance = AsyncMock()
        instance.login.side_effect = Exception("login failed")
        mock_aux.return_value = instance

        result = await flow.async_step_user(user_input=valid_user_input)

    assert result["type"] == "form"
    assert "base" in result["errors"]
    assert result["errors"]["base"] == "cannot_connect"


# Test the options flow when login is successful
@pytest.mark.asyncio
async def test_options_flow_success(hass, valid_user_input) -> None:
    """Test successful options flow."""
    config_entry = MockConfigEntry(valid_user_input)
    options_flow = TornadoOptionsFlow(config_entry)
    options_flow.hass = hass

    # New options with the same region (or could change to any valid option)
    user_input = {CONF_REGION: "eu"}

    with patch("custom_components.tornado.config_flow.AuxCloudAPI") as mock_aux:
        instance = AsyncMock()
        instance.login.return_value = None  # simulate successful login
        mock_aux.return_value = instance

        result = await options_flow.async_step_init(user_input=user_input)

    assert result["type"] == "create_entry"
    assert result["data"] == user_input


# Test the options flow when login fails
@pytest.mark.asyncio
async def test_options_flow_cannot_connect(hass, valid_user_input) -> None:
    """Test failed options flow."""
    config_entry = MockConfigEntry(valid_user_input)
    options_flow = TornadoOptionsFlow(config_entry)
    options_flow.hass = hass

    user_input = {CONF_REGION: "eu"}  # valid region, but login will fail

    with patch("custom_components.tornado.config_flow.AuxCloudAPI") as mock_aux:
        instance = AsyncMock()
        instance.login.side_effect = Exception("login failed")
        mock_aux.return_value = instance

        result = await options_flow.async_step_init(user_input=user_input)

    assert result["type"] == "form"
    assert "base" in result["errors"]
    assert result["errors"]["base"] == "cannot_connect"


# Test that the static method returns an instance of the options flow.
def test_async_get_options_flow(valid_user_input) -> None:
    config_entry = MockConfigEntry(valid_user_input)
    options_flow = TornadoConfigFlow.async_get_options_flow(config_entry)
    from custom_components.tornado.config_flow import TornadoOptionsFlow

    assert isinstance(options_flow, TornadoOptionsFlow)
