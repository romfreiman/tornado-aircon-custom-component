"""Tests for the Tornado AC climate component."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components.climate import (
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_TEMPERATURE,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant

from custom_components.tornado.climate import (
    DOMAIN,
    AuxCloudDataUpdateCoordinator,
    TornadoClimateEntity,
)

MOCK_DEVICE = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 1,
        "ac_mode": 2,  # COOL mode
        "ac_mark": 1,  # Low fan
        "temp": 250,  # 25.0°C
        "envtemp": 270,  # 27.0°C
        "ac_vdir": 1,
        "ac_hdir": 0,
    },
}


@pytest.fixture
def mock_api():
    """Create a mock AuxCloud API."""
    api = MagicMock()
    api.get_devices = AsyncMock(return_value=[MOCK_DEVICE])
    api.set_device_params = AsyncMock()
    return api


@pytest.fixture
async def coordinator(hass, mock_api):
    """Create a mocked coordinator."""
    coordinator = AuxCloudDataUpdateCoordinator(hass, mock_api)
    coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
    return coordinator


@pytest.fixture
async def entity(hass, coordinator):
    """Create a mocked climate entity."""
    entity = TornadoClimateEntity(hass, coordinator, MOCK_DEVICE)
    entity.entity_id = "climate.test_ac"
    entity.hass = hass
    return entity


async def test_climate_entity_initialization(hass: HomeAssistant, coordinator, entity) -> None:
    """Test climate entity initialization."""
    assert entity.unique_id == "test_device_id_climate"
    assert entity.name == "Tornado AC Test AC"
    assert entity.supported_features == (
        ClimateEntityFeature.TARGET_TEMPERATURE
        | ClimateEntityFeature.FAN_MODE
        | ClimateEntityFeature.SWING_MODE
        | ClimateEntityFeature.TURN_ON
        | ClimateEntityFeature.TURN_OFF
    )
    assert entity.temperature_unit == UnitOfTemperature.CELSIUS
    assert entity.min_temp == 16
    assert entity.max_temp == 32


async def test_climate_update(hass: HomeAssistant, coordinator, entity) -> None:
    """Test climate entity state updates."""
    entity._handle_coordinator_update()

    assert entity.hvac_mode == HVACMode.COOL
    assert entity.hvac_action == HVACAction.COOLING
    assert entity.fan_mode == "low"
    assert entity.target_temperature == 25.0
    assert entity.current_temperature == 27.0
    assert entity.swing_mode == "vertical"
    assert entity.available is True


async def test_set_temperature(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test setting temperature."""
    await entity.async_set_temperature(**{ATTR_TEMPERATURE: 24.0})
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"temp": 240})


async def test_set_hvac_mode(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test setting HVAC mode."""
    await entity.async_set_hvac_mode(HVACMode.HEAT)
    mock_api.set_device_params.assert_called_once_with(
        MOCK_DEVICE, {"pwr": 1, "ac_mode": 1}
    )


async def test_turn_off(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test turning device off."""
    await entity.async_turn_off()
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"pwr": 0})


async def test_coordinator_update_error(hass: HomeAssistant, mock_api) -> None:
    """Test coordinator update with error."""
    mock_api.get_devices.side_effect = Exception("API Error")
    coordinator = AuxCloudDataUpdateCoordinator(hass, mock_api)

    with pytest.raises(Exception):
        await coordinator._async_update_data()


@pytest.fixture(autouse=True)
async def setup_comp(hass) -> None:
    """Set up things to be run when tests are started."""
    await hass.async_block_till_done()


@pytest.fixture(autouse=True)
async def cleanup_timers(hass):
    """Clean up timers after test."""
    yield
    for timer in hass.loop._scheduled:
        timer.cancel()


async def test_set_fan_mode(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test setting fan mode."""
    await entity.async_set_fan_mode("high")
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"ac_mark": 3})


async def test_set_swing_mode(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test setting swing mode."""
    # Test vertical mode
    await entity.async_set_swing_mode("vertical")
    mock_api.set_device_params.assert_called_once_with(
        MOCK_DEVICE, {"ac_vdir": 1, "ac_hdir": 0}
    )

    mock_api.set_device_params.reset_mock()

    # Test horizontal mode
    await entity.async_set_swing_mode("horizontal")
    mock_api.set_device_params.assert_called_once_with(
        MOCK_DEVICE, {"ac_vdir": 0, "ac_hdir": 1}
    )

    mock_api.set_device_params.reset_mock()

    # Test both mode
    await entity.async_set_swing_mode("both")
    mock_api.set_device_params.assert_called_once_with(
        MOCK_DEVICE, {"ac_vdir": 1, "ac_hdir": 1}
    )


async def test_turn_on(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test turning device on."""
    await entity.async_turn_on()
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"pwr": 1})


async def test_device_properties(hass: HomeAssistant, coordinator, entity) -> None:
    """Test device properties."""
    assert entity.icon == "mdi:air-conditioner"
    assert entity.available is True
    assert entity.device_info == {
        "identifiers": {(DOMAIN, "test_device_id")},
        "name": "Tornado AC Test AC",
        "manufacturer": "Tornado",
        "model": "AUX Cloud",
    }


async def test_hvac_modes(hass: HomeAssistant, coordinator, entity) -> None:
    """Test HVAC modes."""
    assert HVACMode.OFF in entity.hvac_modes
    assert HVACMode.COOL in entity.hvac_modes
    assert HVACMode.HEAT in entity.hvac_modes
    assert HVACMode.AUTO in entity.hvac_modes
    assert HVACMode.DRY in entity.hvac_modes
    assert HVACMode.FAN_ONLY in entity.hvac_modes


async def test_fan_modes(hass: HomeAssistant, coordinator, entity) -> None:
    """Test fan modes."""
    assert "auto" in entity.fan_modes
    assert "low" in entity.fan_modes
    assert "medium" in entity.fan_modes
    assert "high" in entity.fan_modes


async def test_swing_modes(hass: HomeAssistant, coordinator, entity) -> None:
    """Test swing modes."""
    assert "off" in entity.swing_modes
    assert "vertical" in entity.swing_modes
    assert "horizontal" in entity.swing_modes
    assert "both" in entity.swing_modes


async def test_temperature_limits(hass: HomeAssistant, coordinator, entity) -> None:
    """Test temperature limits."""
    assert entity.min_temp == 16
    assert entity.max_temp == 32
    assert entity.temperature_unit == UnitOfTemperature.CELSIUS


async def test_coordinator_update_with_invalid_data(
    hass: HomeAssistant, coordinator, entity
) -> None:
    """Test coordinator update with invalid data."""
    # Test with missing device
    coordinator.data = {}
    entity._handle_coordinator_update()
    assert entity.available is False

    # Test with invalid params
    coordinator.data = {MOCK_DEVICE["endpointId"]: {"params": {}}}
    entity._handle_coordinator_update()
    assert entity.available is True
    assert entity.hvac_mode == HVACMode.OFF


async def test_set_invalid_temperature(
    hass: HomeAssistant, coordinator, entity, mock_api
) -> None:
    """Test setting invalid temperature."""
    # Test with no temperature provided
    await entity.async_set_temperature()
    mock_api.set_device_params.assert_not_called()


async def test_api_error_handling(hass: HomeAssistant, coordinator, entity, mock_api) -> None:
    """Test API error handling."""
    # Simulate API error
    mock_api.set_device_params.side_effect = Exception("API Error")

    # Test temperature setting with error
    await entity.async_set_temperature(**{ATTR_TEMPERATURE: 24.0})
    mock_api.set_device_params.assert_called_once()

    # Test turn off with error
    mock_api.set_device_params.reset_mock()
    await entity.async_turn_off()
    mock_api.set_device_params.assert_called_once()


async def test_hvac_action_mapping(hass: HomeAssistant, coordinator, entity) -> None:
    """Test HVAC action mapping for different modes."""
    # Test cooling action
    coordinator.data = {
        MOCK_DEVICE["endpointId"]: {
            **MOCK_DEVICE,
            "params": {**MOCK_DEVICE["params"], "pwr": 1, "ac_mode": 2},  # COOL mode
        }
    }
    entity._handle_coordinator_update()
    assert entity.hvac_action == HVACAction.COOLING

    # Test heating action
    coordinator.data = {
        MOCK_DEVICE["endpointId"]: {
            **MOCK_DEVICE,
            "params": {**MOCK_DEVICE["params"], "pwr": 1, "ac_mode": 1},  # HEAT mode
        }
    }
    entity._handle_coordinator_update()
    assert entity.hvac_action == HVACAction.HEATING

    # Test drying action
    coordinator.data = {
        MOCK_DEVICE["endpointId"]: {
            **MOCK_DEVICE,
            "params": {**MOCK_DEVICE["params"], "pwr": 1, "ac_mode": 4},  # DRY mode
        }
    }
    entity._handle_coordinator_update()
    assert entity.hvac_action == HVACAction.DRYING

    # Test fan action
    coordinator.data = {
        MOCK_DEVICE["endpointId"]: {
            **MOCK_DEVICE,
            "params": {**MOCK_DEVICE["params"], "pwr": 1, "ac_mode": 3},  # FAN mode
        }
    }
    entity._handle_coordinator_update()
    assert entity.hvac_action == HVACAction.FAN
