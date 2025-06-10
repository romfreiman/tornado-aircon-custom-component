"""Tests for the Tornado AC climate component."""

import contextlib
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

# Constants for temperature values
MIN_TEMP = 16
MAX_TEMP = 32
CURRENT_TEMP = 27.0
TARGET_TEMP = 25.0

MOCK_DEVICE = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 1,
        "ac_mode": 0,  # COOL mode (updated mapping)
        "ac_mark": 1,  # Low fan
        "temp": 250,  # 25.0°C
        "envtemp": 270,  # 27.0°C
        "ac_vdir": 1,
        "ac_hdir": 0,
    },
}


@pytest.fixture
def mock_api() -> MagicMock:
    """Create a mock AuxCloud API."""
    api = MagicMock()
    api.get_devices = AsyncMock(return_value=[MOCK_DEVICE])
    api.set_device_params = AsyncMock()
    return api


@pytest.fixture
async def coordinator(
    hass: HomeAssistant, mock_api: MagicMock
) -> AuxCloudDataUpdateCoordinator:
    """Create a mocked coordinator."""
    coordinator = AuxCloudDataUpdateCoordinator(hass, mock_api)
    coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
    await coordinator.async_refresh()  # Ensure initial data is set
    return coordinator


@pytest.fixture
async def entity(
    hass: HomeAssistant, coordinator: AuxCloudDataUpdateCoordinator
) -> TornadoClimateEntity:
    """Create a mocked climate entity."""
    entity = TornadoClimateEntity(hass, coordinator, MOCK_DEVICE)
    entity.entity_id = "climate.test_ac"
    entity.hass = hass
    # Ensure the entity registers with the coordinator
    await entity.async_added_to_hass()
    return entity


async def test_climate_entity_initialization(entity: TornadoClimateEntity) -> None:
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
    assert entity.min_temp == MIN_TEMP
    assert entity.max_temp == MAX_TEMP


async def test_climate_update(entity: TornadoClimateEntity) -> None:
    """Test climate entity state updates."""
    await entity.async_update()

    assert entity.hvac_mode == HVACMode.COOL
    assert entity.hvac_action == HVACAction.COOLING
    assert entity.fan_mode == "low"
    assert entity.target_temperature == TARGET_TEMP
    assert entity.current_temperature == CURRENT_TEMP
    assert entity.swing_mode == "vertical"
    assert entity.available is True


async def test_set_temperature(
    entity: TornadoClimateEntity, mock_api: MagicMock
) -> None:
    """Test setting temperature."""
    await entity.async_set_temperature(**{ATTR_TEMPERATURE: 24.0})
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"temp": 240})


async def test_set_hvac_mode(entity: TornadoClimateEntity, mock_api: MagicMock) -> None:
    """Test setting HVAC mode."""
    await entity.async_set_hvac_mode(HVACMode.HEAT)
    mock_api.set_device_params.assert_called_once_with(
        MOCK_DEVICE, {"pwr": 1, "ac_mode": 1}
    )


async def test_turn_off(entity: TornadoClimateEntity, mock_api: MagicMock) -> None:
    """Test turning device off."""
    await entity.async_turn_off()
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"pwr": 0})


async def test_coordinator_update_error(
    hass: HomeAssistant, mock_api: MagicMock
) -> None:
    """Test coordinator update with error."""
    mock_api.get_devices.side_effect = Exception("API Error")
    coordinator = AuxCloudDataUpdateCoordinator(hass, mock_api)

    # Test that the coordinator's update method raises the exception
    with pytest.raises(Exception, match="API Error"):
        # ruff: noqa: SLF001
        await coordinator._async_update_data()


@pytest.fixture(autouse=True)
async def setup_comp(hass: HomeAssistant) -> None:
    """Set up things to be run when tests are started."""
    await hass.async_block_till_done()


@pytest.fixture(autouse=True)
async def cleanup_timers(hass: HomeAssistant) -> None:
    """Clean up timers after test."""
    yield
    import asyncio

    # Cancel all pending tasks
    current_task = asyncio.current_task()
    for task in asyncio.all_tasks():
        if task is not current_task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # Cancel all pending timers
    # ruff: noqa: SLF001
    for handle in hass.loop._scheduled:
        handle.cancel()

    # Allow the loop to process the cancellations
    await asyncio.sleep(0)


async def test_set_fan_mode(entity: TornadoClimateEntity, mock_api: MagicMock) -> None:
    """Test setting fan mode."""
    await entity.async_set_fan_mode("high")
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"ac_mark": 3})


async def test_set_turbo_fan_mode(entity: TornadoClimateEntity, mock_api: MagicMock) -> None:
    """Test setting turbo fan mode."""
    await entity.async_set_fan_mode("turbo")
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"ac_mark": 4})


async def test_set_silent_fan_mode(entity: TornadoClimateEntity, mock_api: MagicMock) -> None:
    """Test setting silent fan mode."""
    await entity.async_set_fan_mode("silent")
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"ac_mark": 5})


async def test_set_swing_mode(
    entity: TornadoClimateEntity, mock_api: MagicMock
) -> None:
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


async def test_turn_on(entity: TornadoClimateEntity, mock_api: MagicMock) -> None:
    """Test turning device on."""
    await entity.async_turn_on()
    mock_api.set_device_params.assert_called_once_with(MOCK_DEVICE, {"pwr": 1})


async def test_device_properties(entity: TornadoClimateEntity) -> None:
    """Test device properties."""
    assert entity.icon == "mdi:air-conditioner"
    assert entity.available is True
    assert entity.device_info == {
        "identifiers": {(DOMAIN, "test_device_id")},
        "name": "Tornado AC Test AC",
        "manufacturer": "Tornado",
        "model": "AUX Cloud",
    }


async def test_hvac_modes(entity: TornadoClimateEntity) -> None:
    """Test HVAC modes."""
    assert HVACMode.OFF in entity.hvac_modes
    assert HVACMode.COOL in entity.hvac_modes
    assert HVACMode.HEAT in entity.hvac_modes
    assert HVACMode.AUTO in entity.hvac_modes
    assert HVACMode.DRY in entity.hvac_modes
    assert HVACMode.FAN_ONLY in entity.hvac_modes


async def test_fan_modes(entity: TornadoClimateEntity) -> None:
    """Test fan modes."""
    assert "auto" in entity.fan_modes
    assert "low" in entity.fan_modes
    assert "medium" in entity.fan_modes
    assert "high" in entity.fan_modes
    assert "turbo" in entity.fan_modes
    assert "silent" in entity.fan_modes


async def test_temperature_limits(entity: TornadoClimateEntity) -> None:
    """Test temperature limits."""
    assert entity.min_temp == MIN_TEMP
    assert entity.max_temp == MAX_TEMP
    assert entity.temperature_unit == UnitOfTemperature.CELSIUS


async def test_coordinator_update_with_invalid_data(
    coordinator: AuxCloudDataUpdateCoordinator, entity: TornadoClimateEntity
) -> None:
    """Test coordinator update with invalid data."""
    # Test with missing device
    coordinator.data = {}
    await entity.async_update()
    assert entity.available  # Entity remains available even without data
    assert entity.hvac_mode == HVACMode.COOL  # Retains last known mode

    # Test with invalid params
    coordinator.data = {MOCK_DEVICE["endpointId"]: {"params": {}}}
    await entity.async_update()
    assert entity.available
    assert entity.hvac_mode == HVACMode.COOL  # Retains last known mode


async def test_set_invalid_temperature(
    entity: TornadoClimateEntity, mock_api: MagicMock
) -> None:
    """Test setting invalid temperature."""
    # Test with no temperature provided
    await entity.async_set_temperature()
    mock_api.set_device_params.assert_not_called()


async def test_api_error_handling(
    entity: TornadoClimateEntity, mock_api: MagicMock
) -> None:
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


async def test_hvac_action_mapping(
    coordinator: AuxCloudDataUpdateCoordinator,
    entity: TornadoClimateEntity,
    mock_api: MagicMock,
) -> None:
    """Test HVAC action mapping for different modes."""

    async def mock_and_refresh(new_device_params: dict) -> None:
        """Update the mock device with new parameters and refresh coordinator."""
        # Update the MOCK_DEVICE with new parameters
        updated_device = {
            **MOCK_DEVICE,
            "params": {
                **MOCK_DEVICE["params"],
                **new_device_params,
            },
        }
        # Mock the API to return the updated device
        mock_api.get_devices.return_value = [updated_device]
        # Trigger a coordinator refresh which will fetch the updated data
        await coordinator.async_refresh()

    # Test cooling action (ac_mode:0 -> COOLING)
    await mock_and_refresh(
        {
            "pwr": 1,
            "ac_mode": 0,  # COOL mode (updated mapping)
            "temp": 250,  # 25.0°C target temperature
            "envtemp": 270,  # 27.0°C current temperature (needs cooling)
        }
    )
    assert entity.hvac_action == HVACAction.COOLING

    # Test heating action (ac_mode:1 -> HEATING)
    await mock_and_refresh(
        {
            "pwr": 1,
            "ac_mode": 1,  # HEAT mode
            "temp": 280,  # 28.0°C target temperature
            "envtemp": 260,  # 26.0°C current temperature (needs heating)
        }
    )
    assert entity.hvac_action == HVACAction.HEATING

    # Test drying action (ac_mode:2 -> DRYING)
    await mock_and_refresh(
        {
            "pwr": 1,
            "ac_mode": 2,  # DRY mode (updated mapping)
        }
    )
    assert entity.hvac_action == HVACAction.DRYING

    # Test fan action (ac_mode:3 -> FAN)
    await mock_and_refresh(
        {
            "pwr": 1,
            "ac_mode": 3,  # FAN_ONLY mode
        }
    )
    assert entity.hvac_action == HVACAction.FAN

    # Test auto action (ac_mode:4 -> AUTO/IDLE)
    await mock_and_refresh(
        {
            "pwr": 1,
            "ac_mode": 4,  # AUTO mode (updated mapping)
        }
    )
    assert entity.hvac_action == HVACAction.IDLE

    # Test off action (pwr:0 -> OFF)
    await mock_and_refresh(
        {
            "pwr": 0,
        }
    )
    assert entity.hvac_action == HVACAction.OFF
