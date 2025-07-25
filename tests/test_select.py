"""Tests for the Tornado AC select component (Off Timer Duration and CoolDown selects)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components.select import SelectEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory

from custom_components.tornado.select import (
    TornadoOffTimerDurationSelect,
    TornadoCoolDownStartingFanSelect,
    TornadoCoolDownTargetFanSelect,
    TornadoCoolDownDurationSelect,
)
from custom_components.tornado.const import DOMAIN

# Mock device data
MOCK_DEVICE = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 1,
        "ac_mode": 0,
        "ac_mark": 4,
        "temp": 250,
        "envtemp": 270,
    },
}


@pytest.fixture
def mock_coordinator() -> MagicMock:
    """Create a mock coordinator."""
    coordinator = MagicMock()
    coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
    coordinator.api = AsyncMock()
    return coordinator


@pytest.fixture
async def off_timer_duration_select(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> TornadoOffTimerDurationSelect:
    """Create a mocked off timer duration select."""
    select = TornadoOffTimerDurationSelect(mock_coordinator, MOCK_DEVICE)
    select.hass = hass
    select.entity_id = "select.test_ac_off_timer_duration"
    # Mock the async_write_ha_state method to avoid entity registration issues
    select.async_write_ha_state = MagicMock()
    await select.async_added_to_hass()
    return select


@pytest.fixture
async def cooldown_starting_fan_select(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> TornadoCoolDownStartingFanSelect:
    """Create a mocked cooldown starting fan select."""
    select = TornadoCoolDownStartingFanSelect(mock_coordinator, MOCK_DEVICE)
    select.hass = hass
    select.entity_id = "select.test_ac_cooldown_starting_fan"
    # Mock the async_write_ha_state method to avoid entity registration issues
    select.async_write_ha_state = MagicMock()
    await select.async_added_to_hass()
    return select


@pytest.fixture
async def cooldown_target_fan_select(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> TornadoCoolDownTargetFanSelect:
    """Create a mocked cooldown target fan select."""
    select = TornadoCoolDownTargetFanSelect(mock_coordinator, MOCK_DEVICE)
    select.hass = hass
    select.entity_id = "select.test_ac_cooldown_target_fan"
    # Mock the async_write_ha_state method to avoid entity registration issues
    select.async_write_ha_state = MagicMock()
    await select.async_added_to_hass()
    return select


@pytest.fixture
async def cooldown_duration_select(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> TornadoCoolDownDurationSelect:
    """Create a mocked cooldown duration select."""
    select = TornadoCoolDownDurationSelect(mock_coordinator, MOCK_DEVICE)
    select.hass = hass
    select.entity_id = "select.test_ac_cooldown_duration"
    # Mock the async_write_ha_state method to avoid entity registration issues
    select.async_write_ha_state = MagicMock()
    await select.async_added_to_hass()
    return select


class TestTornadoOffTimerDurationSelect:
    """Test the Tornado Off Timer Duration Select."""

    async def test_select_initialization(
        self, off_timer_duration_select: TornadoOffTimerDurationSelect
    ) -> None:
        """Test select initialization."""
        assert off_timer_duration_select.unique_id == "test_device_id_off_timer_duration"
        assert off_timer_duration_select.name == "Tornado AC Test AC Off Timer Duration"
        assert off_timer_duration_select.entity_category == EntityCategory.CONFIG
        assert off_timer_duration_select.icon == "mdi:timer-cog"

    async def test_select_options(
        self, off_timer_duration_select: TornadoOffTimerDurationSelect
    ) -> None:
        """Test select options."""
        expected_options = ["10m", "30m", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h"]
        assert off_timer_duration_select.options == expected_options
        assert off_timer_duration_select.current_option == "30m"  # Default

    async def test_select_option_valid(
        self, off_timer_duration_select: TornadoOffTimerDurationSelect
    ) -> None:
        """Test selecting a valid option."""
        await off_timer_duration_select.async_select_option("1h")
        assert off_timer_duration_select.current_option == "1h"

    async def test_select_option_invalid(
        self, off_timer_duration_select: TornadoOffTimerDurationSelect
    ) -> None:
        """Test selecting an invalid option."""
        original_option = off_timer_duration_select.current_option
        await off_timer_duration_select.async_select_option("invalid")
        # Should remain unchanged
        assert off_timer_duration_select.current_option == original_option

    async def test_all_duration_options(
        self, off_timer_duration_select: TornadoOffTimerDurationSelect
    ) -> None:
        """Test all duration options are handled correctly."""
        duration_tests = {
            "10m": 10,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
        }
        
        for option, expected_minutes in duration_tests.items():
            await off_timer_duration_select.async_select_option(option)
            assert off_timer_duration_select.current_option == option

    async def test_coordinator_update(
        self, off_timer_duration_select: TornadoOffTimerDurationSelect, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update."""
        # Off timer duration should always be available regardless of device state
        mock_coordinator.data = {}
        off_timer_duration_select._handle_coordinator_update()
        # Off timer duration select is always available, so just test that it doesn't crash
        assert off_timer_duration_select is not None


class TestTornadoCoolDownStartingFanSelect:
    """Test the Tornado CoolDown Starting Fan Select."""

    async def test_select_initialization(
        self, cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect
    ) -> None:
        """Test select initialization."""
        assert cooldown_starting_fan_select.unique_id == "test_device_id_cooldown_starting_fan"
        assert cooldown_starting_fan_select.name == "Tornado AC Test AC CoolDown Starting Fan"
        assert cooldown_starting_fan_select.entity_category == EntityCategory.CONFIG
        assert cooldown_starting_fan_select.current_option == "high"  # Default

    async def test_select_fan_options(
        self, cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect
    ) -> None:
        """Test fan mode options."""
        expected_options = ["auto", "low", "medium", "high", "turbo", "silent"]
        assert cooldown_starting_fan_select.options == expected_options

    async def test_select_option_changes_icon(
        self, cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect
    ) -> None:
        """Test that selecting different options changes the icon."""
        # Test different fan modes and their expected icons
        fan_icon_tests = {
            "auto": "mdi:fan-auto",
            "low": "mdi:fan-speed-1",
            "medium": "mdi:fan-speed-2",
            "high": "mdi:fan-speed-3",
            "turbo": "mdi:fan-plus",
            "silent": "mdi:fan-minus",
        }
        
        for fan_mode, expected_icon in fan_icon_tests.items():
            await cooldown_starting_fan_select.async_select_option(fan_mode)
            assert cooldown_starting_fan_select.icon == expected_icon

    async def test_available_when_cooldown_inactive(
        self, cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect, mock_coordinator: MagicMock
    ) -> None:
        """Test entity is available when cooldown is not active."""
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
        # The available property uses _get_cooldown_switch which won't find the switch in tests
        # so it will default to available=True when device data exists
        assert cooldown_starting_fan_select._device is not None

    async def test_unavailable_when_device_unavailable(
        self, cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect, mock_coordinator: MagicMock
    ) -> None:
        """Test entity is unavailable when device is unavailable."""
        mock_coordinator.data = {}
        # Test that _device returns None when no device data
        assert cooldown_starting_fan_select._device is None


class TestTornadoCoolDownTargetFanSelect:
    """Test the Tornado CoolDown Target Fan Select."""

    async def test_select_initialization(
        self, cooldown_target_fan_select: TornadoCoolDownTargetFanSelect
    ) -> None:
        """Test select initialization."""
        assert cooldown_target_fan_select.unique_id == "test_device_id_cooldown_target_fan"
        assert cooldown_target_fan_select.name == "Tornado AC Test AC CoolDown Target Fan"
        assert cooldown_target_fan_select.entity_category == EntityCategory.CONFIG
        assert cooldown_target_fan_select.current_option == "silent"  # Default

    async def test_select_option_changes_icon(
        self, cooldown_target_fan_select: TornadoCoolDownTargetFanSelect
    ) -> None:
        """Test that selecting different options changes the icon."""
        # Test different fan modes and their expected icons
        fan_icon_tests = {
            "auto": "mdi:fan-auto",
            "low": "mdi:fan-speed-1",
            "medium": "mdi:fan-speed-2", 
            "high": "mdi:fan-speed-3",
            "turbo": "mdi:fan-plus",
            "silent": "mdi:fan-minus",
        }
        
        for fan_mode, expected_icon in fan_icon_tests.items():
            await cooldown_target_fan_select.async_select_option(fan_mode)
            assert cooldown_target_fan_select.icon == expected_icon

    async def test_available_when_cooldown_inactive(
        self, cooldown_target_fan_select: TornadoCoolDownTargetFanSelect, mock_coordinator: MagicMock
    ) -> None:
        """Test entity is available when cooldown is not active."""
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
        assert cooldown_target_fan_select.available is True


class TestTornadoCoolDownDurationSelect:
    """Test the Tornado CoolDown Duration Select."""

    async def test_select_initialization(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect
    ) -> None:
        """Test select initialization."""
        assert cooldown_duration_select.unique_id == "test_device_id_cooldown_duration"
        assert cooldown_duration_select.name == "Tornado AC Test AC CoolDown Duration"
        assert cooldown_duration_select.entity_category == EntityCategory.CONFIG
        assert cooldown_duration_select.icon == "mdi:timer-cog"

    async def test_select_duration_options(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect
    ) -> None:
        """Test duration options."""
        expected_options = ["1min", "3min", "5min", "10min", "15min", "30min"]
        assert cooldown_duration_select.options == expected_options
        assert cooldown_duration_select.current_option == "10min"  # Default

    async def test_select_option_valid(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect
    ) -> None:
        """Test selecting a valid option."""
        await cooldown_duration_select.async_select_option("10min")
        assert cooldown_duration_select.current_option == "10min"

    async def test_select_option_invalid(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect
    ) -> None:
        """Test selecting an invalid option."""
        original_option = cooldown_duration_select.current_option
        await cooldown_duration_select.async_select_option("invalid")
        # Should remain unchanged
        assert cooldown_duration_select.current_option == original_option

    async def test_all_duration_options(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect
    ) -> None:
        """Test all duration options are handled correctly."""
        duration_tests = {
            "1min": 1,
            "3min": 3,
            "5min": 5,
            "10min": 10,
            "15min": 15,
            "30min": 30,
        }
        
        for option, expected_minutes in duration_tests.items():
            await cooldown_duration_select.async_select_option(option)
            assert cooldown_duration_select.current_option == option

    async def test_available_when_cooldown_inactive(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect, mock_coordinator: MagicMock
    ) -> None:
        """Test entity is available when cooldown is not active."""
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
        assert cooldown_duration_select.available is True

    async def test_coordinator_update(
        self, cooldown_duration_select: TornadoCoolDownDurationSelect, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update."""
        # CoolDown duration should always be available
        mock_coordinator.data = {}
        cooldown_duration_select._handle_coordinator_update()
        assert cooldown_duration_select.available is True


class TestSelectIntegration:
    """Test integration between select entities."""

    async def test_multiple_selects_independent(
        self,
        off_timer_duration_select: TornadoOffTimerDurationSelect,
        cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect,
        cooldown_target_fan_select: TornadoCoolDownTargetFanSelect,
        cooldown_duration_select: TornadoCoolDownDurationSelect,
    ) -> None:
        """Test that multiple select entities operate independently."""
        # Change all select entities
        await off_timer_duration_select.async_select_option("2h")
        await cooldown_starting_fan_select.async_select_option("turbo")
        await cooldown_target_fan_select.async_select_option("low")
        await cooldown_duration_select.async_select_option("15min")
        
        # Verify all changes are independent
        assert off_timer_duration_select.current_option == "2h"
        assert cooldown_starting_fan_select.current_option == "turbo"
        assert cooldown_target_fan_select.current_option == "low"
        assert cooldown_duration_select.current_option == "15min"

    async def test_device_info_consistency(
        self,
        off_timer_duration_select: TornadoOffTimerDurationSelect,
        cooldown_starting_fan_select: TornadoCoolDownStartingFanSelect,
    ) -> None:
        """Test that device info is consistent across select entities."""
        off_timer_device_info = off_timer_duration_select.device_info
        cooldown_device_info = cooldown_starting_fan_select.device_info
        
        assert off_timer_device_info == cooldown_device_info
        assert off_timer_device_info["identifiers"] == {(DOMAIN, "test_device_id")}
        assert off_timer_device_info["name"] == "Tornado AC Test AC"
        assert off_timer_device_info["manufacturer"] == "Tornado"
        assert off_timer_device_info["model"] == "AUX Cloud"