"""Tests for the Tornado AC sensor component (Off Timer Status)."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.tornado.const import DOMAIN
from custom_components.tornado.sensor import TornadoOffTimerSensor

# Mock device data
MOCK_DEVICE = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 1,
        "ac_mode": 0,
        "ac_mark": 1,
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
async def off_timer_sensor(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> TornadoOffTimerSensor:
    """Create a mocked off timer sensor."""
    sensor = TornadoOffTimerSensor(mock_coordinator, MOCK_DEVICE)
    sensor.hass = hass
    sensor.entity_id = "sensor.test_ac_off_timer"
    # Mock the async_write_ha_state method to avoid entity registration issues
    sensor.async_write_ha_state = MagicMock()
    await sensor.async_added_to_hass()
    return sensor


class TestTornadoOffTimerSensor:
    """Test the Tornado Off Timer Sensor."""

    async def test_sensor_initialization(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test sensor initialization."""
        assert off_timer_sensor.unique_id == "test_device_id_off_timer"
        assert off_timer_sensor.name == "Tornado AC Test AC Off Timer Status"
        assert off_timer_sensor.device_class == SensorDeviceClass.DURATION
        assert off_timer_sensor.state_class == SensorStateClass.MEASUREMENT
        assert off_timer_sensor.native_unit_of_measurement == UnitOfTime.MINUTES
        assert off_timer_sensor.icon == "mdi:timer-outline"

    async def test_sensor_initial_state(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test sensor initial state."""
        assert off_timer_sensor.native_value == 0
        assert off_timer_sensor.extra_state_attributes["timer_active"] is False
        assert off_timer_sensor.extra_state_attributes["display_text"] == "No off timer"

    async def test_update_timer_state_active(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test updating timer state when timer is active."""
        # Set up timer end time 30 minutes from now
        end_time = dt_util.utcnow() + timedelta(minutes=30)
        
        await off_timer_sensor.async_update_timer_state_from_control(True, end_time)
        
        assert off_timer_sensor.native_value == 30
        assert off_timer_sensor.extra_state_attributes["timer_active"] is True
        assert "30m" in off_timer_sensor.extra_state_attributes["display_text"]
        assert off_timer_sensor.extra_state_attributes["timer_end_time"] == end_time.isoformat()
        assert off_timer_sensor.extra_state_attributes["remaining_seconds"] == pytest.approx(1800, abs=5)

    async def test_update_timer_state_inactive(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test updating timer state when timer is inactive."""
        await off_timer_sensor.async_update_timer_state_from_control(False, None)
        
        assert off_timer_sensor.native_value == 0
        assert off_timer_sensor.extra_state_attributes["timer_active"] is False
        assert off_timer_sensor.extra_state_attributes["display_text"] == "No off timer"
        assert off_timer_sensor.extra_state_attributes["timer_end_time"] is None
        assert off_timer_sensor.extra_state_attributes["remaining_seconds"] == 0

    async def test_update_timer_state_expired(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test updating timer state when timer has expired."""
        # Set end time in the past
        end_time = dt_util.utcnow() - timedelta(minutes=5)
        
        await off_timer_sensor.async_update_timer_state_from_control(True, end_time)
        
        assert off_timer_sensor.native_value == 0
        assert off_timer_sensor.extra_state_attributes["timer_active"] is False
        assert off_timer_sensor.extra_state_attributes["display_text"] == "Off timer finished"
        assert off_timer_sensor.extra_state_attributes["remaining_seconds"] == 0

    async def test_format_time_remaining(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test time formatting."""
        # Test various time formats
        assert off_timer_sensor._format_time_remaining(0) == "Off timer finished"
        assert off_timer_sensor._format_time_remaining(-10) == "Off timer finished"
        assert off_timer_sensor._format_time_remaining(30) == "1m"  # Less than 1 minute shows as 1m
        assert off_timer_sensor._format_time_remaining(60) == "1m"
        assert off_timer_sensor._format_time_remaining(90) == "2m"
        assert off_timer_sensor._format_time_remaining(3600) == "1h 0m"
        assert off_timer_sensor._format_time_remaining(3660) == "1h 1m"
        assert off_timer_sensor._format_time_remaining(7200) == "2h 0m"

    async def test_coordinator_update_available(
        self, off_timer_sensor: TornadoOffTimerSensor, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update when device is available."""
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
        off_timer_sensor._handle_coordinator_update()
        
        assert off_timer_sensor._attr_available is True

    async def test_coordinator_update_unavailable(
        self, off_timer_sensor: TornadoOffTimerSensor, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update when device is unavailable."""
        mock_coordinator.data = {}
        off_timer_sensor._handle_coordinator_update()
        
        assert off_timer_sensor._attr_available is False

    async def test_timer_progression(self, off_timer_sensor: TornadoOffTimerSensor) -> None:
        """Test timer countdown progression."""
        # Start with 5 minutes
        end_time = dt_util.utcnow() + timedelta(minutes=5)
        await off_timer_sensor.async_update_timer_state_from_control(True, end_time)
        
        initial_remaining = off_timer_sensor.extra_state_attributes["remaining_seconds"]
        assert initial_remaining == pytest.approx(300, abs=5)
        
        # Wait a short time and check progression
        await asyncio.sleep(0.1)
        
        # Update again (simulating periodic update)
        await off_timer_sensor.async_update_timer_state_from_control(True, end_time)
        
        # Should still be active but with slightly less time
        assert off_timer_sensor.extra_state_attributes["timer_active"] is True
        assert off_timer_sensor.native_value == 5  # Still 5 minutes (rounded up)

    async def test_device_property(
        self, off_timer_sensor: TornadoOffTimerSensor, mock_coordinator: MagicMock
    ) -> None:
        """Test device property returns correct device data."""
        # Test with device available
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE}
        device = off_timer_sensor._device
        assert device == MOCK_DEVICE
        
        # Test with no coordinator data
        mock_coordinator.data = None
        device = off_timer_sensor._device
        assert device is None
        
        # Test with device not in coordinator data
        mock_coordinator.data = {}
        device = off_timer_sensor._device
        assert device is None
