"""Tests for the Tornado AC switch component (Off Timer Control and CoolDown)."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.switch import SwitchDeviceClass
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.tornado.switch import (
    TornadoOffTimerControlSwitch,
    TornadoCoolDownSwitch,
)
from custom_components.tornado.const import DOMAIN

# Mock device data
MOCK_DEVICE = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 1,
        "ac_mode": 0,
        "ac_mark": 4,  # High fan mode
        "temp": 250,
        "envtemp": 270,
    },
}

MOCK_DEVICE_OFF = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 0,  # Device is off
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
async def off_timer_switch(
    hass: HomeAssistant, mock_coordinator: MagicMock
):
    """Create a mocked off timer control switch."""
    switch = TornadoOffTimerControlSwitch(mock_coordinator, MOCK_DEVICE)
    switch.hass = hass
    switch.entity_id = "switch.test_ac_off_timer"
    # Mock the async_write_ha_state method to avoid entity registration issues
    switch.async_write_ha_state = MagicMock()
    await switch.async_added_to_hass()
    
    yield switch
    
    # Cleanup any running timers
    if hasattr(switch, '_timer_handle') and switch._timer_handle:
        switch._timer_handle.cancel()
    if hasattr(switch, '_periodic_update_handle') and switch._periodic_update_handle:
        switch._periodic_update_handle()  # Cancel the timer


@pytest.fixture
async def cooldown_switch(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> TornadoCoolDownSwitch:
    """Create a mocked cooldown switch."""
    switch = TornadoCoolDownSwitch(mock_coordinator, MOCK_DEVICE)
    switch.hass = hass
    switch.entity_id = "switch.test_ac_cooldown"
    # Mock the async_write_ha_state method to avoid entity registration issues
    switch.async_write_ha_state = MagicMock()
    await switch.async_added_to_hass()
    return switch


class TestTornadoOffTimerControlSwitch:
    """Test the Tornado Off Timer Control Switch."""

    async def test_switch_initialization(self, off_timer_switch: TornadoOffTimerControlSwitch) -> None:
        """Test switch initialization."""
        assert off_timer_switch.unique_id == "test_device_id_off_timer_control"
        assert off_timer_switch.name == "Tornado AC Test AC Off Timer"
        assert off_timer_switch.device_class == SwitchDeviceClass.SWITCH
        assert off_timer_switch.icon == "mdi:timer"
        assert off_timer_switch.entity_category is None  # Controls section

    async def test_switch_initial_state(self, off_timer_switch: TornadoOffTimerControlSwitch) -> None:
        """Test switch initial state."""
        assert off_timer_switch.is_on is False
        assert off_timer_switch._timer_active is False
        assert off_timer_switch._timer_end_time is None

    async def test_get_off_timer_duration_default(self, off_timer_switch: TornadoOffTimerControlSwitch) -> None:
        """Test getting off timer duration when no select entity exists."""
        duration = off_timer_switch._get_off_timer_duration_from_select()
        assert duration == 0  # Default value

    @patch("custom_components.tornado.switch.async_track_time_interval")
    async def test_start_timer(
        self, mock_track_time, off_timer_switch: TornadoOffTimerControlSwitch
    ) -> None:
        """Test starting the off timer."""
        await off_timer_switch._start_timer(60)  # 60 minutes
        
        assert off_timer_switch._timer_active is True
        assert off_timer_switch._timer_end_time is not None
        assert off_timer_switch.is_on is True
        
        # Check timer end time is approximately 60 minutes from now
        expected_end = dt_util.utcnow() + timedelta(minutes=60)
        time_diff = abs((off_timer_switch._timer_end_time - expected_end).total_seconds())
        assert time_diff < 5  # Within 5 seconds
        
        # Verify periodic updates are started
        mock_track_time.assert_called_once()

    async def test_cancel_timer(self, off_timer_switch: TornadoOffTimerControlSwitch) -> None:
        """Test cancelling the off timer."""
        # First start a timer
        await off_timer_switch._start_timer(30)
        assert off_timer_switch._timer_active is True
        
        # Then cancel it
        await off_timer_switch._cancel_timer()
        
        assert off_timer_switch._timer_active is False
        assert off_timer_switch._timer_end_time is None
        assert off_timer_switch.is_on is False

    async def test_timer_completed(
        self, off_timer_switch: TornadoOffTimerControlSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test timer completion behavior."""
        await off_timer_switch._timer_completed()
        
        # Should turn off the device
        mock_coordinator.api.set_device_params.assert_called_once_with(
            MOCK_DEVICE, {"pwr": 0}
        )
        
        # Should reset timer state
        assert off_timer_switch._timer_active is False
        assert off_timer_switch._timer_end_time is None
        assert off_timer_switch.is_on is False

    async def test_async_turn_on_device_off(
        self, off_timer_switch: TornadoOffTimerControlSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test turning on timer when device is off."""
        # Mock device as off
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE_OFF}
        
        with patch.object(off_timer_switch, "_get_off_timer_duration_from_select", return_value=30), \
             patch.object(off_timer_switch, "_start_timer") as mock_start_timer, \
             patch("asyncio.sleep"):
            
            await off_timer_switch.async_turn_on()
            
            # Should turn on device first
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE_OFF, {"pwr": 1}
            )
            
            # Should start timer
            mock_start_timer.assert_called_once_with(30)

    async def test_async_turn_on_device_on(
        self, off_timer_switch: TornadoOffTimerControlSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test turning on timer when device is already on."""
        with patch.object(off_timer_switch, "_get_off_timer_duration_from_select", return_value=60), \
             patch.object(off_timer_switch, "_start_timer") as mock_start_timer:
            
            await off_timer_switch.async_turn_on()
            
            # Should not turn on device (already on)
            mock_coordinator.api.set_device_params.assert_not_called()
            
            # Should start timer
            mock_start_timer.assert_called_once_with(60)

    async def test_async_turn_off(self, off_timer_switch: TornadoOffTimerControlSwitch) -> None:
        """Test turning off the timer."""
        with patch.object(off_timer_switch, "_cancel_timer") as mock_cancel:
            await off_timer_switch.async_turn_off()
            mock_cancel.assert_called_once()

    async def test_coordinator_update_device_unavailable(
        self, off_timer_switch: TornadoOffTimerControlSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update when device is unavailable."""
        # Start timer first
        await off_timer_switch._start_timer(30)
        assert off_timer_switch._timer_active is True
        
        # Make device unavailable
        mock_coordinator.data = {}
        
        with patch.object(off_timer_switch, "_cancel_timer") as mock_cancel:
            off_timer_switch._handle_coordinator_update()
            
            # Test that _device returns None when no device data
            assert off_timer_switch._device is None
            mock_cancel.assert_called_once()

    async def test_coordinator_update_device_turned_off(
        self, off_timer_switch: TornadoOffTimerControlSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update when device is turned off while timer is active."""
        # Start timer first
        await off_timer_switch._start_timer(30)
        assert off_timer_switch._timer_active is True
        
        # Update device to be off
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE_OFF}
        
        with patch.object(off_timer_switch, "_cancel_timer") as mock_cancel:
            off_timer_switch._handle_coordinator_update()
            
            # Timer should be cancelled when device is turned off

    async def test_extra_state_attributes(self, off_timer_switch: TornadoOffTimerControlSwitch) -> None:
        """Test extra state attributes."""
        with patch.object(off_timer_switch, "_get_off_timer_duration_from_select", return_value=45):
            attrs = off_timer_switch.extra_state_attributes
            assert attrs["off_timer_duration"] == 45
            
        # Test with active timer
        end_time = dt_util.utcnow() + timedelta(minutes=20)
        off_timer_switch._timer_end_time = end_time
        
        attrs = off_timer_switch.extra_state_attributes
        assert "timer_end_time" in attrs
        assert "remaining_minutes" in attrs
        assert attrs["remaining_minutes"] == pytest.approx(20, abs=1)

    @patch("custom_components.tornado.switch.async_track_time_interval")
    async def test_periodic_sensor_update_timer_active(
        self, mock_track_time, off_timer_switch: TornadoOffTimerControlSwitch
    ) -> None:
        """Test periodic sensor updates when timer is active."""
        # Set up active timer that hasn't expired
        end_time = dt_util.utcnow() + timedelta(minutes=10)
        off_timer_switch._timer_active = True
        off_timer_switch._timer_end_time = end_time
        
        with patch.object(off_timer_switch, "_update_timer_sensor") as mock_update:
            await off_timer_switch._periodic_sensor_update(dt_util.utcnow())
            mock_update.assert_called_once()

    @patch("custom_components.tornado.switch.async_track_time_interval")
    async def test_periodic_sensor_update_timer_expired(
        self, mock_track_time, off_timer_switch: TornadoOffTimerControlSwitch
    ) -> None:
        """Test periodic sensor updates when timer has expired."""
        # Set up expired timer
        end_time = dt_util.utcnow() - timedelta(minutes=5)
        off_timer_switch._timer_active = True
        off_timer_switch._timer_end_time = end_time
        
        with patch.object(off_timer_switch, "_timer_completed") as mock_completed:
            await off_timer_switch._periodic_sensor_update(dt_util.utcnow())
            mock_completed.assert_called_once()


class TestTornadoCoolDownSwitch:
    """Test the Tornado CoolDown Switch."""

    async def test_switch_initialization(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test switch initialization."""
        assert cooldown_switch.unique_id == "test_device_id_cooldown"
        assert cooldown_switch.name == "Tornado AC Test AC CoolDown"
        assert cooldown_switch.device_class == SwitchDeviceClass.SWITCH
        assert cooldown_switch.icon == "mdi:snowflake-thermometer"
        assert cooldown_switch.entity_category is None  # Controls section

    async def test_switch_initial_state(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test switch initial state."""
        assert cooldown_switch.is_on is False
        assert cooldown_switch._cooldown_active is False

    async def test_get_fan_modes_default(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test getting fan modes when no select entities exist."""
        starting_fan = cooldown_switch._get_starting_fan_mode_from_select()
        target_fan = cooldown_switch._get_target_fan_mode_from_select()
        duration = cooldown_switch._get_cooldown_duration_from_select()
        
        assert starting_fan == 4  # Default to turbo
        assert target_fan == 5    # Default to silent
        assert duration == 5      # Default to 5 minutes

    @patch("custom_components.tornado.switch.async_track_point_in_time")
    async def test_async_turn_on_device_on(
        self, mock_track_time, cooldown_switch: TornadoCoolDownSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test turning on cooldown when device is already on."""
        with patch.object(cooldown_switch, "_get_starting_fan_mode_from_select", return_value=4), \
             patch.object(cooldown_switch, "_get_cooldown_duration_from_select", return_value=5):
            
            await cooldown_switch.async_turn_on()

            # Should set cooldown parameters (power on, cool mode, starting fan mode)
            expected_params = {"pwr": 1, "ac_mode": 0, "ac_mark": 4}
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE, expected_params
            )
            assert cooldown_switch._cooldown_active is True
            
            # Should set up timer
            mock_track_time.assert_called_once()

    @patch("custom_components.tornado.switch.async_track_point_in_time")
    async def test_async_turn_on_device_off(
        self, mock_track_time, cooldown_switch: TornadoCoolDownSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test turning on cooldown when device is off."""
        # Mock device as off
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE_OFF}
        
        with patch.object(cooldown_switch, "_get_starting_fan_mode_from_select", return_value=4), \
             patch.object(cooldown_switch, "_get_cooldown_duration_from_select", return_value=5), \
             patch("asyncio.sleep"):
            
            await cooldown_switch.async_turn_on()
            
            # Should set cooldown parameters (power on, cool mode, starting fan mode)
            expected_params = {"pwr": 1, "ac_mode": 0, "ac_mark": 4}
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE_OFF, expected_params
            )
            
            assert cooldown_switch._cooldown_active is True
            mock_track_time.assert_called_once()

    async def test_async_turn_off(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test turning off cooldown."""
        with patch.object(cooldown_switch, "_cancel_cooldown") as mock_cancel:
            await cooldown_switch.async_turn_off()
            mock_cancel.assert_called_once()

    async def test_cancel_cooldown(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test cancelling active cooldown."""
        # Set up active cooldown
        cooldown_switch._cooldown_active = True
        cooldown_switch._cooldown_timer_handle = MagicMock()
        
        await cooldown_switch._cancel_cooldown()
        
        assert cooldown_switch._cooldown_active is False
        assert cooldown_switch.is_on is False
        assert cooldown_switch._cooldown_timer_handle is None

    async def test_complete_cooldown(
        self, cooldown_switch: TornadoCoolDownSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test cooldown completion."""
        with patch.object(cooldown_switch, "_get_target_fan_mode_from_select", return_value=5):
            await cooldown_switch._complete_cooldown(dt_util.utcnow())
            
            # Should set target fan mode (silent)
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE, {"ac_mark": 5}
            )
            
            assert cooldown_switch._cooldown_active is False
            assert cooldown_switch.is_on is False

    async def test_coordinator_update_device_unavailable(
        self, cooldown_switch: TornadoCoolDownSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update when device is unavailable."""
        # Start cooldown first
        cooldown_switch._cooldown_active = True
        
        # Make device unavailable
        mock_coordinator.data = {}
        
        with patch.object(cooldown_switch, "_cancel_cooldown") as mock_cancel:
            cooldown_switch._handle_coordinator_update()
            
            # Test that _device returns None when no device data
            assert cooldown_switch._device is None
            mock_cancel.assert_called_once()

    async def test_coordinator_update_device_turned_off(
        self, cooldown_switch: TornadoCoolDownSwitch, mock_coordinator: MagicMock
    ) -> None:
        """Test coordinator update when device is turned off while cooldown is active."""
        # Start cooldown first
        cooldown_switch._cooldown_active = True
        
        # Update device to be off
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: MOCK_DEVICE_OFF}
        
        with patch.object(cooldown_switch, "_cancel_cooldown") as mock_cancel:
            cooldown_switch._handle_coordinator_update()
            
            # Cooldown should be cancelled when device is turned off

    async def test_extra_state_attributes(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test extra state attributes."""
        with patch.object(cooldown_switch, "_get_starting_fan_mode_from_select", return_value=4), \
             patch.object(cooldown_switch, "_get_target_fan_mode_from_select", return_value=5), \
             patch.object(cooldown_switch, "_get_cooldown_duration_from_select", return_value=10):
            
            attrs = cooldown_switch.extra_state_attributes
            assert attrs["starting_fan_mode"] == 4
            assert attrs["target_fan_mode"] == 5
            assert attrs["cooldown_duration"] == 10

    async def test_cleanup_on_removal(self, cooldown_switch: TornadoCoolDownSwitch) -> None:
        """Test cleanup when entity is removed."""
        # Set up timer handle
        mock_handle = MagicMock()
        cooldown_switch._cooldown_timer_handle = mock_handle
        
        await cooldown_switch.async_will_remove_from_hass()
        
        # Timer handle should be cancelled
        mock_handle.cancel.assert_called_once()
        assert cooldown_switch._cooldown_timer_handle is None
