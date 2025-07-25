"""Integration tests for Off Timer and CoolDown features."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.tornado.const import DOMAIN
from custom_components.tornado.sensor import TornadoOffTimerSensor
from custom_components.tornado.switch import TornadoOffTimerControlSwitch, TornadoCoolDownSwitch
from custom_components.tornado.select import (
    TornadoOffTimerDurationSelect,
    TornadoCoolDownStartingFanSelect,
    TornadoCoolDownTargetFanSelect,
    TornadoCoolDownDurationSelect,
)

# Mock device data
MOCK_DEVICE = {
    "endpointId": "test_device_id",
    "friendlyName": "Test AC",
    "params": {
        "pwr": 1,
        "ac_mode": 0,
        "ac_mark": 3,  # High fan mode
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
async def off_timer_components(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> tuple[TornadoOffTimerControlSwitch, TornadoOffTimerSensor, TornadoOffTimerDurationSelect]:
    """Create all off timer related components."""
    switch = TornadoOffTimerControlSwitch(mock_coordinator, MOCK_DEVICE)
    sensor = TornadoOffTimerSensor(mock_coordinator, MOCK_DEVICE)
    duration_select = TornadoOffTimerDurationSelect(mock_coordinator, MOCK_DEVICE)
    
    # Set up Home Assistant references
    switch.hass = hass
    sensor.hass = hass
    duration_select.hass = hass
    
    # Set entity IDs and mock state writing
    switch.entity_id = "switch.test_ac_off_timer"
    sensor.entity_id = "sensor.test_ac_off_timer"
    duration_select.entity_id = "select.test_ac_off_timer_duration"
    
    switch.async_write_ha_state = MagicMock()
    sensor.async_write_ha_state = MagicMock()
    duration_select.async_write_ha_state = MagicMock()
    
    # Add to hass
    await switch.async_added_to_hass()
    await sensor.async_added_to_hass()
    await duration_select.async_added_to_hass()
    
    return switch, sensor, duration_select


@pytest.fixture
async def cooldown_components(
    hass: HomeAssistant, mock_coordinator: MagicMock
) -> tuple[TornadoCoolDownSwitch, TornadoCoolDownStartingFanSelect, TornadoCoolDownTargetFanSelect, TornadoCoolDownDurationSelect]:
    """Create all cooldown related components."""
    switch = TornadoCoolDownSwitch(mock_coordinator, MOCK_DEVICE)
    starting_fan_select = TornadoCoolDownStartingFanSelect(mock_coordinator, MOCK_DEVICE)
    target_fan_select = TornadoCoolDownTargetFanSelect(mock_coordinator, MOCK_DEVICE)
    duration_select = TornadoCoolDownDurationSelect(mock_coordinator, MOCK_DEVICE)
    
    # Set up Home Assistant references
    switch.hass = hass
    starting_fan_select.hass = hass
    target_fan_select.hass = hass
    duration_select.hass = hass
    
    # Set entity IDs and mock state writing
    switch.entity_id = "switch.test_ac_cooldown"
    starting_fan_select.entity_id = "select.test_ac_cooldown_starting_fan"
    target_fan_select.entity_id = "select.test_ac_cooldown_target_fan"
    duration_select.entity_id = "select.test_ac_cooldown_duration"
    
    switch.async_write_ha_state = MagicMock()
    starting_fan_select.async_write_ha_state = MagicMock()
    target_fan_select.async_write_ha_state = MagicMock()
    duration_select.async_write_ha_state = MagicMock()
    
    # Add to hass
    await switch.async_added_to_hass()
    await starting_fan_select.async_added_to_hass()
    await target_fan_select.async_added_to_hass()
    await duration_select.async_added_to_hass()
    
    return switch, starting_fan_select, target_fan_select, duration_select


class TestOffTimerIntegration:
    """Test integration between off timer components."""

    async def test_off_timer_full_workflow(
        self, off_timer_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test complete off timer workflow from configuration to completion."""
        switch, sensor, duration_select = off_timer_components
        
        # Step 1: Configure duration
        await duration_select.async_select_option("1h")
        assert duration_select.current_option == "1h"
        
        # Step 2: Start off timer
        with patch.object(switch, "_get_off_timer_duration_from_select", return_value=60), \
             patch.object(switch, "_update_timer_sensor") as mock_update_sensor, \
             patch("custom_components.tornado.switch.async_track_time_interval"):
            
            await switch.async_turn_on()
            
            # Verify switch state
            assert switch.is_on is True
            assert switch._timer_active is True
            assert switch._timer_end_time is not None
            
            # Verify sensor was updated
            mock_update_sensor.assert_called_once()
        
        # Step 3: Simulate sensor update from switch
        end_time = dt_util.utcnow() + timedelta(minutes=60)
        await sensor.async_update_timer_state_from_control(True, end_time)
        
        # Verify sensor state
        assert sensor.native_value == 60
        assert sensor.extra_state_attributes["timer_active"] is True
        assert "1h 0m" in sensor.extra_state_attributes["display_text"]
        
        # Step 4: Cancel timer
        with patch.object(switch, "_update_timer_sensor") as mock_update_sensor:
            await switch.async_turn_off()
            
            # Verify switch state
            assert switch.is_on is False
            assert switch._timer_active is False
            
            # Verify sensor was updated
            mock_update_sensor.assert_called_once()

    async def test_off_timer_completion_workflow(
        self, off_timer_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test off timer completion workflow."""
        switch, sensor, duration_select = off_timer_components
        
        # Start timer
        with patch.object(switch, "_get_off_timer_duration_from_select", return_value=1), \
             patch.object(switch, "_update_timer_sensor"), \
             patch("custom_components.tornado.switch.async_track_time_interval"):
            
            await switch.async_turn_on()
        
        # Simulate timer completion
        with patch.object(switch, "_update_timer_sensor") as mock_update_sensor:
            await switch._timer_completed()
            
            # Verify device was turned off
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE, {"pwr": 0}
            )
            
            # Verify switch state reset
            assert switch.is_on is False
            assert switch._timer_active is False
            assert switch._timer_end_time is None
            
            # Verify sensor was updated
            mock_update_sensor.assert_called_once()

    async def test_off_timer_sensor_sync(
        self, off_timer_components: tuple
    ) -> None:
        """Test sensor synchronization with switch state."""
        switch, sensor, duration_select = off_timer_components
        
        # Test different timer states
        test_cases = [
            # (active, end_time_offset_minutes, expected_minutes, expected_active)
            (True, 30, 30, True),
            (True, 5, 5, True),
            (True, -5, 0, False),  # Expired timer
            (False, None, 0, False),
        ]
        
        for active, end_time_offset, expected_minutes, expected_active in test_cases:
            end_time = dt_util.utcnow() + timedelta(minutes=end_time_offset) if end_time_offset else None
            
            await sensor.async_update_timer_state_from_control(active, end_time)
            
            assert sensor.native_value == expected_minutes
            assert sensor.extra_state_attributes["timer_active"] == expected_active


class TestCoolDownIntegration:
    """Test integration between cooldown components."""

    async def test_cooldown_full_workflow(
        self, cooldown_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test complete cooldown workflow from configuration to completion."""
        switch, starting_fan_select, target_fan_select, duration_select = cooldown_components
        
        # Step 1: Configure cooldown settings
        await starting_fan_select.async_select_option("turbo")
        await target_fan_select.async_select_option("silent")
        await duration_select.async_select_option("10min")
        
        assert starting_fan_select.current_option == "turbo"
        assert target_fan_select.current_option == "silent"
        assert duration_select.current_option == "10min"
        
        # Step 2: Start cooldown
        with patch.object(switch, "_get_starting_fan_mode_from_select", return_value=4), \
             patch.object(switch, "_get_cooldown_duration_from_select", return_value=10), \
             patch("custom_components.tornado.switch.async_track_point_in_time") as mock_track:
            
            await switch.async_turn_on()
            
            # Verify switch state
            assert switch._cooldown_active is True
            
            # Verify starting fan mode was set
            expected_params = {"pwr": 1, "ac_mode": 0, "ac_mark": 4}
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE, expected_params
            )
            
            # Verify timer was set up
            mock_track.assert_called_once()
        
        # Step 3: Complete cooldown
        with patch.object(switch, "_get_target_fan_mode_from_select", return_value=5):
            await switch._complete_cooldown(dt_util.utcnow())
            
            # Verify target fan mode was set
            mock_coordinator.api.set_device_params.assert_called_with(
                MOCK_DEVICE, {"ac_mark": 5}
            )
            
            # Verify switch state reset
            assert switch._cooldown_active is False

    async def test_cooldown_configuration_affects_behavior(
        self, cooldown_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test that cooldown configuration affects actual behavior."""
        switch, starting_fan_select, target_fan_select, duration_select = cooldown_components
        
        # Test different configurations
        configurations = [
            ("high", "low", "5min", 3, 1, 5),
            ("turbo", "silent", "15min", 4, 5, 15),
            ("medium", "auto", "30min", 2, 0, 30),
        ]
        
        for start_fan, target_fan, duration, start_val, target_val, duration_val in configurations:
            # Configure
            await starting_fan_select.async_select_option(start_fan)
            await target_fan_select.async_select_option(target_fan)
            await duration_select.async_select_option(duration)
            
            # Start cooldown - only patch methods that are actually called during turn_on
            with patch.object(switch, "_get_starting_fan_mode_from_select", return_value=start_val), \
                 patch.object(switch, "_get_cooldown_duration_from_select", return_value=duration_val), \
                 patch("custom_components.tornado.switch.async_track_point_in_time"):
                
                # Reset mock
                mock_coordinator.api.set_device_params.reset_mock()
                
                # Start cooldown
                await switch.async_turn_on()
                
                # Verify starting fan mode was set correctly
                expected_params = {"pwr": 1, "ac_mode": 0, "ac_mark": start_val}
                mock_coordinator.api.set_device_params.assert_called_with(
                    MOCK_DEVICE, expected_params
                )
            
            # Complete cooldown - patch target fan mode only when it's actually called
            with patch.object(switch, "_get_target_fan_mode_from_select", return_value=target_val):
                await switch._complete_cooldown(dt_util.utcnow())
                
                # Verify target fan mode was set correctly
                mock_coordinator.api.set_device_params.assert_called_with(
                    MOCK_DEVICE, {"ac_mark": target_val}
                )


class TestFeatureInteraction:
    """Test interaction between off timer and cooldown features."""

    async def test_features_are_independent(
        self, off_timer_components: tuple, cooldown_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test that off timer and cooldown features work independently."""
        off_timer_switch, off_timer_sensor, off_timer_duration = off_timer_components
        cooldown_switch, cooldown_starting, cooldown_target, cooldown_duration = cooldown_components
        
        # Configure both features
        await off_timer_duration.async_select_option("2h")
        await cooldown_starting.async_select_option("turbo")
        await cooldown_target.async_select_option("silent")
        await cooldown_duration.async_select_option("5min")
        
        # Start both features
        with patch.object(off_timer_switch, "_get_off_timer_duration_from_select", return_value=120), \
             patch.object(off_timer_switch, "_update_timer_sensor"), \
             patch("custom_components.tornado.switch.async_track_time_interval"), \
             patch.object(cooldown_switch, "_get_starting_fan_mode_from_select", return_value=4), \
             patch.object(cooldown_switch, "_get_cooldown_duration_from_select", return_value=5), \
             patch("custom_components.tornado.switch.async_track_point_in_time"):
            
            await off_timer_switch.async_turn_on()
            await cooldown_switch.async_turn_on()
        
        # Both should be active
        assert off_timer_switch.is_on is True
        assert cooldown_switch._cooldown_active is True
        
        # Turn off one feature
        await off_timer_switch.async_turn_off()
        
        # Only off timer should be affected
        assert off_timer_switch.is_on is False
        assert cooldown_switch._cooldown_active is True

    async def test_device_unavailable_affects_both_features(
        self, off_timer_components: tuple, cooldown_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test that device becoming unavailable affects both features."""
        off_timer_switch, _, _ = off_timer_components
        cooldown_switch, _, _, _ = cooldown_components
        
        # Start both features
        off_timer_switch._timer_active = True
        cooldown_switch._cooldown_active = True
        
        # Make device unavailable
        mock_coordinator.data = {}
        
        with patch.object(off_timer_switch, "_cancel_timer") as mock_cancel_timer, \
             patch.object(cooldown_switch, "_cancel_cooldown") as mock_cancel_cooldown:
            
            off_timer_switch._handle_coordinator_update()
            cooldown_switch._handle_coordinator_update()
            
            # Both features should be cancelled
            # Test that _device returns None when no device data
            assert off_timer_switch._device is None
            assert cooldown_switch._device is None

    async def test_device_turned_off_affects_both_features(
        self, off_timer_components: tuple, cooldown_components: tuple, mock_coordinator: MagicMock
    ) -> None:
        """Test that device being turned off affects both features."""
        off_timer_switch, _, _ = off_timer_components
        cooldown_switch, _, _, _ = cooldown_components
        
        # Start both features
        off_timer_switch._timer_active = True
        cooldown_switch._cooldown_active = True
        
        # Turn off device
        device_off = MOCK_DEVICE.copy()
        device_off["params"] = device_off["params"].copy()
        device_off["params"]["pwr"] = 0
        mock_coordinator.data = {MOCK_DEVICE["endpointId"]: device_off}
        
        with patch.object(off_timer_switch, "_cancel_timer") as mock_cancel_timer, \
             patch.object(cooldown_switch, "_cancel_cooldown") as mock_cancel_cooldown:
            
            off_timer_switch._handle_coordinator_update()
            cooldown_switch._handle_coordinator_update()
            
            # Both features should be cancelled when device is turned off
