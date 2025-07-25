"""Platform for Tornado AC switch integration."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.switch import (
    SwitchEntity,
    SwitchEntityDescription,
    SwitchDeviceClass,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.helpers.event import async_track_point_in_time, async_track_time_interval
from homeassistant.util import dt as dt_util

from .const import DOMAIN, COOLDOWN_DURATION_MAP, OFF_TIMER_DURATION_MAP
from .climate import FAN_MODE_MAP_REVERSE
from .utils import find_entity_by_unique_id

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from . import AuxCloudDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Tornado switch platform."""
    # Get the coordinator from the climate platform
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = entry_data.get("coordinator")
    
    if not coordinator:
        _LOGGER.error("No coordinator found for switch setup")
        return

    try:
        devices = await coordinator.api.get_devices()
        entities = []

        for device in devices:
            try:
                # Add Sleep Mode switch
                entities.append(
                    TornadoSleepModeSwitch(
                        coordinator,
                        device,
                    )
                )
                
                # Add Eco Mode switch
                entities.append(
                    TornadoEcoModeSwitch(
                        coordinator,
                        device,
                    )
                )
                
                # Add Off Timer Control switch (Controls section)
                entities.append(
                    TornadoOffTimerControlSwitch(
                        coordinator,
                        device,
                    )
                )
                
                # Add CoolDown switch (Configuration section) 
                entities.append(
                    TornadoCoolDownSwitch(
                        coordinator,
                        device,
                    )
                )
            except Exception:
                _LOGGER.exception(
                    "Error setting up switches for device %s", device.get("endpointId")
                )

        async_add_entities(entities)

    except Exception:
        _LOGGER.exception("Error setting up Tornado switch platform")


class TornadoSleepModeSwitch(CoordinatorEntity, SwitchEntity):
    """Representation of a Tornado AC Sleep Mode switch."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the sleep mode switch."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_sleep_mode"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up switch entity attributes
        self.entity_description = SwitchEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} Sleep Mode",
            translation_key=f"{DOMAIN}_sleep_mode",
            device_class=SwitchDeviceClass.SWITCH,
            icon="mdi:sleep",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} Sleep Mode"
        self._attr_is_on = False

        _LOGGER.info("Sleep mode switch initialized for device %s", self._device_id)

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        _LOGGER.debug(
            "Handling coordinator update for sleep mode switch %s with data: %s",
            self._device_id,
            self._device,
        )

        if not self._device:
            self._attr_available = False
            self.async_write_ha_state()
            return

        try:
            device_params = self._device.get("params", {})
            
            # Update switch state based on sleep mode parameter
            self._attr_is_on = bool(device_params.get("ac_slp", 0))
            self._attr_available = True

            _LOGGER.debug(
                "Updated sleep mode switch state for %s: is_on=%s",
                self._device_id,
                self._attr_is_on,
            )

        except Exception:
            _LOGGER.exception("Error updating sleep mode switch state for %s", self._device_id)
            self._attr_available = False

        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on sleep mode."""
        try:
            _LOGGER.info("Turning on sleep mode for device %s", self._device_id)
            await self.coordinator.api.set_device_params(
                self._device, {"ac_slp": 1}
            )
        except Exception:
            _LOGGER.exception(
                "Error turning on sleep mode for device %s",
                self._device_id,
            )

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off sleep mode."""
        try:
            _LOGGER.info("Turning off sleep mode for device %s", self._device_id)
            await self.coordinator.api.set_device_params(
                self._device, {"ac_slp": 0}
            )
        except Exception:
            _LOGGER.exception(
                "Error turning off sleep mode for device %s",
                self._device_id,
            )


class TornadoEcoModeSwitch(CoordinatorEntity, SwitchEntity):
    """Representation of a Tornado AC Eco Mode switch."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the eco mode switch."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_eco_mode"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up switch entity attributes
        self.entity_description = SwitchEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} Eco Mode",
            translation_key=f"{DOMAIN}_eco_mode",
            device_class=SwitchDeviceClass.SWITCH,
            icon="mdi:leaf",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} Eco Mode"
        self._attr_is_on = False

        _LOGGER.info("Eco mode switch initialized for device %s", self._device_id)

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        _LOGGER.debug(
            "Handling coordinator update for eco mode switch %s with data: %s",
            self._device_id,
            self._device,
        )

        if not self._device:
            self._attr_available = False
            self.async_write_ha_state()
            return

        try:
            device_params = self._device.get("params", {})
            
            # Update switch state based on eco mode parameter
            self._attr_is_on = bool(device_params.get("ecomode", 0))
            self._attr_available = True

            _LOGGER.debug(
                "Updated eco mode switch state for %s: is_on=%s",
                self._device_id,
                self._attr_is_on,
            )

        except Exception:
            _LOGGER.exception("Error updating eco mode switch state for %s", self._device_id)
            self._attr_available = False

        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on eco mode."""
        try:
            _LOGGER.info("Turning on eco mode for device %s", self._device_id)
            await self.coordinator.api.set_device_params(
                self._device, {"ecomode": 1}
            )
        except Exception:
            _LOGGER.exception(
                "Error turning on eco mode for device %s",
                self._device_id,
            )

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off eco mode."""
        try:
            _LOGGER.info("Turning off eco mode for device %s", self._device_id)
            await self.coordinator.api.set_device_params(
                self._device, {"ecomode": 0}
            )
        except Exception:
            _LOGGER.exception(
                "Error turning off eco mode for device %s",
                self._device_id,
            )


class TornadoCoolDownSwitch(CoordinatorEntity, RestoreEntity, SwitchEntity):
    """Representation of a Tornado AC CoolDown switch."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the cooldown switch."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_cooldown"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up switch entity attributes
        self.entity_description = SwitchEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} CoolDown",
            translation_key=f"{DOMAIN}_cooldown",
            device_class=SwitchDeviceClass.SWITCH,
            icon="mdi:snowflake-thermometer",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} CoolDown"
        self._attr_is_on = False

        # CoolDown state management
        self._cooldown_active = False
        self._cooldown_timer_handle = None
        self._cooldown_start_time = None
        self._cooldown_end_time = None  # NEW: Track end time for persistence
        # Duration and fan modes are retrieved from select entities when needed
        
        # Store original state to restore after cooldown
        self._original_state = {}
        
        # Store the original starting fan mode used when cooldown was started
        # This prevents configuration changes from affecting active cooldown
        self._original_starting_fan_mode = None

        _LOGGER.info("CoolDown switch initialized for device %s", self._device_id)

    async def async_added_to_hass(self) -> None:
        """Restore cooldown state when entity is added to hass."""
        await super().async_added_to_hass()
        
        # Try to restore previous state
        last_state = await self.async_get_last_state()
        if last_state is None:
            _LOGGER.debug("No previous state found for cooldown switch %s", self._device_id)
            return
        
        _LOGGER.info("Restoring cooldown state for device %s: %s", self._device_id, last_state.state)
        
        # Restore extra data
        last_extra_data = await self.async_get_last_extra_data()
        
        if last_extra_data and last_state.state == "on":
            cooldown_end_time_iso = last_extra_data.as_dict().get("cooldown_end_time")
            original_state = last_extra_data.as_dict().get("original_state")
            original_starting_fan_mode = last_extra_data.as_dict().get("original_starting_fan_mode")
            
            if cooldown_end_time_iso:
                try:
                    cooldown_end_time = dt_util.parse_datetime(cooldown_end_time_iso)
                    now = dt_util.utcnow()
                    
                    if cooldown_end_time and cooldown_end_time > now:
                        # Cooldown is still active - restore state
                        self._cooldown_end_time = cooldown_end_time
                        self._cooldown_active = True
                        self._attr_is_on = True
                        
                        # Restore original state
                        if original_state:
                            self._original_state = original_state
                        if original_starting_fan_mode is not None:
                            self._original_starting_fan_mode = original_starting_fan_mode
                        
                        # Calculate start time from end time (for display purposes)
                        cooldown_duration = self._get_cooldown_duration_from_select()
                        self._cooldown_start_time = cooldown_end_time - timedelta(minutes=cooldown_duration)
                        
                        _LOGGER.info(
                            "Cooldown state restored for device %s - scheduled to complete at %s",
                            self._device_id,
                            cooldown_end_time.isoformat()
                        )
                        
                        # IMPORTANT: Validate the device is still in the expected state
                        # Wait for coordinator to get initial data before validation
                        await self.coordinator.async_config_entry_first_refresh()
                        
                        # Now check if we should still continue the cooldown
                        should_cancel, reason = self._should_cancel_cooldown()
                        if should_cancel:
                            _LOGGER.warning(
                                "Cooldown restored but device state invalid for device %s: %s - cancelling",
                                self._device_id,
                                reason
                            )
                            await self._cancel_cooldown()
                            return
                        
                        # State is valid - resume the timer
                        self._cooldown_timer_handle = async_track_point_in_time(
                            self.hass,
                            self._complete_cooldown,
                            cooldown_end_time
                        )
                        
                        _LOGGER.info(
                            "Resumed cooldown timer for device %s - will complete at %s",
                            self._device_id,
                            cooldown_end_time.isoformat()
                        )
                    else:
                        # Cooldown expired while Home Assistant was off - complete it now
                        _LOGGER.info(
                            "Cooldown expired while offline for device %s - completing now",
                            self._device_id
                        )
                        await self._complete_cooldown(now)
                        
                except Exception:
                    _LOGGER.exception("Error restoring cooldown state for device %s", self._device_id)

    @property
    def extra_state_attributes(self) -> dict:
        """Return extra state attributes for display AND persistence."""
        attributes = {
            "cooldown_active": self._cooldown_active,
            "cooldown_duration": self._get_cooldown_duration_from_select(),
            "starting_fan_mode": self._get_starting_fan_mode_from_select(),
            "target_fan_mode": self._get_target_fan_mode_from_select(),
        }
        
        if self._cooldown_active and self._cooldown_start_time:
            cooldown_duration = self._get_cooldown_duration_from_select()
            elapsed = (datetime.now() - self._cooldown_start_time).total_seconds()
            remaining = max(0, (cooldown_duration * 60) - elapsed)
            attributes.update({
                "cooldown_start_time": self._cooldown_start_time.isoformat(),
                "remaining_seconds": int(remaining),
                "remaining_minutes": int(remaining / 60) + (1 if remaining % 60 > 0 else 0),
            })
            
        return attributes

    @property
    def extra_restore_state_data(self) -> dict[str, Any] | None:
        """Return extra data to persist for this entity."""
        if not self._cooldown_active or not self._cooldown_end_time:
            return None
            
        return {
            "cooldown_end_time": self._cooldown_end_time.isoformat(),
            "original_state": self._original_state,
            "original_starting_fan_mode": self._original_starting_fan_mode,
        }

    @property
    def entity_category(self) -> EntityCategory | None:
        """Return the entity category (Controls section)."""
        return None  # Controls section (no category)

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    def _should_cancel_cooldown(self) -> tuple[bool, str]:
        """Check if cooldown should be cancelled based on device state.
        
        NOTE: This method assumes device is available. The caller should check
        device availability separately before calling this method.
        
        Returns:
            tuple[bool, str]: (should_cancel, reason)
        """
        if not self._cooldown_active:
            return False, ""
            
        # Device availability is checked by the caller
        # We assume device exists when this method is called
        if not self._device:
            return False, ""  # Let caller handle unavailability
            
        device_params = self._device.get("params", {})
        
        # SAFETY CHECK 1: AC turned off during cooldown
        if not device_params.get("pwr", 0):
            return True, "AC turned off"
        
        # SAFETY CHECK 2: User changed AC settings during cooldown
        current_mode = device_params.get("ac_mode", 0)
        current_fan = device_params.get("ac_mark", 0)
        current_power = device_params.get("pwr", 0)
        
        # Get the ORIGINAL starting fan mode that was used when cooldown started
        original_starting_fan = getattr(self, '_original_starting_fan_mode', None)
        if original_starting_fan is None:
            # Fallback: use current starting fan mode if we don't have the original
            original_starting_fan = self._get_starting_fan_mode_from_select()
        
        # Expected state during cooldown: Cool mode (0) + ORIGINAL starting fan mode + Power on (1)
        expected_state = (
            current_power == 1 and 
            current_mode == 0 and 
            current_fan == original_starting_fan
        )
        
        if not expected_state:
            return True, f"Hardware settings changed (pwr:{current_power}, mode:{current_mode}, fan:{current_fan}, expected_fan:{original_starting_fan})"
        
        return False, ""

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        _LOGGER.debug(
            "Handling coordinator update for cooldown switch %s with data: %s",
            self._device_id,
            self._device,
        )

        if not self._device:
            self._attr_available = False
            if self._cooldown_active:
                # Don't cancel immediately - device might reconnect
                # Timer will be validated on reconnection or cancelled by restoration timeout
                _LOGGER.warning(
                    "Device unavailable during cooldown for %s - timer paused, will validate on reconnection",
                    self._device_id
                )
            self.async_write_ha_state()
            return

        try:
            # Device is available now
            self._attr_available = True
            
            # Validate cooldown state during updates
            if self._cooldown_active:
                should_cancel, reason = self._should_cancel_cooldown()
                if should_cancel:
                    _LOGGER.info("Cancelling CoolDown for %s: %s", self._device_id, reason)
                    self.hass.async_create_task(self._cancel_cooldown())

            _LOGGER.debug(
                "Updated cooldown switch state for %s: is_on=%s, active=%s",
                self._device_id,
                self._attr_is_on,
                self._cooldown_active,
            )

        except Exception:
            _LOGGER.exception("Error updating cooldown switch state for %s", self._device_id)
            self._attr_available = False

        self.async_write_ha_state()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on cooldown mode."""
        try:
            # Get current configuration from select entities
            starting_fan_mode = self._get_starting_fan_mode_from_select()
            cooldown_duration = self._get_cooldown_duration_from_select()

            _LOGGER.info("Starting CoolDown for device %s, using starting fan mode %s, will switch to target fan mode in %d minutes",
                          self._device_id, starting_fan_mode, cooldown_duration)

            # Cancel any existing cooldown first
            if self._cooldown_active:
                await self._cancel_cooldown()
            
            # Store current state to restore later AND the original starting fan mode
            if self._device:
                device_params = self._device.get("params", {})
                self._original_state = {
                    "ac_mark": device_params.get("ac_mark", 1),  # Default to low if unknown
                    "was_on": bool(device_params.get("pwr", 0)),
                }
            
            # Store the starting fan mode that will be used for this cooldown session
            # This prevents configuration changes from affecting the active cooldown
            self._original_starting_fan_mode = starting_fan_mode
            
            # Step 1: Turn on AC, set to Cool mode with configurable starting fan
            cooldown_params = {
                "pwr": 1,                    # Turn on AC
                "ac_mode": 0,                # Cool mode
                "ac_mark": starting_fan_mode, # Configurable starting fan mode
            }
            
            await self.coordinator.api.set_device_params(self._device, cooldown_params)
            
            # Step 2: Set up cooldown timer
            self._cooldown_active = True
            self._attr_is_on = True
            self._cooldown_start_time = datetime.now()
            
            # Schedule transition to target fan mode
            end_time = dt_util.utcnow() + timedelta(minutes=cooldown_duration)
            self._cooldown_end_time = end_time  # Store for persistence
            self._cooldown_timer_handle = async_track_point_in_time(
                self.hass,
                self._complete_cooldown,
                end_time
            )
            
            _LOGGER.info(
                "CoolDown started for %s - using starting fan mode %s, will switch to target fan mode in %d minutes",
                self._device_id,
                starting_fan_mode,
                cooldown_duration
            )
            
            self.async_write_ha_state()
            
        except Exception:
            _LOGGER.exception(
                "Error starting cooldown for device %s",
                self._device_id,
            )
            await self._cancel_cooldown()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off cooldown mode."""
        try:
            _LOGGER.info("Manually stopping CoolDown for device %s", self._device_id)
            await self._cancel_cooldown()
        except Exception:
            _LOGGER.exception(
                "Error stopping cooldown for device %s",
                self._device_id,
            )

    async def _cancel_cooldown(self) -> None:
        """Cancel the active cooldown and cleanup."""
        try:
            if self._cooldown_timer_handle:
                self._cooldown_timer_handle()
                self._cooldown_timer_handle = None
            
            self._cooldown_active = False
            self._attr_is_on = False
            self._cooldown_start_time = None
            self._cooldown_end_time = None  # Clear end time for persistence
            # Clear the stored original starting fan mode
            self._original_starting_fan_mode = None
            
            _LOGGER.info("CoolDown cancelled for device %s", self._device_id)
            self.async_write_ha_state()
            
        except Exception:
            _LOGGER.exception("Error cancelling cooldown for device %s", self._device_id)

    async def _complete_cooldown(self, now) -> None:
        """Complete the cooldown by switching to target fan mode."""
        try:
            # Get current target fan mode from select
            target_fan_mode = self._get_target_fan_mode_from_select()
            
            _LOGGER.info(
                "CoolDown timer expired for %s - switching to fan mode %s",
                self._device_id,
                target_fan_mode
            )
            
            # Switch to target fan mode (keeping Cool mode and AC on)
            completion_params = {
                "ac_mark": target_fan_mode,  # Switch to target fan mode
            }
            
            await self.coordinator.api.set_device_params(self._device, completion_params)
            
            # Mark cooldown as complete
            self._cooldown_active = False
            self._attr_is_on = False
            self._cooldown_start_time = None
            self._cooldown_end_time = None  # Clear end time for persistence
            # Clear the stored original starting fan mode
            self._original_starting_fan_mode = None
            
            _LOGGER.info("CoolDown completed for device %s", self._device_id)
            self.async_write_ha_state()
            
        except Exception:
            _LOGGER.exception("Error completing cooldown for device %s", self._device_id)
            await self._cancel_cooldown()

    def _get_starting_fan_select(self):
        """Get the starting fan select entity for this device."""
        starting_fan_unique_id = f"{self._device_id}_cooldown_starting_fan"
        return find_entity_by_unique_id(self.hass, starting_fan_unique_id, "current_option")

    def _get_target_fan_select(self):
        """Get the target fan select entity for this device."""
        target_fan_unique_id = f"{self._device_id}_cooldown_target_fan"
        return find_entity_by_unique_id(self.hass, target_fan_unique_id, "current_option")

    def _get_starting_fan_mode_from_select(self) -> int:
        """Get starting fan mode from select entity."""
        starting_fan_select = self._get_starting_fan_select()
        if starting_fan_select and hasattr(starting_fan_select, 'current_option'):
            return FAN_MODE_MAP_REVERSE.get(starting_fan_select.current_option, 4)  # Default to turbo
        return 4  # Default to turbo if select not found

    def _get_target_fan_mode_from_select(self) -> int:
        """Get target fan mode from select entity."""
        target_fan_select = self._get_target_fan_select()
        if target_fan_select and hasattr(target_fan_select, 'current_option'):
            return FAN_MODE_MAP_REVERSE.get(target_fan_select.current_option, 5)  # Default to silent
        return 5  # Default to silent if select not found

    def _get_cooldown_duration_from_select(self) -> int:
        """Get cooldown duration from select entity."""
        cooldown_duration_unique_id = f"{self._device_id}_cooldown_duration"
        entity = find_entity_by_unique_id(self.hass, cooldown_duration_unique_id, "current_option")
        
        if entity and entity.current_option:
            return COOLDOWN_DURATION_MAP.get(entity.current_option, 5)  # Default to 5 minutes
        
        return 5  # Default to 5 minutes if select not found

    async def async_will_remove_from_hass(self) -> None:
        """Clean up when entity is removed."""
        if self._cooldown_timer_handle:
            self._cooldown_timer_handle.cancel()
            self._cooldown_timer_handle = None


class TornadoOffTimerControlSwitch(CoordinatorEntity, RestoreEntity, SwitchEntity):
    """Representation of a Tornado AC Off Timer Control switch."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the off timer control switch."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_off_timer_control"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up switch entity attributes - this goes in Controls section
        self.entity_description = SwitchEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} Off Timer",
            translation_key=f"{DOMAIN}_off_timer_control",
            device_class=SwitchDeviceClass.SWITCH,
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} Off Timer"
        self._attr_icon = "mdi:timer"
        
        # Off timer state tracking
        self._timer_active = False
        self._timer_end_time = None
        self._periodic_update_handle = None
        # Off timer duration is retrieved from select entity when needed
        
        _LOGGER.info("Off timer control switch initialized for device %s", self._device_id)

    async def async_added_to_hass(self) -> None:
        """Restore off timer state when entity is added to hass."""
        await super().async_added_to_hass()
        
        # Try to restore previous state
        last_state = await self.async_get_last_state()
        if last_state is None:
            _LOGGER.debug("No previous state found for off timer switch %s", self._device_id)
            return
        
        _LOGGER.info("Restoring off timer state for device %s: %s", self._device_id, last_state.state)
        
        # Restore extra data
        last_extra_data = await self.async_get_last_extra_data()
        
        if last_extra_data and last_state.state == "on":
            timer_end_time_iso = last_extra_data.as_dict().get("timer_end_time")
            
            if timer_end_time_iso:
                try:
                    timer_end_time = dt_util.parse_datetime(timer_end_time_iso)
                    now = dt_util.utcnow()
                    
                    if timer_end_time and timer_end_time > now:
                        # Timer is still active - validate device state before resuming
                        self._timer_end_time = timer_end_time
                        self._timer_active = True
                        
                        _LOGGER.info(
                            "Off timer state restored for device %s - scheduled to complete at %s",
                            self._device_id,
                            timer_end_time.isoformat()
                        )
                        
                        # IMPORTANT: Validate the device is still ON
                        # Wait for coordinator to get initial data before validation
                        await self.coordinator.async_config_entry_first_refresh()
                        
                        # Check if device is still ON (off timer only makes sense if device is ON)
                        if not self._device:
                            _LOGGER.warning(
                                "Off timer restored but device unavailable for device %s - cancelling",
                                self._device_id
                            )
                            await self._cancel_timer()
                            return
                        
                        device_params = self._device.get("params", {})
                        if not device_params.get("pwr", 0):
                            _LOGGER.warning(
                                "Off timer restored but device is already OFF for device %s - cancelling",
                                self._device_id
                            )
                            await self._cancel_timer()
                            return
                        
                        # Device is still ON - resume the timer and periodic updates
                        if self._periodic_update_handle:
                            self._periodic_update_handle()
                        self._periodic_update_handle = async_track_time_interval(
                            self.hass,
                            self._update_timer_sensor,
                            timedelta(seconds=30)
                        )
                        
                        _LOGGER.info(
                            "Resumed off timer for device %s - will complete at %s",
                            self._device_id,
                            timer_end_time.isoformat()
                        )
                    else:
                        # Timer expired while Home Assistant was off - complete it now
                        _LOGGER.info(
                            "Off timer expired while offline for device %s - turning off AC now",
                            self._device_id
                        )
                        await self._timer_completed(now)
                        
                except Exception:
                    _LOGGER.exception("Error restoring off timer state for device %s", self._device_id)

    @property
    def extra_restore_state_data(self) -> dict[str, Any] | None:
        """Return extra data to persist for this entity."""
        if not self._timer_active or not self._timer_end_time:
            return None
            
        return {
            "timer_end_time": self._timer_end_time.isoformat(),
        }

    @property
    def entity_category(self) -> str | None:
        """Return the entity category (Controls section)."""
        return None  # Controls section (no category)

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    @property
    def is_on(self) -> bool:
        """Return true if off timer is active."""
        return self._timer_active

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        _LOGGER.debug(
            "Handling coordinator update for off timer control switch %s with data: %s",
            self._device_id,
            self._device,
        )

        if not self._device:
            self._attr_available = False
            if self._timer_active:
                # Timer will be validated on reconnection or cancelled by restoration timeout
                _LOGGER.warning(
                    "Device unavailable during off timer for %s - timer paused, will validate on reconnection",
                    self._device_id
                )
            self.async_write_ha_state()
            return

        try:
            device_params = self._device.get("params", {})
            self._attr_available = True
            
            # Validate timer state during updates
            # SAFETY CHECK: If device is turned off while off timer is active, cancel the timer
            if self._timer_active and not device_params.get("pwr", 0):
                _LOGGER.info("Device turned off - cancelling off timer for %s", self._device_id)
                self.hass.async_create_task(self._cancel_timer())

            _LOGGER.debug(
                "Updated off timer control switch state for %s: timer_active=%s, device_power=%s",
                self._device_id,
                self._timer_active,
                device_params.get("pwr", 0),
            )

        except Exception:
            _LOGGER.exception("Error updating off timer control switch state for %s", self._device_id)
            self._attr_available = False

        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        attributes = {
            "off_timer_duration": self._get_off_timer_duration_from_select(),
        }
        if self._timer_end_time:
            attributes["timer_end_time"] = self._timer_end_time.isoformat()
            remaining = (self._timer_end_time - dt_util.utcnow()).total_seconds()
            attributes["remaining_minutes"] = max(0, int(remaining / 60))
        return attributes

    async def async_turn_on(self) -> None:
        """Turn on the off timer (start timer with selected duration)."""
        try:
            # Get off timer duration from select entity
            duration_minutes = self._get_off_timer_duration_from_select()
            if duration_minutes <= 0:
                _LOGGER.warning("No valid off timer duration set, cannot start off timer")
                return

            # Check if device is on using coordinator data instead of climate entity
            device_is_on = False
            if self._device:
                device_params = self._device.get("params", {})
                device_is_on = bool(device_params.get("pwr", 0))
            
            # If device is off, turn it on first using coordinator API
            if not device_is_on:
                _LOGGER.info("Device is off, turning on before starting off timer")
                await self.coordinator.api.set_device_params(self._device, {"pwr": 1})
                # Wait a bit for the state to update
                await asyncio.sleep(1)

            # Start the off timer
            await self._start_timer(duration_minutes)
            
        except Exception as e:
            _LOGGER.error("Error starting off timer: %s", e)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the off timer (cancel active timer)."""
        await self._cancel_timer()

    async def _start_timer(self, duration_minutes: int) -> None:
        """Start the off timer with specified duration."""
        # Cancel any existing off timer
        if self._periodic_update_handle:
            self._periodic_update_handle()
            self._periodic_update_handle = None

        self._timer_active = True
        self._timer_end_time = dt_util.utcnow() + timedelta(minutes=duration_minutes)
        
        # Update off timer sensor immediately
        await self._update_timer_sensor()
        
        # Start periodic updates to sensor every 30 seconds (this will also check for completion)
        self._start_periodic_sensor_updates()
        
        self.async_write_ha_state()
        _LOGGER.info("Off timer started for %d minutes on device %s", duration_minutes, self._device_id)

    async def _cancel_timer(self) -> None:
        """Cancel the active off timer."""
        # Cancel periodic updates
        if self._periodic_update_handle:
            self._periodic_update_handle()
            self._periodic_update_handle = None

        self._timer_active = False
        self._timer_end_time = None
        
        # Update off timer sensor
        await self._update_timer_sensor()
        
        self.async_write_ha_state()
        _LOGGER.info("Off timer cancelled for device %s", self._device_id)

    async def _timer_completed(self) -> None:
        """Handle off timer completion."""
        _LOGGER.info("Off timer completed for device %s, turning off", self._device_id)
        
        # Check if device is available BEFORE trying to turn it off
        if not self._device:
            _LOGGER.debug(
                "Device unavailable when off timer completed for %s - will retry on next check",
                self._device_id
            )
            # DON'T clear timer state - keep retrying
            # DON'T stop periodic updates - we need them to keep checking!
            return
        
        # Device is available - turn it off
        try:
            await self.coordinator.api.set_device_params(self._device, {"pwr": 0})
        except Exception as e:
            _LOGGER.error("Error turning off device after off timer completion: %s", e)
            # API failed but device is available - will retry on next check
            return
        
        # SUCCESS - now clean up everything
        if self._periodic_update_handle:
            self._periodic_update_handle()
            self._periodic_update_handle = None
    
        self._timer_active = False
        self._timer_end_time = None
        
        await self._update_timer_sensor()
        self.async_write_ha_state()
        
        _LOGGER.info("Off timer completed successfully for device %s", self._device_id)

    def _start_periodic_sensor_updates(self) -> None:
        """Start periodic updates to the off timer sensor every 30 seconds."""
        # Cancel any existing periodic updates
        if self._periodic_update_handle:
            self._periodic_update_handle()
        
        # Start new periodic updates every 30 seconds
        self._periodic_update_handle = async_track_time_interval(
            self.hass,
            self._periodic_sensor_update,
            timedelta(seconds=30)
        )

    async def _periodic_sensor_update(self, now) -> None:
        """Periodic callback to update the off timer sensor and check for completion."""
        if not self._timer_active:
            _LOGGER.debug("Off timer not active, skipping periodic sensor update")
            return
            
        # Check if off timer has completed
        if self._timer_end_time and dt_util.utcnow() >= self._timer_end_time:
            await self._timer_completed()
        else:
            # Off timer still active, update sensor
            await self._update_timer_sensor()

    def _get_off_timer_duration_from_select(self) -> int:
        """Get off timer duration from select entity."""
        off_timer_duration_unique_id = f"{self._device_id}_off_timer_duration"
        entity = find_entity_by_unique_id(self.hass, off_timer_duration_unique_id, "current_option")
        
        if entity:
            option = entity.current_option
            if option:
                return OFF_TIMER_DURATION_MAP.get(option, 30)  # Default to 30 minutes
            else:
                _LOGGER.debug("Off timer duration select has no current option set")
                return 0
        
        _LOGGER.debug("Off timer duration select entity not found, defaulting to 30 minutes")
        return 0
    
    async def _update_timer_sensor(self) -> None:
        """Update the off timer sensor with current timer state."""
        sensor_unique_id = f"{self._device_id}_off_timer"
        
        _LOGGER.debug("Searching for off timer sensor with unique_id: %s", sensor_unique_id)
        
        sensor_entity = find_entity_by_unique_id(self.hass, sensor_unique_id, "async_update_timer_state_from_control")
        
        if sensor_entity:
            _LOGGER.debug(
                "Found off timer sensor by unique_id %s, updating with timer_active=%s, timer_end_time=%s",
                sensor_unique_id, self._timer_active, self._timer_end_time
            )
            await sensor_entity.async_update_timer_state_from_control(
                self._timer_active, 
                self._timer_end_time
            )
        else:
            _LOGGER.warning(
                "Off timer sensor not found for device %s (unique_id: %s)",
                self._device_id, sensor_unique_id
            )

    async def async_will_remove_from_hass(self) -> None:
        """Clean up when entity is removed."""
        if self._periodic_update_handle:
            self._periodic_update_handle()
            self._periodic_update_handle = None
