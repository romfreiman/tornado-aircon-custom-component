"""Platform for Tornado AC sensor integration."""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorEntityDescription,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.const import UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN

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
    """Set up Tornado sensor platform."""
    # Get the coordinator from the entry data (created in __init__.py)
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = entry_data.get("coordinator")
    
    if not coordinator:
        _LOGGER.error("No coordinator found for sensor setup")
        return

    _LOGGER.info("Setting up off timer sensors with coordinator from __init__.py")

    try:
        devices = await coordinator.api.get_devices()
        entities = []

        for device in devices:
            try:
                entities.append(
                    TornadoOffTimerSensor(
                        coordinator,
                        device,
                    )
                )
                _LOGGER.info("Created off timer sensor for device %s", device.get("endpointId"))
            except Exception:
                _LOGGER.exception(
                    "Error setting up off timer sensor for device %s", device.get("endpointId")
                )

        if entities:
            async_add_entities(entities)
            _LOGGER.info("Added %d off timer sensor entities", len(entities))
        else:
            _LOGGER.warning("No off timer sensor entities created")

    except Exception:
        _LOGGER.exception("Error setting up Tornado sensor platform")


class TornadoOffTimerSensor(CoordinatorEntity, SensorEntity):
    """Representation of a Tornado AC Off Timer sensor."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the off timer sensor."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_off_timer"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up sensor attributes
        self.entity_description = SensorEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} Off Timer Status",
            translation_key=f"{DOMAIN}_off_timer_status",
            device_class=SensorDeviceClass.DURATION,
            state_class=SensorStateClass.MEASUREMENT,
            native_unit_of_measurement=UnitOfTime.MINUTES,
            icon="mdi:timer-outline",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} Off Timer Status"
        self._timer_end_time = None
        self._attr_native_value = 0
        self._attr_extra_state_attributes = {
            "timer_active": False,
            "display_text": "No off timer",
        }

        _LOGGER.info("Off timer sensor initialized for device %s", self._device_id)

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        if not self._device:
            self._attr_available = False
        else:
            self._attr_available = True
        
        self.async_write_ha_state()

    def _format_time_remaining(self, total_seconds: int) -> str:
        """Format remaining time in a human-readable way."""
        if total_seconds <= 0:
            return "Off timer finished"
        
        total_minutes = math.ceil(total_seconds / 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    async def async_update_timer_state_from_control(self, timer_active: bool, timer_end_time: datetime = None) -> None:
        """Update off timer state from external timer control switch."""
        if timer_active and timer_end_time is not None:
            # Off timer is active
            self._timer_end_time = timer_end_time
            
            # Calculate remaining time using consistent datetime types
            now = dt_util.utcnow()
            if now >= timer_end_time:
                remaining_seconds = 0
                remaining_minutes = 0
                display_text = "Off timer finished"
                timer_active = False
            else:
                remaining_seconds = int((timer_end_time - now).total_seconds())
                remaining_seconds = max(0, remaining_seconds)  # Ensure non-negative
                
                # For display purposes, always show at least 1 minute if there are any seconds left
                remaining_minutes = math.ceil(remaining_seconds / 60) if remaining_seconds > 0 else 0
                display_text = self._format_time_remaining(remaining_seconds)
            
            self._attr_native_value = remaining_minutes
            self._attr_extra_state_attributes = {
                "timer_active": timer_active,
                "display_text": display_text,
                "timer_end_time": timer_end_time.isoformat() if timer_end_time else None,
                "remaining_seconds": remaining_seconds,
            }
            
            _LOGGER.debug(
                "Off timer sensor updated for %s: active=%s, remaining_seconds=%s, remaining_minutes=%s, display=%s",
                self._device_id, timer_active, remaining_seconds, remaining_minutes, display_text
            )
        else:
            # Off timer cancelled or not active
            self._timer_end_time = None
            self._attr_native_value = 0
            self._attr_extra_state_attributes = {
                "timer_active": False,
                "display_text": "No off timer",
                "timer_end_time": None,
                "remaining_seconds": 0,
            }
            
            _LOGGER.debug(
                "Off timer sensor cleared for %s: off timer cancelled or not active",
                self._device_id
            )
        
        self.async_write_ha_state()
