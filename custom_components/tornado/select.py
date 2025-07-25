"""Platform for Tornado AC select integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.select import (
    SelectEntity,
    SelectEntityDescription,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    OFF_TIMER_DURATION_MAP,
    COOLDOWN_DURATION_MAP,
)
from .climate import FAN_MODE_MAP, FAN_MODE_MAP_REVERSE

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from . import AuxCloudDataUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

# Use fan modes from climate.py to avoid duplication
FAN_MODE_OPTIONS = list(FAN_MODE_MAP.values())
FAN_MODE_NAME_TO_VALUE = FAN_MODE_MAP_REVERSE
FAN_MODE_VALUE_TO_NAME = FAN_MODE_MAP


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Tornado select platform."""
    # Get the coordinator from the entry data (created in __init__.py)
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    coordinator = entry_data.get("coordinator")
    
    if not coordinator:
        _LOGGER.error("No coordinator found for select setup")
        return

    _LOGGER.info("Setting up select entities with coordinator from __init__.py")

    try:
        devices = await coordinator.api.get_devices()
        entities = []

        for device in devices:
            try:
 
                # Add CoolDown Starting Fan select
                entities.append(
                    TornadoCoolDownStartingFanSelect(
                        coordinator,
                        device,
                    )
                )
                _LOGGER.info("Created cooldown starting fan select for device %s", device.get("endpointId"))
                
                # Add CoolDown Target Fan select
                entities.append(
                    TornadoCoolDownTargetFanSelect(
                        coordinator,
                        device,
                    )
                )
                _LOGGER.info("Created cooldown target fan select for device %s", device.get("endpointId"))
                
                # Add CoolDown Duration select
                entities.append(
                    TornadoCoolDownDurationSelect(
                        coordinator,
                        device,
                    )
                )
                _LOGGER.info("Created cooldown duration select for device %s", device.get("endpointId"))
                
                # Add Off Timer Duration select (Configuration section)
                entities.append(
                    TornadoOffTimerDurationSelect(
                        coordinator,
                        device,
                    )
                )
                _LOGGER.info("Created off timer duration select for device %s", device.get("endpointId"))
                              
            except Exception:
                _LOGGER.exception(
                    "Error setting up select entities for device %s", device.get("endpointId")
                )

        if entities:
            async_add_entities(entities)
            _LOGGER.info("Added %d select entities", len(entities))
        else:
            _LOGGER.warning("No select entities created")

    except Exception:
        _LOGGER.exception("Error setting up Tornado select platform")


class TornadoCoolDownStartingFanSelect(CoordinatorEntity, SelectEntity):
    """Representation of a Tornado AC CoolDown Starting Fan select."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the cooldown starting fan select."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_cooldown_starting_fan"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up select entity attributes
        self.entity_description = SelectEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} CoolDown Starting Fan",
            translation_key=f"{DOMAIN}_cooldown_starting_fan",
            icon="mdi:fan-speed-1",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} CoolDown Starting Fan"
        self._attr_options = FAN_MODE_OPTIONS
        self._attr_current_option = "high"  # Default starting fan mode

        _LOGGER.info("CoolDown starting fan select initialized for device %s", self._device_id)

    @property
    def entity_category(self) -> EntityCategory | None:
        """Return the entity category (Configuration section)."""
        return EntityCategory.CONFIG

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        if not self._device:
            return False
        
        # Keep entity available even when cooldown is active
        return True

    @property
    def icon(self) -> str:
        """Return the icon for the entity."""
        fan_mode_value = FAN_MODE_NAME_TO_VALUE.get(self._attr_current_option, 4)
        icons = {
            0: "mdi:fan-auto",
            1: "mdi:fan-speed-1", 
            2: "mdi:fan-speed-2",
            3: "mdi:fan-speed-3",
            4: "mdi:fan-plus",
            5: "mdi:fan-minus",
        }
        return icons.get(fan_mode_value, "mdi:fan")

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        try:
            fan_mode_value = FAN_MODE_NAME_TO_VALUE.get(option, 4)
            _LOGGER.info(
                "Setting cooldown starting fan mode for device %s to %s (%d)",
                self._device_id, option, fan_mode_value
            )

            # Update our value
            self._attr_current_option = option

            _LOGGER.info(
                "CoolDown starting fan mode changed to '%s' for device %s",
                option, self._device_id
            )

            self.async_write_ha_state()

        except Exception:
            _LOGGER.exception(
                "Error setting cooldown starting fan mode for device %s", self._device_id
            )

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        if not self._device:
            self._attr_available = False
        else:
            self._attr_available = True

        self.async_write_ha_state()


class TornadoCoolDownTargetFanSelect(CoordinatorEntity, SelectEntity):
    """Representation of a Tornado AC CoolDown Target Fan select."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the cooldown target fan select."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_cooldown_target_fan"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up select entity attributes
        self.entity_description = SelectEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} CoolDown Target Fan",
            translation_key=f"{DOMAIN}_cooldown_target_fan",
            icon="mdi:fan-minus",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} CoolDown Target Fan"
        self._attr_options = FAN_MODE_OPTIONS
        self._attr_current_option = "silent"  # Default target fan mode

        _LOGGER.info("CoolDown target fan select initialized for device %s", self._device_id)

    @property
    def entity_category(self) -> EntityCategory | None:
        """Return the entity category (Configuration section)."""
        return EntityCategory.CONFIG

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._device_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        if not self._device:
            return False
        
        # Keep entity available even when cooldown is active
        return True

    @property
    def icon(self) -> str:
        """Return the icon for the entity."""
        fan_mode_value = FAN_MODE_NAME_TO_VALUE.get(self._attr_current_option, 5)
        icons = {
            0: "mdi:fan-auto",
            1: "mdi:fan-speed-1", 
            2: "mdi:fan-speed-2",
            3: "mdi:fan-speed-3",
            4: "mdi:fan-plus",
            5: "mdi:fan-minus",
        }
        return icons.get(fan_mode_value, "mdi:fan")

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        try:
            fan_mode_value = FAN_MODE_NAME_TO_VALUE.get(option, 5)
            _LOGGER.info(
                "Setting cooldown target fan mode for device %s to %s (%d)",
                self._device_id, option, fan_mode_value
            )

            # Update our value
            self._attr_current_option = option

            _LOGGER.info(
                "CoolDown target fan mode changed to '%s' for device %s",
                option, self._device_id
            )

            self.async_write_ha_state()

        except Exception:
            _LOGGER.exception(
                "Error setting cooldown target fan mode for device %s", self._device_id
            )

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        if not self._device:
            self._attr_available = False
        else:
            self._attr_available = True

        self.async_write_ha_state()


class TornadoOffTimerDurationSelect(CoordinatorEntity, SelectEntity):
    """Representation of a Tornado AC Off Timer Duration select."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the off timer duration select."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_off_timer_duration"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Off timer duration options
        self._attr_options = list(OFF_TIMER_DURATION_MAP.keys())
        self._attr_current_option = "30m"  # Default to 30 minutes

        # Set up select entity attributes - Configuration section
        self.entity_description = SelectEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} Off Timer Duration",
            translation_key=f"{DOMAIN}_off_timer_duration",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} Off Timer Duration"
        self._attr_icon = "mdi:timer-cog"
        
        _LOGGER.info("Off timer duration select initialized for device %s", self._device_id)

    @property
    def entity_category(self) -> EntityCategory | None:
        """Return the entity category (Configuration section)."""
        return EntityCategory.CONFIG  # Configuration section

    @property
    def current_option(self) -> str | None:
        """Return the current selected option."""
        return self._attr_current_option

    @property
    def options(self) -> list[str]:
        """Return the list of available options."""
        return self._attr_options

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        if option not in self._attr_options:
            _LOGGER.error("Invalid off timer duration option: %s", option)
            return

        try:
            # Parse duration from option string
            duration_minutes = OFF_TIMER_DURATION_MAP.get(option, 30)
            
            _LOGGER.info(
                "Setting off timer duration for device %s to %s (%d minutes)",
                self._device_id, option, duration_minutes
            )

            # Update our current option
            self._attr_current_option = option
            self.async_write_ha_state()

        except Exception:
            _LOGGER.exception(
                "Error setting off timer duration for device %s", self._device_id
            )

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        # Off timer duration is not dependent on device state
        self._attr_available = True
        self.async_write_ha_state()


class TornadoCoolDownDurationSelect(CoordinatorEntity, SelectEntity):
    """Representation of a Tornado AC CoolDown Duration select."""

    def __init__(
        self,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
    ) -> None:
        """Initialize the cooldown duration select."""
        super().__init__(coordinator)
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_cooldown_duration"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        # Set up select entity attributes
        self.entity_description = SelectEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')} CoolDown Duration",
            translation_key=f"{DOMAIN}_cooldown_duration",
            icon="mdi:timer-cog",
        )

        self._attr_name = f"Tornado AC {device.get('friendlyName')} CoolDown Duration"
        self._attr_options = list(COOLDOWN_DURATION_MAP.keys())
        self._attr_current_option = "10min"  # Default 10 minutes

        _LOGGER.info("CoolDown duration select initialized for device %s", self._device_id)

    @property
    def entity_category(self) -> EntityCategory | None:
        """Return the entity category (Configuration section)."""
        return EntityCategory.CONFIG

    @property
    def current_option(self) -> str | None:
        """Return the current selected option."""
        return self._attr_current_option

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        # Keep entity available even when cooldown is active
        return True

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        if option not in self._attr_options:
            _LOGGER.error("Invalid cooldown duration option: %s", option)
            return

        try:
            # Parse duration from option string
            duration_minutes = COOLDOWN_DURATION_MAP.get(option, 5)
            
            _LOGGER.info(
                "Setting cooldown duration for device %s to %s (%d minutes)",
                self._device_id, option, duration_minutes
            )

            # Update our current option
            self._attr_current_option = option

            _LOGGER.info(
                "CoolDown duration changed to '%s' for device %s",
                option, self._device_id
            )

            self.async_write_ha_state()

        except Exception:
            _LOGGER.exception(
                "Error setting cooldown duration for device %s", self._device_id
            )

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        # CoolDown duration is not dependent on device state
        self._attr_available = True
        self.async_write_ha_state()
