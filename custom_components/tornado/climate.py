"""Platform for Tornado AC climate integration."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.components.climate import (
    ClimateEntity,
    ClimateEntityDescription,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_TEMPERATURE,
    UnitOfTemperature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .aux_cloud import AuxCloudAPI
from .const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

_LOGGER = logging.getLogger(__name__)

# Map Tornado modes to Home Assistant modes
HVAC_MODE_MAP = {
    0: HVACMode.AUTO,
    1: HVACMode.HEAT,
    2: HVACMode.COOL,
    3: HVACMode.FAN_ONLY,
    4: HVACMode.DRY,
}

HVAC_MODE_MAP_REVERSE = {v: k for k, v in HVAC_MODE_MAP.items()}

# Map Tornado fan modes to Home Assistant fan modes
FAN_MODE_MAP = {
    0: "auto",
    1: "low",
    2: "medium",
    3: "high",
}

FAN_MODE_MAP_REVERSE = {v: k for k, v in FAN_MODE_MAP.items()}

# Available swing modes
SWING_MODES = ["off", "vertical", "horizontal", "both"]

# Parameter validation
PARAMETER_VALIDATION = {
    "ac_vdir": {"type": int, "range": (0, 1), "required": False},
    "ac_hdir": {"type": int, "range": (0, 1), "required": False},
    "ac_mark": {"type": int, "range": (0, 4), "required": True},  # Fan modes
    "ac_mode": {"type": int, "range": (0, 4), "required": True},
    "ac_slp": {"type": int, "range": (0, 1), "required": True},
    "pwr": {"type": int, "range": (0, 1), "required": True},
    "ac_astheat": {"type": int, "range": (0, 1), "required": True},
    "ecomode": {"type": int, "range": (0, 1), "required": True},
    "ac_clean": {"type": int, "range": (0, 1), "required": False},
    "ac_health": {"type": int, "range": (0, 1), "required": False},
    "scrdisp": {"type": int, "range": (0, 1), "required": False},
    "mldprf": {"type": int, "range": (0, 1), "required": False},
    "pwrlimitswitch": {"type": int, "range": (0, 1), "required": False},
    "pwrlimit": {"type": int, "range": (0, 100), "required": False},
    "comfwind": {"type": int, "range": (0, 1), "required": True},
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Tornado climate platform."""
    # Get the client from the domain data
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    client = entry_data["client"]

    # Ensure client is logged in before creating coordinator
    if not hasattr(client, "loginsession") or not client.loginsession:
        _LOGGER.info("Initial login for AuxCloud client")
        await client.login()

    coordinator = AuxCloudDataUpdateCoordinator(hass, client)
    await coordinator.async_refresh()

    try:
        devices = await client.get_devices()
        entities = []

        for device in devices:
            try:
                entities.append(
                    TornadoClimateEntity(
                        hass,
                        coordinator,
                        device,
                    )
                )
            except Exception:
                _LOGGER.exception(
                    "Error setting up device %s", device.get("endpointId")
                )

        async_add_entities(entities)

    except Exception:
        _LOGGER.exception("Error setting up Tornado climate platform")


class AuxCloudDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching AuxCloud data."""

    def __init__(self, hass: HomeAssistant, api: AuxCloudAPI) -> None:
        """Initialize the coordinator."""
        self.api = api
        super().__init__(
            hass,
            _LOGGER,
            name="AuxCloud",
            update_interval=timedelta(minutes=1),
        )

    async def _async_update_data(self) -> dict:
        """Fetch data from AuxCloud."""
        try:
            # First check if we need to re-authenticate
            if not hasattr(self.api, "loginsession") or not self.api.loginsession:
                _LOGGER.info("No valid login session, attempting to login")
                await self.api.login()

            devices = await self.api.get_devices()
            _LOGGER.debug("Coordinator fetched data: %s", devices)
            return {device["endpointId"]: device for device in devices}
        except Exception as err:
            _LOGGER.exception("Error fetching data")
            error_msg = f"Error fetching data: {err}"
            raise UpdateFailed(error_msg) from err


class TornadoClimateEntity(ClimateEntity):
    """Representation of a Tornado AC Climate device."""

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator: AuxCloudDataUpdateCoordinator,
        device: dict,
        *_: Any,  # Using *_ to ignore additional arguments like config_entry
    ) -> None:
        """Initialize the climate device."""
        super().__init__()
        self.hass = hass
        self._coordinator = coordinator
        self._client = coordinator.api
        self._device_id = device["endpointId"]
        self._attr_unique_id = f"{device['endpointId']}_climate"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, device["endpointId"])},
            "name": f"Tornado AC {device.get('friendlyName')}",
            "manufacturer": "Tornado",
            "model": "AUX Cloud",
        }

        self._attr_supported_features = (
            ClimateEntityFeature.TARGET_TEMPERATURE
            | ClimateEntityFeature.FAN_MODE
            | ClimateEntityFeature.SWING_MODE
            | ClimateEntityFeature.TURN_ON
            | ClimateEntityFeature.TURN_OFF
        )

        # Set available modes and temperature limits
        self._attr_hvac_modes = [*list(HVAC_MODE_MAP.values()), HVACMode.OFF]
        self._attr_hvac_mode = HVACMode.OFF
        self._attr_fan_modes = list(FAN_MODE_MAP.values())
        self._attr_fan_mode = FAN_MODE_MAP[0]
        self._attr_swing_modes = SWING_MODES
        self._attr_min_temp = 16
        self._attr_max_temp = 32

        # Initialize other attributes
        self._attr_current_temperature = None
        self._attr_target_temperature = None
        self._attr_temperature_unit = UnitOfTemperature.CELSIUS
        self._attr_swing_mode = None
        self._attr_hvac_action = HVACAction.OFF
        self._attr_available = False
        # Create entity description
        self.entity_description = ClimateEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')}",
            translation_key=DOMAIN,
        )

        # Add coordinator listener
        coordinator.async_add_listener(self._handle_coordinator_update)
        _LOGGER.info("Entity initialized for device %s", self._device_id)

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self._coordinator.last_update_success and self._device is not None

    @property
    def _device(self) -> dict | None:
        """Get current device data from coordinator."""
        if not self._coordinator.data:
            return None
        return self._coordinator.data.get(self._device_id)

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend."""
        return "mdi:air-conditioner"

    @property
    def device_info(self) -> dict:
        """Return device specific attributes."""
        return self._attr_device_info

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self._handle_coordinator_update()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from coordinator."""
        _LOGGER.debug(
            "Handling coordinator update for device %s with data: %s",
            self._device_id,
            self._device,
        )

        if not self._device:
            self._attr_available = False
            self.async_write_ha_state()
            return

        try:
            device_params = self._device.get("params", {})

            # Update power and HVAC mode/action
            if not device_params.get("pwr", 0):
                self._attr_hvac_mode = HVACMode.OFF
                self._attr_hvac_action = HVACAction.OFF
            else:
                self._attr_hvac_mode = HVAC_MODE_MAP.get(
                    device_params.get("ac_mode", 0), HVACMode.OFF
                )
                self._attr_hvac_action = {
                    HVACMode.COOL: HVACAction.COOLING,
                    HVACMode.HEAT: HVACAction.HEATING,
                    HVACMode.DRY: HVACAction.DRYING,
                    HVACMode.FAN_ONLY: HVACAction.FAN,
                    HVACMode.AUTO: HVACAction.IDLE,
                }.get(self._attr_hvac_mode, HVACAction.IDLE)

            # Update other attributes
            self._attr_fan_mode = FAN_MODE_MAP.get(
                device_params.get("ac_mark", 0), "auto"
            )
            self._attr_target_temperature = device_params.get("temp", 0) / 10
            self._attr_current_temperature = device_params.get("envtemp", 0) / 10
            # Update swing mode based on vertical and horizontal direction
            v_dir = device_params.get("ac_vdir", 0)
            h_dir = device_params.get("ac_hdir", 0)
            self._attr_swing_mode = {
                (0, 0): "off",
                (1, 0): "vertical",
                (0, 1): "horizontal",
                (1, 1): "both",
            }.get((v_dir, h_dir), "off")

            self._attr_available = True

            _LOGGER.debug(
                "Updated state for %s: mode=%s, action=%s, fan=%s, temp=%s, envtemp=%s",
                self._device_id,
                self._attr_hvac_mode,
                self._attr_hvac_action,
                self._attr_fan_mode,
                self._attr_target_temperature,
                self._attr_current_temperature,
            )

        except Exception:
            _LOGGER.exception("Error updating state for %s", self._device_id)
            self._attr_available = False

        self.async_write_ha_state()

    async def async_update(self) -> None:
        """Update the entity."""
        await self._coordinator.async_request_refresh()

    async def _set_device_params(self, params: dict) -> None:
        """Set device parameters and handle any errors."""
        try:
            await self._client.set_device_params(self._device, params)
            _LOGGER.info(
                "Connection pool stat - Size: %s, Acquired: %s, Active connections: %s",
                self._client.session.connector.size
                if self._client.session and self._client.session.connector
                else "N/A",
                self._client.session.connector.acquired
                if self._client.session and self._client.session.connector
                else "N/A",
                len(self._client.session.connector._acquired)
                if self._client.session and self._client.session.connector
                else "N/A",
            )
        except Exception:
            _LOGGER.exception(
                "Error setting parameters for %s",
                self._device.get("endpointId", "Unknown"),
            )

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temp := kwargs.get(ATTR_TEMPERATURE)) is None:
            _LOGGER.info(
                "No temperature provided for %s %s",
                self._device.get("endpointId"),
                kwargs,
            )
            return

        _LOGGER.info(
            "Setting temperature to %s for %s", temp, self._device.get("endpointId")
        )
        await self._set_device_params({"temp": int(temp * 10)})

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set new target hvac mode."""
        _LOGGER.info(
            "Setting HVAC mode to %s for %s",
            hvac_mode,
            self._device.get("endpointId", "Unknown"),
        )
        params = (
            {"pwr": 0}
            if hvac_mode == HVACMode.OFF
            else {"pwr": 1, "ac_mode": HVAC_MODE_MAP_REVERSE.get(hvac_mode, "auto")}
        )
        await self._set_device_params(params)

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        """Set new target fan mode."""
        _LOGGER.info(
            "Setting fan mode (ac_mark) to %s for %s",
            fan_mode,
            self._device.get("endpointId", "Unknown"),
        )
        await self._set_device_params(
            {"ac_mark": FAN_MODE_MAP_REVERSE.get(fan_mode, 1)}
        )

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        """Set new target swing mode."""
        device_id = self._device.get("endpointId", "Unknown")
        _LOGGER.info("Setting swing mode to %s for %s", swing_mode, device_id)
        params = {
            "ac_vdir": 1 if swing_mode in ["vertical", "both"] else 0,
            "ac_hdir": 1 if swing_mode in ["horizontal", "both"] else 0,
        }
        await self._set_device_params(params)

    async def async_turn_on(self) -> None:
        """Turn the device on."""
        _LOGGER.info("Turning on %s", self._device.get("endpointId", "Unknown"))
        try:
            await self._client.set_device_params(self._device, {"pwr": 1})
            _LOGGER.debug(
                "Connection pool stats- Size: %s, Acquired: %s, Active connections: %s",
                self._client.session.connector.size
                if self._client.session and self._client.session.connector
                else "N/A",
                self._client.session.connector.acquired
                if self._client.session and self._client.session.connector
                else "N/A",
                len(self._client.session.connector._acquired)
                if self._client.session and self._client.session.connector
                else "N/A",
            )
        except Exception:
            _LOGGER.exception(
                "Error turning on %s", self._device.get("endpointId", "Unknown")
            )

    async def async_turn_off(self) -> None:
        """Turn the device off."""
        _LOGGER.info("Turning off %s", self._device.get("endpointId", "Unknown"))
        try:
            await self._client.set_device_params(self._device, {"pwr": 0})
            _LOGGER.debug(
                "Connection pool stats- Size: %s, Acquired: %s, Active connections: %s",
                self._client.session.connector.size
                if self._client.session and self._client.session.connector
                else "N/A",
                self._client.session.connector.acquired
                if self._client.session and self._client.session.connector
                else "N/A",
                len(self._client.session.connector._acquired)
                if self._client.session and self._client.session.connector
                else "N/A",
            )
        except Exception:
            _LOGGER.exception(
                "Error turning off %s", self._device.get("endpointId", "Unknown")
            )


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Unload Tornado climate platform."""
    entry_data = hass.data[DOMAIN].get(config_entry.entry_id, {})
    client = entry_data.get("client")

    if client:
        await client.cleanup()
        # Cleanup shared resources when the last client is removed
        if len(hass.data[DOMAIN]) == 1:
            await AuxCloudAPI.cleanup_shared_resources()
        _LOGGER.info("Cleaned up AuxCloud client and resources")

    return True
