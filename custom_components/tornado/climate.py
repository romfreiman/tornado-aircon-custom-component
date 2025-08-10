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
from homeassistant.const import ATTR_TEMPERATURE, UnitOfTemperature
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .aux_cloud import AuxCloudAPI
from .const import DOMAIN

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

_LOGGER = logging.getLogger(__name__)

HVAC_MODE_MAP = {
    0: HVACMode.COOL,
    1: HVACMode.HEAT,
    2: HVACMode.DRY,
    3: HVACMode.FAN_ONLY,
    4: HVACMode.AUTO,
}
HVAC_MODE_MAP_REVERSE = {v: k for k, v in HVAC_MODE_MAP.items()}

FAN_MODE_MAP = {
    0: "auto",
    1: "low",
    2: "medium",
    3: "high",
    4: "turbo",
    5: "silent",
}
FAN_MODE_MAP_REVERSE = {v: k for k, v in FAN_MODE_MAP.items()}

SWING_MODES = ["off", "vertical", "horizontal", "both"]


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Tornado climate platform."""
    entry_data = hass.data[DOMAIN][config_entry.entry_id]
    client = entry_data["client"]

    if not hasattr(client, "loginsession") or not client.loginsession:
        _LOGGER.info("Initial login for AuxCloud client")
        await client.login()

    coordinator = AuxCloudDataUpdateCoordinator(hass, client)
    await coordinator.async_refresh()

    hass.data[DOMAIN][config_entry.entry_id]["coordinator"] = coordinator

    try:
        devices = await client.get_devices()
        entities = []

        for device in devices:
            try:
                entities.append(TornadoClimateEntity(hass, coordinator, device))
            except Exception:
                _LOGGER.exception(
                    "Error setting up device %s", device.get("endpointId")
                )

        async_add_entities(entities)
        _LOGGER.debug("Tornado entities added: %s", [e.name for e in entities])

    except Exception:
        _LOGGER.exception("Error setting up Tornado climate platform")


class AuxCloudDataUpdateCoordinator(DataUpdateCoordinator):
    """Coordinator for AUX Cloud data."""

    def __init__(self, hass: HomeAssistant, api: AuxCloudAPI) -> None:
        self.api = api
        super().__init__(
            hass,
            _LOGGER,
            name="AuxCloud",
            update_interval=timedelta(minutes=1),
        )

    async def _async_update_data(self) -> dict:
        try:
            if not hasattr(self.api, "loginsession") or not self.api.loginsession:
                _LOGGER.info("No valid login session, attempting to login")
                await self.api.login()

            devices = await self.api.get_devices()
            _LOGGER.debug("Coordinator fetched data: %s", devices)
            return {device["endpointId"]: device for device in devices}
        except Exception as err:
            _LOGGER.exception("Error fetching data")
            raise UpdateFailed(f"Error fetching data: {err}") from err


class TornadoClimateEntity(ClimateEntity):
    """AUX Cloud AC as a ClimateEntity."""

    def __init__(self, hass, coordinator, device):
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
        self._attr_hvac_modes = list(HVAC_MODE_MAP.values()) + [HVACMode.OFF]
        self._attr_fan_modes = list(FAN_MODE_MAP.values())
        self._attr_swing_modes = SWING_MODES
        self._attr_min_temp = 16
        self._attr_max_temp = 32
        self._attr_temperature_unit = UnitOfTemperature.CELSIUS

        self._attr_current_temperature = None
        self._attr_target_temperature = None
        self._attr_hvac_mode = HVACMode.OFF
        self._attr_fan_mode = FAN_MODE_MAP[0]
        self._attr_swing_mode = None
        self._attr_hvac_action = HVACAction.OFF
        self._attr_available = False

        self.entity_description = ClimateEntityDescription(
            key=self._attr_unique_id,
            name=f"Tornado AC {device.get('friendlyName')}",
            translation_key=DOMAIN,
        )

        _LOGGER.info("Entity initialized for device %s", self._device_id)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._unsub_update_listener = self._coordinator.async_add_listener(
            self._handle_coordinator_update
        )
        self._handle_coordinator_update()

    async def async_will_remove_from_hass(self) -> None:
        if hasattr(self, "_unsub_update_listener") and self._unsub_update_listener:
            self._unsub_update_listener()

    @callback
    def _handle_coordinator_update(self) -> None:
        device = self._device
        if not device:
            self._attr_available = False
            if self.hass:
                self.async_write_ha_state()
            return

        try:
            params = device.get("params", {})

            if not params.get("pwr", 0):
                self._attr_hvac_mode = HVACMode.OFF
                self._attr_hvac_action = HVACAction.OFF
            else:
                self._attr_hvac_mode = HVAC_MODE_MAP.get(
                    params.get("ac_mode", 0), HVACMode.OFF
                )
                self._attr_hvac_action = {
                    HVACMode.COOL: HVACAction.COOLING,
                    HVACMode.HEAT: HVACAction.HEATING,
                    HVACMode.DRY: HVACAction.DRYING,
                    HVACMode.FAN_ONLY: HVACAction.FAN,
                    HVACMode.AUTO: HVACAction.IDLE,
                }.get(self._attr_hvac_mode, HVACAction.IDLE)

            self._attr_fan_mode = FAN_MODE_MAP.get(params.get("ac_mark", 0), "auto")
            self._attr_target_temperature = params.get("temp", 0) / 10
            self._attr_current_temperature = params.get("envtemp", 0) / 10

            v_dir = params.get("ac_vdir", 0)
            h_dir = params.get("ac_hdir", 0)
            self._attr_swing_mode = {
                (0, 0): "off",
                (1, 0): "vertical",
                (0, 1): "horizontal",
                (1, 1): "both",
            }.get((v_dir, h_dir), "off")

            self._attr_available = True
        except Exception:
            _LOGGER.exception("Error updating state for %s", self._device_id)
            self._attr_available = False

        if self.hass:
            self.async_write_ha_state()

    @property
    def available(self) -> bool:
        return self._coordinator.last_update_success and self._device is not None

    @property
    def _device(self) -> dict | None:
        return (
            self._coordinator.data.get(self._device_id)
            if self._coordinator.data
            else None
        )

    @property
    def icon(self) -> str:
        return "mdi:air-conditioner"

    @property
    def device_info(self) -> dict:
        return self._attr_device_info

    async def async_update(self) -> None:
        await self._coordinator.async_request_refresh()

    async def _set_device_params(self, params: dict) -> None:
        try:
            await self._client.set_device_params(self._device, params)
        except Exception:
            _LOGGER.exception(
                "Error setting parameters for %s",
                self._device.get("endpointId", "Unknown"),
            )

    async def async_set_temperature(self, **kwargs: Any) -> None:
        if (temp := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        await self._set_device_params({"temp": int(temp * 10)})

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        params = (
            {"pwr": 0}
            if hvac_mode == HVACMode.OFF
            else {"pwr": 1, "ac_mode": HVAC_MODE_MAP_REVERSE.get(hvac_mode, 0)}
        )
        await self._set_device_params(params)

    async def async_set_fan_mode(self, fan_mode: str) -> None:
        await self._set_device_params(
            {"ac_mark": FAN_MODE_MAP_REVERSE.get(fan_mode, 1)}
        )

    async def async_set_swing_mode(self, swing_mode: str) -> None:
        params = {
            "ac_vdir": 1 if swing_mode in ["vertical", "both"] else 0,
            "ac_hdir": 1 if swing_mode in ["horizontal", "both"] else 0,
        }
        await self._set_device_params(params)

    async def async_turn_on(self) -> None:
        await self._client.set_device_params(self._device, {"pwr": 1})

    async def async_turn_off(self) -> None:
        await self._client.set_device_params(self._device, {"pwr": 0})


async def async_unload_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    entry_data = hass.data[DOMAIN].get(config_entry.entry_id, {})
    client = entry_data.get("client")

    if client:
        await client.cleanup()
        if len(hass.data[DOMAIN]) == 1:
            await AuxCloudAPI.cleanup_shared_resources()
        _LOGGER.info("Cleaned up AuxCloud client and resources")

    return True
