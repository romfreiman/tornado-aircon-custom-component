# __init__.py
"""The AUX AC integration."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING

import aiohttp
from homeassistant.const import Platform
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from datetime import timedelta

from .aux_cloud import AuxCloudAPI
from .const import CONF_EMAIL, CONF_PASSWORD, CONF_REGION, DOMAIN

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

PLATFORMS: list[Platform] = [Platform.CLIMATE, Platform.SENSOR, Platform.SWITCH, Platform.SELECT]
_LOGGER = logging.getLogger(__name__)


class AuxCloudDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching AuxCloud data."""

    def __init__(self, hass: HomeAssistant, api: AuxCloudAPI) -> None:
        """Initialize the coordinator."""
        self.api = api
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=30),
        )
        _LOGGER.info("Adding websocket listener that triggers coordinator update")
        # Add websocket listener that triggers coordinator update
        self.api.ws_api.add_websocket_listener(self._handle_websocket_message)

    async def _handle_websocket_message(self, message: dict) -> None:
        """Handle incoming websocket message and update device state."""
        _LOGGER.debug("Received websocket message: %s", message)
        
        # Check if this is a device push message with state data
        if (message.get("msgtype") == "push" and 
            message.get("topic") == "devpush" and 
            "data" in message and 
            "endpointId" in message["data"]):
            
            try:
                endpoint_id = message["data"]["endpointId"]
                
                # Decode the base64 data to get device parameters
                if "data" in message["data"]:
                    encoded_data = message["data"]["data"]
                    decoded_bytes = base64.b64decode(encoded_data)
                    decoded_str = decoded_bytes.decode('utf-8')
                    device_params = json.loads(decoded_str)
                    
                    _LOGGER.debug("Decoded device params for %s: %s", endpoint_id, device_params)
                    
                    # Update the coordinator's data directly
                    if self.data and endpoint_id in self.data:
                        device = self.data[endpoint_id]
                        
                        # Update device params with new data
                        if "params" not in device:
                            device["params"] = {}
                        
                        # Update device parameters directly - websocket uses same param names as integration
                        for param, value in device_params.items():
                            device["params"][param] = value
                        
                        _LOGGER.debug("Updated device %s params: %s", endpoint_id, device["params"])
                        
                        # Trigger entity updates for this specific device
                        self.async_update_listeners()
                        return
                    
                    _LOGGER.warning("Device %s not found in coordinator data", endpoint_id)
                    
            except Exception as ex:
                _LOGGER.error("Error processing websocket message: %s", ex)
        
        # For other message types or if processing fails, fall back to full refresh
        _LOGGER.debug("Triggering full coordinator refresh")
        await self.async_request_refresh()

    async def _async_update_data(self):
        """Fetch data from API endpoint."""
        try:
            _LOGGER.debug("Coordinator fetching fresh data from API")
            devices = await self.api.get_devices()
            data = {}
            for device in devices:
                data[device["endpointId"]] = device
            _LOGGER.debug("Coordinator updated data for %d devices", len(data))
            return data
        except Exception as err:
            _LOGGER.error("Error communicating with API: %s", err)
            raise UpdateFailed(f"Error communicating with API: {err}") from err


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AUX AC from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(entry.entry_id, {})

    # Create a shared session for the entry
    session = aiohttp.ClientSession()
    hass.data[DOMAIN][entry.entry_id]["session"] = session

    client = AuxCloudAPI(
        email=entry.data[CONF_EMAIL],
        password=entry.data[CONF_PASSWORD],
        region=entry.data[CONF_REGION],
        session=session,
    )

    try:
        await client.login()
        await client.initialize_websocket()
        await client.refresh()
    except Exception:
        await client.cleanup()
        await session.close()
        _LOGGER.exception("Failed to connect to AUX AC")
        return False

    # Create coordinator HERE in __init__.py so all platforms can access it
    coordinator = AuxCloudDataUpdateCoordinator(hass, client)
    await coordinator.async_config_entry_first_refresh()

    # Store both client and coordinator in entry data
    hass.data[DOMAIN][entry.entry_id]["client"] = client
    hass.data[DOMAIN][entry.entry_id]["coordinator"] = coordinator
    
    _LOGGER.info("Coordinator created and stored in __init__.py for all platforms")

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        entry_data = hass.data[DOMAIN].get(entry.entry_id, {})
        client = entry_data.get("client")
        session = entry_data.get("session")

        if client:
            await client.cleanup()
        if session and not session.closed:
            await session.close()

        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
