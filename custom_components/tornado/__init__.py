# __init__.py
"""The AUX AC integration."""
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_EMAIL, CONF_PASSWORD, CONF_REGION
# Updated import name
from .aux_cloud import AuxCloudAPI

PLATFORMS = [Platform.CLIMATE]
_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up AUX AC from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    client = AuxCloudAPI(
        email=entry.data[CONF_EMAIL],
        password=entry.data[CONF_PASSWORD],
        region=entry.data[CONF_REGION]
    )
    
    try:
        await client.login()
        await client.refresh()
    except Exception as ex:
        _LOGGER.error("Failed to connect to AUX AC: %s", str(ex))
        return False
    
    hass.data[DOMAIN][entry.entry_id] = client
    
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok
