"""Config flow for Tornado AC integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_EMAIL, CONF_PASSWORD
from homeassistant.core import callback

from .aux_cloud import AuxCloudAPI
from .const import CONF_REGION, DOMAIN, REGIONS

if TYPE_CHECKING:
    from homeassistant.data_entry_flow import FlowResult

_LOGGER = logging.getLogger(__name__)


class TornadoConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Tornado AC."""

    VERSION = 1

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
        config_entry: config_entries.ConfigEntry | None = None,
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        if config_entry:
            self._config_entry = config_entry

        if user_input is not None:
            try:
                client = AuxCloudAPI(
                    email=user_input[CONF_EMAIL],
                    password=user_input[CONF_PASSWORD],
                    region=user_input[CONF_REGION],
                )
                await client.login()
                await client.initialize_websocket()

                await self.async_set_unique_id(user_input[CONF_EMAIL])
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=user_input[CONF_EMAIL], data=user_input
                )
            except Exception:
                _LOGGER.exception("Failed to connect to Tornado AC")
                errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_EMAIL): str,
                    vol.Required(CONF_PASSWORD): str,
                    vol.Required(CONF_REGION, default="eu"): vol.In(REGIONS),
                }
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> TornadoOptionsFlow:
        """Get the options flow for this handler."""
        return TornadoOptionsFlow(config_entry)


class TornadoOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Tornado AC."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage options."""
        errors = {}

        if user_input is not None:
            try:
                # Test the new configuration
                client = AuxCloudAPI(
                    email=self.config_entry.data[CONF_EMAIL],
                    password=self.config_entry.data[CONF_PASSWORD],
                    region=user_input[CONF_REGION],
                )
                await client.login()
                await client.initialize_websocket()
                
                return self.async_create_entry(title="", data=user_input)
            except Exception:
                _LOGGER.exception("Failed to connect to Tornado AC")
                errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_REGION,
                        default=self.config_entry.data.get(CONF_REGION, "eu"),
                    ): vol.In(REGIONS),
                }
            ),
            errors=errors,
        )
