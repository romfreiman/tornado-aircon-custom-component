"""Utility functions for Tornado integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


def find_entity_by_unique_id(hass: HomeAssistant, unique_id: str, required_attr: str | None = None):
    """Helper function to find an entity by its unique_id.
    
    Args:
        hass: Home Assistant instance
        unique_id: The unique_id to search for
        required_attr: Optional attribute that the entity must have
        
    Returns:
        The entity if found, None otherwise
    """
    for component in hass.data.get("entity_components", {}).values():
        if hasattr(component, "entities"):
            for entity in component.entities:
                if hasattr(entity, "unique_id") and entity.unique_id == unique_id:
                    if required_attr is None or hasattr(entity, required_attr):
                        return entity
    return None
