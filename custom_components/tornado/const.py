# custom_components/tornado-control/const.py
"""Constants for the Tornado AC integration."""
DOMAIN = "tornado-control"  # Changed from aux_ac to match the folder name
CONF_EMAIL = "email"
# Configuration key for password field - not the actual password value
CONF_PASSWORD = "password"  # noqa: S105
CONF_REGION = "region"

# Supported regions
REGIONS = ["eu", "usa"]
