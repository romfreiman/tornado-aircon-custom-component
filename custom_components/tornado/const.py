# custom_components/tornado/const.py
"""Constants for the Tornado AC integration."""

DOMAIN = "tornado"
CONF_EMAIL = "email"
# Configuration key for password field - not the actual password value
CONF_PASSWORD = "password"  # noqa: S105
CONF_REGION = "region"

# Supported regions
REGIONS = ["eu", "usa"]

# Duration options and mappings
OFF_TIMER_DURATION_MAP = {
    "10m": 10,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "3h": 180,
    "4h": 240,
    "5h": 300,
    "6h": 360,
    "7h": 420,
    "8h": 480,
    "9h": 540,
    "10h": 600,
    "11h": 660,
    "12h": 720
}

COOLDOWN_DURATION_MAP = {
    "1min": 1,
    "3min": 3,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "30min": 30
}
