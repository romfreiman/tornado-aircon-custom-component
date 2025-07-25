# Tornado Aircon Custom Component for Home Assistant

## Description

This custom component integrates Tornado Aircon devices with Home Assistant, allowing you to control and monitor your air conditioning units directly from the Home Assistant interface.

## Installation

### Option 1: Manual Installation

1. Download the `custom_components` folder from this repository.
2. Copy the `custom_components/tornado_aircon` directory into your Home Assistant `config/custom_components` directory.
3. Restart Home Assistant.

### Option 2: Installation via HACS

1. Ensure you have [HACS](https://hacs.xyz/) installed in your Home Assistant setup.
2. Navigate to **HACS** → **Integrations**.
3. Click the three dots menu in the top right corner and select **Custom repositories**.
4. Add the repository URL `https://github.com/romfreiman/tornado-aircon-custom-component` and select the category as **Integration**.
5. Find and install the "Tornado Air Conditioner" integration from the HACS store.
6. Restart Home Assistant.

## Configuration

To set up the Tornado Air Conditioner integration in Home Assistant:

1. Navigate to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for "Tornado Air Conditioner"
4. In the configuration screen, enter:
   - Your Tornado app email address
   - Your Tornado app password
   - Region: Select USA (Note: Verified working with Israel-based deployments)
5. Click **Submit** to complete the setup

## Features

- **Climate Control**: Full control of power, mode, temperature, and fan speed
- **Real-time Monitoring**: Current temperature, humidity, and operational status
- **Advanced Timer System**: Set timers with multiple actions (turn off, sleep mode)
- **🆕 CoolDown Mode**: Instant cooling relief with automatic transition to quiet operation
- **Sleep Mode**: Direct control of quiet operation mode
- **Smart Automations**: Perfect integration with Home Assistant automations and scripts

### CoolDown Feature
The new CoolDown feature provides instant cooling relief:
- **One-button activation** - Even if AC is off, starts cooling immediately
- **Maximum cooling power** - Uses Turbo fan speed in Cool mode
- **Configurable duration** - 1-30 minutes (default: 5 minutes)
- **Smart transition** - Automatically switches to your preferred fan mode (default: Silent)
- **Safety mechanisms** - Cancels if AC is turned off or settings changed manually

See [COOLDOWN_USAGE.md](COOLDOWN_USAGE.md) for detailed usage instructions.

### Timer System
Advanced software-based timer functionality:
- **Flexible durations** - Up to 8 hours
- **Multiple actions** - Turn off or enable sleep mode
- **Smart cancellation** - Automatically cancels when AC is turned off

See [TIMER_USAGE.md](TIMER_USAGE.md) for detailed usage instructions.

## Usage

Once configured, you will see new entities in Home Assistant for each Tornado Aircon unit:

### Core Entities
- **Climate Entity**: `climate.tornado_[device_id]` - Main AC control
- **Timer Sensor**: `sensor.tornado_[device_id]_timer` - Timer status and remaining time
- **Sleep Mode Switch**: `switch.tornado_[device_id]_sleep_mode` - Direct sleep mode control
- **🆕 CoolDown Switch**: `switch.tornado_[device_id]_cooldown` - One-button cooling relief

### Configuration Entities  
- **Timer Duration**: `number.tornado_[device_id]_timer_duration` - Set timer duration (0-480 min)
- **🆕 CoolDown Duration**: `number.tornado_[device_id]_cooldown_duration` - Set cooldown duration (1-30 min)
- **🆕 CoolDown Target Fan**: `number.tornado_[device_id]_cooldown_target_fan` - Set fan mode after cooldown

### Services
- `tornado.set_timer` - Set timer with custom duration and action
- `tornado.cancel_timer` - Cancel active timer
- **🆕** `tornado.start_cooldown` - Start cooldown with custom settings
- **🆕** `tornado.cancel_cooldown` - Cancel active cooldown

All entities can be used in automations, scripts, and dashboards for complete smart home integration.

## Troubleshooting

If you encounter any issues, please check the Home Assistant logs for error messages. You can also open an issue on the [GitHub repository](https://github.com/romfreiman/tornado-aircon-custom-component/issues).

## Contributing

Contributions are welcome! Please open a pull request with your changes. Make sure to follow the [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Resources

- [Home Assistant Documentation](https://www.home-assistant.io/docs/)
- [Home Assistant Community Forum](https://community.home-assistant.io/)
- [HACS Documentation](https://hacs.xyz/docs/)

## Acknowledgements

Special shoutout to [@maeek](https://github.com/maeek) for their great work on [ha-aux-cloud](https://github.com/maeek/ha-aux-cloud) as a baseline for this Home Assistant component.
Also, thanks to [@thewh1teagle](https://github.com/thewh1teagle) for their excellent work on [tornado-control](https://github.com/thewh1teagle/tornado-control) which inspired this component.

## TODO

- Add a custom integration icon.
