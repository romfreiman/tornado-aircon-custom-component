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

- Control power, mode, temperature, and fan speed of your Tornado Aircon units.
- Monitor current temperature, humidity, and operational status.
- Automate your air conditioning based on Home Assistant automations.

## Usage

Once configured, you will see new entities in Home Assistant for each Tornado Aircon unit. You can use these entities in automations, scripts, and dashboards.

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
