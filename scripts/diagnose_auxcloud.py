"""
Probe AuxCloud device requests without Home Assistant state handling.

This script mirrors the three per-device requests made by the integration and
reports whether each request returned a complete result, a partial result, or
an exception. Credentials are read from environment variables so they are not
printed or committed with the diagnostic output.
"""

# This is an interactive diagnostic script; its output is intentionally printed.
# ruff: noqa: INP001, T201

from __future__ import annotations

import argparse
import asyncio
import base64
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

configured_root = os.getenv("TORNADO_COMPONENT_ROOT")
REPO_ROOT = (
    Path(configured_root)
    if configured_root
    else Path(__file__).resolve().parents[1]
)


def load_auxcloud_module() -> ModuleType:
    """Load aux_cloud without executing the integration's Home Assistant init."""
    package_name = "_tornado_diagnostic_auxcloud"
    auxcloud_dir = REPO_ROOT / "custom_components" / "tornado" / "aux_cloud"
    package = ModuleType(package_name)
    package.__path__ = [str(auxcloud_dir)]
    sys.modules[package_name] = package

    util_name = f"{package_name}.util"
    util_spec = importlib.util.spec_from_file_location(
        util_name, auxcloud_dir / "util.py"
    )
    if util_spec is None or util_spec.loader is None:
        message = "unable to load AuxCloud utility module"
        raise RuntimeError(message)
    util_module = importlib.util.module_from_spec(util_spec)
    sys.modules[util_name] = util_module
    util_spec.loader.exec_module(util_module)

    auxcloud_name = f"{package_name}.client"
    auxcloud_spec = importlib.util.spec_from_file_location(
        auxcloud_name, auxcloud_dir / "__init__.py"
    )
    if auxcloud_spec is None or auxcloud_spec.loader is None:
        message = "unable to load AuxCloud client module"
        raise RuntimeError(message)
    auxcloud_module = importlib.util.module_from_spec(auxcloud_spec)
    sys.modules[auxcloud_name] = auxcloud_module
    auxcloud_spec.loader.exec_module(auxcloud_module)
    return auxcloud_module


AUXCLOUD = load_auxcloud_module()
AuxCloudAPI = AUXCLOUD.AuxCloudAPI
LICENSE = AUXCLOUD.LICENSE

REQUIRED_PARAMS = ("pwr", "ac_mode", "temp", "envtemp")


def describe_params(value: Any) -> str:
    """Describe parameter data without dumping cookies or credentials."""
    if not isinstance(value, dict):
        return f"type={type(value).__name__}"

    missing = [
        key for key in REQUIRED_PARAMS if key not in value or value[key] is None
    ]
    values = {key: value.get(key) for key in REQUIRED_PARAMS if key in value}
    result = f"type=dict keys={len(value)} values={values}"
    if missing:
        result += f" missing={missing}"
    return result


def describe_error(name: str, result: dict[str, Any]) -> str:
    """Describe a raw request exception."""
    error = result["error"]
    message = str(error).replace("\n", " ")
    return f"{name}: ERROR {type(error).__name__}: {message[:300]}"


def describe_raw_state(result: dict[str, Any]) -> str:
    """Describe the raw state response."""
    if not result["ok"]:
        return describe_error("state", result)
    payload = result["payload"]
    event_payload = payload.get("event", {}).get("payload", {})
    return (
        f"state: HTTP {result['http_status']} api_status={event_payload.get('status')} "
        f"data={event_payload.get('data')!r}"
    )


def describe_raw_params(
    name: str, result: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Describe raw parameter response and return parsed parameter values."""
    if not result["ok"]:
        return describe_error(name, result), {}

    payload = result["payload"]
    event_payload = payload.get("event", {}).get("payload", {})
    encoded_data = event_payload.get("data")
    if not isinstance(encoded_data, str):
        return (
            f"{name}: HTTP {result['http_status']} "
            f"api_status={event_payload.get('status')} missing event.payload.data",
            {},
        )

    try:
        data = json.loads(encoded_data)
    except json.JSONDecodeError as err:
        return f"{name}: invalid nested JSON: {err}", {}

    parameter_names = data.get("params")
    values = data.get("vals")
    if not isinstance(parameter_names, list) or not isinstance(values, list):
        return f"{name}: nested response missing params/vals: {data.keys()}", {}

    mapped_values: dict[str, Any] = {}
    for index, parameter_name in enumerate(parameter_names):
        try:
            mapped_values[parameter_name] = values[index][0]["val"]
        except (IndexError, KeyError, TypeError):
            mapped_values[parameter_name] = None

    return (
        f"{name}: HTTP {result['http_status']} "
        f"api_status={event_payload.get('status')} {describe_params(mapped_values)}",
        mapped_values,
    )


async def raw_query_state(
    api: AuxCloudAPI, device_id: str, dev_session: str
) -> dict[str, Any]:
    """Send the state request and retain the unmodified JSON response."""
    timestamp = int(time.time())
    data = {
        "directive": {
            "header": api._get_directive_header(
                namespace="DNA.QueryState",
                name="queryState",
                message_id_prefix=api.userid,
                timestamp=str(timestamp),
            ),
            "payload": {
                "studata": [{"did": device_id, "devSession": dev_session}],
                "msgtype": "batch",
            },
        }
    }
    try:
        session = await api._get_session()
        async with session.post(
            f"{api.url}/device/control/v2/querystate",
            data=json.dumps(data, separators=(",", ":")),
            headers=api._get_headers(),
        ) as response:
            body = await response.text()
            return {
                "ok": True,
                "http_status": response.status,
                "payload": json.loads(body),
            }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "error": err}


async def raw_get_params(
    api: AuxCloudAPI, device: dict[str, Any], requested: list[str]
) -> dict[str, Any]:
    """Send a parameter request and retain the unmodified JSON response."""
    cookie = json.loads(base64.b64decode(device["cookie"].encode()))
    mapped_cookie = base64.b64encode(
        json.dumps(
            {
                "device": {
                    "id": cookie["terminalid"],
                    "key": cookie["aeskey"],
                    "devSession": device["devSession"],
                    "aeskey": cookie["aeskey"],
                    "did": device["endpointId"],
                    "pid": device["productId"],
                    "mac": device["mac"],
                }
            },
            separators=(",", ":"),
        ).encode()
    ).decode()
    data = {
        "directive": {
            "header": api._get_directive_header(
                namespace="DNA.KeyValueControl",
                name="KeyValueControl",
                message_id_prefix=device["endpointId"],
            ),
            "endpoint": {
                "devicePairedInfo": {
                    "did": device["endpointId"],
                    "pid": device["productId"],
                    "mac": device["mac"],
                    "devicetypeflag": device["devicetypeFlag"],
                    "cookie": mapped_cookie,
                },
                "endpointId": device["endpointId"],
                "cookie": {},
                "devSession": device["devSession"],
            },
            "payload": {
                "act": "get",
                "params": requested,
                "vals": [],
            },
        }
    }
    if requested == ["mode"]:
        data["directive"]["payload"]["did"] = device["endpointId"]
        data["directive"]["payload"]["vals"] = [[{"val": 0, "idx": 1}]]

    try:
        session = await api._get_session()
        async with session.post(
            f"{api.url}/device/control/v2/sdkcontrol",
            params={"license": LICENSE},
            data=json.dumps(data, separators=(",", ":")),
            headers=api._get_headers(),
        ) as response:
            body = await response.text()
            return {
                "ok": True,
                "http_status": response.status,
                "payload": json.loads(body),
            }
    except Exception as err:  # noqa: BLE001
        return {"ok": False, "error": err}


async def fetch_raw_devices(
    api: AuxCloudAPI, family_id: str, *, shared: bool
) -> list[dict[str, Any]]:
    """Fetch endpoint metadata without the integration's per-device processing."""
    endpoint = (
        "dev/query?action=select"
        if not shared
        else "sharedev/querylist?querytype=shared"
    )
    session = await api._get_session()
    async with session.post(
        f"{api.url}/appsync/group/{endpoint}",
        data='{"pids":[]}' if not shared else '{"endpointId":""}',
        headers=api._get_headers(familyid=family_id),
    ) as response:
        body = await response.text()
        payload = json.loads(body)

    if payload.get("status") != 0:
        message = (
            f"device-list request failed: status={payload.get('status')} "
            f"message={payload.get('msg', body[:200])}"
        )
        raise RuntimeError(message)

    data = payload.get("data", {})
    if "endpoints" in data:
        return data["endpoints"]
    if "shareFromOther" in data:
        return [device["devinfo"] for device in data["shareFromOther"]]
    message = "device-list response did not contain endpoints"
    raise RuntimeError(message)


async def probe_device(api: AuxCloudAPI, device: dict[str, Any]) -> None:
    """Run the same three concurrent requests used by the integration."""
    device_id = device.get("endpointId", "unknown")
    print(f"  device={device_id} friendly_name={device.get('friendlyName')!r}")
    print(f"    raw endpoint params: {describe_params(device.get('params'))}")

    results = await asyncio.gather(
        raw_query_state(api, device_id, device["devSession"]),
        raw_get_params(api, device, []),
        raw_get_params(api, device, ["mode"]),
    )
    state_result, params_result, ambient_result = results

    print(f"    {describe_raw_state(state_result)}")
    params_description, params_values = describe_raw_params("params", params_result)
    ambient_description, ambient_values = describe_raw_params(
        "ambient", ambient_result
    )
    print(f"    {params_description}")
    print(f"    {ambient_description}")

    effective_params: dict[str, Any] = {}
    effective_params.update(params_values)
    if "envtemp" in ambient_values:
        effective_params["envtemp"] = ambient_values["envtemp"]

    missing = [
        key
        for key in REQUIRED_PARAMS
        if key not in effective_params or effective_params[key] is None
    ]
    if missing:
        print(
            "    classification: INCOMPLETE after integration merge; "
            f"missing={missing}"
        )
    else:
        print("    classification: COMPLETE after integration merge")


async def run(args: argparse.Namespace) -> None:
    """Run one or more independent AuxCloud probes."""
    api = AuxCloudAPI(args.email, args.password, region=args.region)
    try:
        await api.login()
        print(f"Logged in to {api.url} as {args.email}")

        for poll_number in range(1, args.repeat + 1):
            print(f"\n=== poll {poll_number}/{args.repeat} ===")
            try:
                families = await api.list_families()
            except Exception as err:  # noqa: BLE001
                print(f"family list: ERROR {type(err).__name__}: {err}")
                if poll_number != args.repeat:
                    await asyncio.sleep(args.interval)
                continue

            print(f"families: {len(families)}")
            seen: set[str] = set()
            for family in families:
                family_id = family["familyid"]
                for shared in (False, True) if args.include_shared else (False,):
                    try:
                        devices = await fetch_raw_devices(
                            api, family_id, shared=shared
                        )
                    except Exception as err:  # noqa: BLE001
                        kind = "shared" if shared else "regular"
                        print(
                            f"family={family_id} {kind} device list: "
                            f"ERROR {type(err).__name__}: {err}"
                        )
                        continue

                    for device in devices:
                        device_id = device.get("endpointId")
                        if device_id in seen:
                            continue
                        seen.add(device_id)
                        await probe_device(api, device)

            if poll_number != args.repeat:
                await asyncio.sleep(args.interval)
    finally:
        await AuxCloudAPI.cleanup_shared_resources()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and environment-backed credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--email",
        default=os.getenv("TORNADO_EMAIL"),
        help="AuxCloud email (defaults to TORNADO_EMAIL)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("TORNADO_PASSWORD"),
        help="AuxCloud password (prefer TORNADO_PASSWORD)",
    )
    parser.add_argument("--region", choices=("eu", "usa"), default="eu")
    parser.add_argument("--repeat", type=int, default=1, help="Number of polls")
    parser.add_argument(
        "--interval", type=float, default=65, help="Seconds between polls"
    )
    parser.add_argument(
        "--include-shared", action="store_true", help="Probe shared devices too"
    )
    args = parser.parse_args()
    if not args.email or not args.password:
        parser.error("set TORNADO_EMAIL and TORNADO_PASSWORD")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
