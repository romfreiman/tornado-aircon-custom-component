"""AuxCloud API client for Tornado AC."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import socket
import time
from typing import TYPE_CHECKING, Any, ClassVar

import aiohttp
from async_lru import alru_cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from collections.abc import Callable

from .util import encrypt_aes_cbc_zero_padding

_LOGGER = logging.getLogger(__name__)

# Constants from original implementation
TIMESTAMP_TOKEN_ENCRYPT_KEY = "kdixkdqp54545^#*"  # noqa: S105
PASSWORD_ENCRYPT_KEY = "4969fj#k23#"  # noqa: S105
BODY_ENCRYPT_KEY = "xgx3d*fe3478$ukx"
AES_INITIAL_VECTOR = bytes(
    [
        (b + 256) % 256
        for b in [
            -22,
            -86,
            -86,
            58,
            -69,
            88,
            98,
            -94,
            25,
            24,
            -75,
            119,
            29,
            22,
            21,
            -86,
        ]
    ]
)
LICENSE = (
    "PAFbJJ3WbvDxH5vvWezXN5BujETtH/iuTtIIW5CE/SeHN7oNKqnEajgljTcL0fBQQWM0XAAAAAAnBh"
    "JyhMi7zIQMsUcwR/PEwGA3uB5HLOnr+xRrci+FwHMkUtK7v4yo0ZHa+jPvb6djelPP893k7SagmffZ"
    "mOkLSOsbNs8CAqsu8HuIDs2mDQAAAAA="
)
LICENSE_ID = "3c015b249dd66ef0f11f9bef59ecd737"
COMPANY_ID = "48eb1b36cf0202ab2ef07b880ecda60d"
SPOOF_APP_VERSION = "2.2.10.456537160"
SPOOF_USER_AGENT = "Dalvik/2.1.0 (Linux; U; Android 12; SM-G991B Build/SP1A.210812.016)"
SPOOF_SYSTEM = "android"
SPOOF_APP_PLATFORM = "android"
API_SERVER_URL_EU = "https://app-service-deu-f0e9ebbb.smarthomecs.de"
API_SERVER_URL_USA = "https://app-service-usa-fd7cc04c.smarthomecs.com"


class AuxCloudError(Exception):
    """Base exception for AuxCloud API."""


class AuxCloudAuthError(AuxCloudError):
    """Authentication error occurred."""


class AuxCloudApiError(AuxCloudError):
    """API error occurred."""


class AuxCloudConnectionError(AuxCloudError):
    """Connection error occurred."""


# Add retry decorator configuration
def create_retry_decorator(
    max_attempts: int = 3,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Create a retry decorator with specified parameters."""
    return retry(
        retry=retry_if_exception_type(
            (
                aiohttp.ClientError,
                TimeoutError,
                AuxCloudConnectionError,
                AuxCloudApiError,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(max_attempts),
        before_sleep=lambda retry_state: _LOGGER.warning(
            "Request failed, retrying in %s seconds...",
            retry_state.next_action.sleep,
        ),
    )


class AuxCloudAPI:
    """API Client for AuxCloud."""

    LOGIN_VALIDATION_FAILED = -30129

    _shared_connector: ClassVar[aiohttp.TCPConnector | None] = None
    _shared_connector_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _shared_session: ClassVar[aiohttp.ClientSession | None] = None
    _shared_session_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_shared_connector(cls) -> aiohttp.TCPConnector:
        """Get or create shared connector."""
        async with cls._shared_connector_lock:
            if cls._shared_connector is None or cls._shared_connector.closed:
                cls._shared_connector = aiohttp.TCPConnector(
                    limit=10,
                    ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                    use_dns_cache=True,
                    family=socket.AF_INET,
                    keepalive_timeout=120,
                    force_close=False,
                )
                _LOGGER.info("Created new connector: %s", id(cls._shared_connector))

            if cls._shared_connector and hasattr(cls._shared_connector, "_conns"):
                _LOGGER.info(
                    "Connector status - Active connections: %d, Acquired: %d",
                    len(cls._shared_connector._conns),
                    len(cls._shared_connector._acquired),
                )
            return cls._shared_connector

    @classmethod
    async def get_shared_session(cls) -> aiohttp.ClientSession:
        """Get or create shared session."""
        async with cls._shared_session_lock:
            if cls._shared_session is None or cls._shared_session.closed:
                connector = await cls.get_shared_connector()
                cls._shared_session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(
                        total=30, connect=10, sock_connect=10, sock_read=10
                    ),
                    raise_for_status=True,
                )
                _LOGGER.info(
                    "Created new shared aiohttp session: %s", id(cls._shared_session)
                )
            return cls._shared_session

    def __init__(
        self,
        email: str,
        password: str,
        session: aiohttp.ClientSession | None = None,
        region: str = "eu",
    ) -> None:
        """Initialize the API client."""
        self.url = API_SERVER_URL_EU if region == "eu" else API_SERVER_URL_USA
        self.email = email
        self.password = password
        self.session = session
        # If session is provided externally, we don't own it
        self._session_owner = session is None
        self.data: dict[str, Any] = {}
        self.timeout = aiohttp.ClientTimeout(
            total=30, connect=10, sock_connect=10, sock_read=10
        )
        _LOGGER.info(
            "Initialized AuxCloudAPI with email: %s, region: %s", email, region
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get aiohttp client session."""
        if self._session_owner:
            if self.session is None or self.session.closed:
                if getattr(self, "_cleaned_up", False):
                    msg = "Cannot create new session after cleanup"
                    raise RuntimeError(msg)
                self.session = await self.get_shared_session()
                _LOGGER.debug("Using shared session: %s", id(self.session))
            return self.session

        return await self.get_shared_session()

    async def cleanup(self) -> None:
        """Clean up resources."""
        # We never close the session in instance cleanup
        # External sessions are managed by their owners
        # Shared sessions are managed by cleanup_shared_resources

        # Just clear the reference and mark as cleaned up
        self.session = None
        self._cleaned_up = True

    @classmethod
    async def cleanup_shared_resources(cls) -> None:
        """Cleanup shared resources."""
        # First cleanup session since it depends on the connector
        session = None
        async with cls._shared_session_lock:
            if cls._shared_session and not cls._shared_session.closed:
                session = cls._shared_session
                cls._shared_session = None

        if session:
            await session.close()

        # Then cleanup connector
        connector = None
        async with cls._shared_connector_lock:
            if cls._shared_connector and not cls._shared_connector.closed:
                connector = cls._shared_connector
                cls._shared_connector = None

        if connector:
            await connector.close()

        # Ensure the shared resources are fully cleaned up
        async with cls._shared_session_lock:
            cls._shared_session = None

        async with cls._shared_connector_lock:
            cls._shared_connector = None

    def _get_headers(self, **kwargs: str) -> dict[str, str]:
        """
        Get request headers.

        Args:
            **kwargs: Additional header key-value pairs

        Returns:
            Dictionary of headers

        """
        headers = {
            "Content-Type": "application/x-java-serialized-object",
            "licenseId": LICENSE_ID,
            "lid": LICENSE_ID,
            "language": "en",
            "appVersion": SPOOF_APP_VERSION,
            "User-Agent": SPOOF_USER_AGENT,
            "system": SPOOF_SYSTEM,
            "appPlatform": SPOOF_APP_PLATFORM,
            "loginsession": getattr(self, "loginsession", ""),
            "userid": getattr(self, "userid", ""),
            **kwargs,
        }
        _LOGGER.debug("Generated headers: %s", headers)
        return headers

    def _get_directive_header(
        self, namespace: str, name: str, message_id_prefix: str, **kwargs: str
    ) -> dict[str, str]:
        """
        Get directive header for device control.

        Args:
            namespace: Directive namespace
            name: Directive name
            message_id_prefix: Prefix for message ID
            **kwargs: Additional header key-value pairs

        Returns:
            Dictionary containing directive header

        """
        timestamp = int(time.time())
        header = {
            "namespace": namespace,
            "name": name,
            "interfaceVersion": "2",
            "senderId": "sdk",
            "messageId": f"{message_id_prefix}-{timestamp}",
            **kwargs,
        }
        _LOGGER.debug("Generated directive header: %s", header)
        return header

    async def login(
        self, email: str | None = None, password: str | None = None
    ) -> bool:
        """Login to AuxCloud."""
        try:
            success = await self._perform_login(email, password)
        except (aiohttp.ClientError, TimeoutError) as ex:
            self._log_and_raise_auth_error(ex)
            return False  # Unreachable, but satisfies type checker
        except json.JSONDecodeError as ex:
            self._log_and_raise_auth_error(ex)
            return False  # Unreachable, but satisfies type checker

        if not success:
            msg = "Login failed: Invalid credentials"
            raise AuxCloudAuthError(msg)
        # Wait a brief moment after successful login before making other requests
        await asyncio.sleep(1)
        return success

    def _log_and_raise_auth_error(self, ex: Exception) -> None:
        """Log the exception and raises an AuxCloudAuthError."""
        _LOGGER.exception("Login error: %s", str(ex))
        msg = f"Login failed: {ex!s}"
        raise AuxCloudAuthError(msg) from ex

    @create_retry_decorator()
    async def _perform_login(
        self, email: str | None = None, password: str | None = None
    ) -> bool:
        """Perform the actual login operation with retry."""
        email = email if email is not None else self.email
        password = password if password is not None else self.password
        _LOGGER.info("Attempting to login with email: %s", email)

        current_time = time.time()
        # Note: SHA1 is used here to match the original API implementation
        sha_password = hashlib.sha1(  # noqa: S324
            f"{password}{PASSWORD_ENCRYPT_KEY}".encode()
        ).hexdigest()
        payload = {
            "email": email,
            "password": sha_password,
            "companyid": COMPANY_ID,
            "lid": LICENSE_ID,
        }
        json_payload = json.dumps(payload, separators=(",", ":"))
        # Note: MD5 is used here to match the original API implementation
        token = hashlib.md5(  # noqa: S324
            f"{json_payload}{BODY_ENCRYPT_KEY}".encode()
        ).hexdigest()
        md5_hash = hashlib.md5(  # noqa: S324
            f"{current_time}{TIMESTAMP_TOKEN_ENCRYPT_KEY}".encode()
        ).digest()

        session = await self._get_session()
        async with session.post(
            f"{self.url}/account/login",
            data=encrypt_aes_cbc_zero_padding(
                AES_INITIAL_VECTOR, md5_hash, json_payload.encode()
            ),
            headers=self._get_headers(timestamp=f"{current_time}", token=token),
        ) as resp:
            data = await resp.text()
            json_data = json.loads(data)

            if "status" in json_data and json_data["status"] == 0:
                self.loginsession = json_data["loginsession"]
                self.userid = json_data["userid"]
                _LOGGER.info("Login successful for email: %s", email)
                return True

            error_msg = f"Login failed: {json_data.get('msg', data)}"
            raise AuxCloudAuthError(error_msg)

    async def get_devices(self) -> list[dict[str, Any]]:
        """Get all devices across all families."""
        _LOGGER.debug("Fetching all devices")
        all_devices = []
        try:
            # First get all families
            families = await self.list_families()
            _LOGGER.debug("Fetched families: %s", families)

            # Then get devices for each family
            for family in families:
                family_id = family["familyid"]
                # Get regular devices
                devices = await self.list_devices(family_id)
                _LOGGER.debug("Fetched devices for family %s: %s", family_id, devices)
                if devices:
                    all_devices.extend(devices)

                # Check for shared devices using cached method
                if await self._has_shared_devices(family_id):
                    shared_devices = await self.list_devices(family_id, shared=True)
                    _LOGGER.debug(
                        "Fetched shared devices for family %s: %s",
                        family_id,
                        shared_devices,
                    )
                    if shared_devices:
                        all_devices.extend(shared_devices)
                else:
                    _LOGGER.debug(
                        "No shared devices found for family %s (cached)", family_id
                    )

        except Exception:
            _LOGGER.exception("Error getting devices")
            raise

        return all_devices

    @alru_cache(maxsize=1, ttl=3600)  # Cache 1 result for 1 hour
    @create_retry_decorator()
    async def list_families(self, retry_count: int = 0) -> list[dict[str, Any]]:
        """Get list of all families with retry."""
        _LOGGER.debug("Fetching list of families")
        max_retries = 3

        # Check if we're logged in
        if not hasattr(self, "loginsession") or not self.loginsession:
            _LOGGER.debug("No login session found, attempting to login")
            await self.login()

        session = await self._get_session()
        async with session.post(
            f"{self.url}/appsync/group/member/getfamilylist",
            headers=self._get_headers(),
        ) as response:
            data = await response.text()
            try:
                json_data = json.loads(data)
            except json.JSONDecodeError as ex:
                msg = f"Failed to decode JSON: {ex}"
                _LOGGER.exception(msg)
                json_data = {"status": -1, "msg": data}

            if "status" in json_data:
                if json_data["status"] == 0:
                    self.data = {}
                    for family in json_data["data"]["familyList"]:
                        self.data[family["familyid"]] = {
                            "id": family["familyid"],
                            "name": family["name"],
                            "rooms": [],
                            "devices": [],
                        }
                    _LOGGER.debug(
                        "Fetched family list: %s", json_data["data"]["familyList"]
                    )
                    return json_data["data"]["familyList"]
                if json_data["status"] == self.LOGIN_VALIDATION_FAILED:
                    if retry_count >= max_retries:
                        msg = "Login validation failed after retries"
                        _LOGGER.error(msg)
                        raise AuxCloudAuthError(msg)
                    # Login validation failed, re-login and retry the request
                    _LOGGER.warning("Login validation failed, attempting to re-login")
                    await self.login()
                    # Retry the request after re-login
                    return await self.list_families(retry_count + 1)

            msg = f"Failed to get families list: {data}"
            _LOGGER.error(msg)
            raise AuxCloudApiError(msg)

    @alru_cache(maxsize=1, ttl=3600)  # Cache 1 result for 1 hour
    async def _has_shared_devices(self, family_id: str) -> bool:
        """Check if family has any shared devices with 1-hour cache."""
        _LOGGER.debug("Checking for shared devices in family: %s", family_id)
        try:
            shared_devices = await self.list_devices(family_id, shared=True)
            _LOGGER.debug(
                "Shared devices for family %s: %s",
                family_id,
                shared_devices,
            )
            return len(shared_devices) > 0
        except AuxCloudApiError as ex:
            _LOGGER.warning("API error checking shared devices: %s", ex)
            return False
        except AuxCloudAuthError as ex:
            _LOGGER.warning("Authentication error checking shared devices: %s", ex)
            return False
        except aiohttp.ClientError as ex:
            _LOGGER.warning("Network error checking shared devices: %s", ex)
            return False

    @create_retry_decorator()
    async def list_devices(
        self, family_id: str, *, shared: bool = False
    ) -> list[dict[str, Any]]:
        """Get devices for a specific family with retry."""
        _LOGGER.debug(
            "Fetching devices for family_id: %s, shared: %s", family_id, shared
        )
        session = await self._get_session()
        device_endpoint = (
            "dev/query?action=select"
            if not shared
            else "sharedev/querylist?querytype=shared"
        )
        async with session.post(
            f"{self.url}/appsync/group/{device_endpoint}",
            data='{"pids":[]}' if not shared else '{"endpointId":""}',
            headers=self._get_headers(familyid=family_id),
        ) as response:
            data = await response.text()
            json_data = json.loads(data)

            if "status" in json_data and json_data["status"] == 0:
                if "endpoints" in json_data["data"]:
                    devices = json_data["data"]["endpoints"]
                elif "shareFromOther" in json_data["data"]:
                    devices = [
                        dev["devinfo"] for dev in json_data["data"]["shareFromOther"]
                    ]

                # Initialize family data structure if needed
                if family_id not in self.data:
                    self.data[family_id] = {
                        "id": family_id,
                        "name": "",  # Could be populated from family data
                        "rooms": [],
                        "devices": [],
                    }

                # Process devices and update internal cache
                processed_devices = []
                for dev in devices:
                    # Create tasks for all API calls for each device
                    state_task = self.query_device_state(
                        dev["endpointId"], dev["devSession"]
                    )
                    params_task = self.get_device_params(dev)
                    ambient_task = self.get_device_params(dev, ["mode"])
                    results = await asyncio.gather(
                        state_task, params_task, ambient_task, return_exceptions=True
                    )

                    state_result, params_result, ambient_result = results

                    # Handle results, checking for exceptions
                    if not isinstance(state_result, Exception):
                        dev["state"] = state_result["data"][0]["state"]

                    if not isinstance(params_result, Exception):
                        dev["params"] = params_result

                    if not isinstance(ambient_result, Exception) and "params" in dev:
                        dev["params"]["envtemp"] = ambient_result["envtemp"]
                    _LOGGER.debug("Processed device: %s", dev)
                    processed_devices.append(dev)

                # Update internal cache - replace existing devices for this family
                if not shared:
                    self.data[family_id]["devices"] = processed_devices
                else:
                    # For shared devices, append to existing devices
                    existing_ids = {
                        d["endpointId"] for d in self.data[family_id]["devices"]
                    }
                    new_devices = [
                        d
                        for d in processed_devices
                        if d["endpointId"] not in existing_ids
                    ]
                    self.data[family_id]["devices"].extend(new_devices)

                return processed_devices

            msg = f"Failed to get devices: {data}"
            raise AuxCloudApiError(msg)

    async def get_device_params(
        self, device: dict[str, Any], params: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Get device parameters.

        Args:
            device: Device information dictionary
            params: List of parameter names to get

        Returns:
            Dictionary of parameter values

        """
        if params is None:
            params = []
        _LOGGER.debug(
            "Fetching device parameters for device: %s, params = %s", device, params
        )
        return await self._act_device_params(device, "get", params)

    async def set_device_params(
        self, device: dict[str, Any], values: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Set device parameters.

        Args:
            device: Device information dictionary
            values: Dictionary of parameter names and values to set

        Returns:
            Dictionary of updated parameter values

        """
        _LOGGER.info(
            "Setting device parameters for device: %s with values: %s", device, values
        )
        params = list(values.keys())
        vals = [[{"val": val, "idx": 1}] for val in values.values()]
        return await self._act_device_params(device, "set", params, vals)

    @create_retry_decorator()
    async def query_device_state(
        self, device_id: str, dev_session: str
    ) -> dict[str, Any]:
        """Query device state with retry."""
        _LOGGER.debug("Querying device state for device_id: %s", device_id)
        session = await self._get_session()
        timestamp = int(time.time())
        data = {
            "directive": {
                "header": self._get_directive_header(
                    namespace="DNA.QueryState",
                    name="queryState",
                    messageType="controlgw.batch",
                    message_id_prefix=self.userid,
                    timestamp=f"{timestamp}",
                ),
                "payload": {
                    "studata": [{"did": device_id, "devSession": dev_session}],
                    "msgtype": "batch",
                },
            }
        }

        _LOGGER.debug("Sending query state request with data: %s", data)

        async with session.post(
            f"{self.url}/device/control/v2/querystate",
            data=json.dumps(data, separators=(",", ":")),
            headers=self._get_headers(),
        ) as response:
            data = await response.text()
            _LOGGER.debug("Received response: %s", data)
            json_data = json.loads(data)

            if (
                "event" in json_data
                and "payload" in json_data["event"]
                and json_data["event"]["payload"]["status"] == 0
            ):
                _LOGGER.debug(
                    "Queried device state for device_id %s: %s",
                    device_id,
                    json_data["event"]["payload"],
                )
                return json_data["event"]["payload"]

            _LOGGER.error("Failed to query device state: %s", data)
            msg = f"Failed to query device state: {data}"
            raise AuxCloudApiError(msg)

    @create_retry_decorator()
    async def query_device_temperature(
        self, device_id: str, dev_session: str
    ) -> dict[str, Any]:
        """Query device temperature with retry."""
        _LOGGER.debug("Querying device temperature for device_id: %s", device_id)
        session = await self._get_session()
        async with session.post(
            f"{self.url}/device/control/v2/temperaturesensor",
            data=json.dumps(
                self._build_temperature_query_data(device_id, dev_session),
                separators=(",", ":"),
            ),
            headers=self._get_headers(),
        ) as resp:
            data = await resp.text()
            _LOGGER.debug("Received response: %s", data)
            json_data = json.loads(data)

            if (
                "event" in json_data
                and "payload" in json_data["event"]
                and json_data["event"]["payload"]["status"] == 0
            ):
                _LOGGER.debug(
                    "Queried device temperature for device_id %s: %s",
                    device_id,
                    json_data["event"]["payload"],
                )
                return json_data["event"]["payload"]

            error_msg = f"Failed to query device temperature: {data}"
            _LOGGER.error(error_msg)
            raise AuxCloudApiError(error_msg)

    def _build_temperature_query_data(
        self, device_id: str, dev_session: str
    ) -> dict[str, Any]:
        """Build the data payload for temperature query."""
        timestamp = int(time.time())
        return {
            "directive": {
                "header": self._get_directive_header(
                    namespace="DNA.TemperatureSensor",
                    name="ReportState",
                    message_id_prefix=self.userid,
                    timestamp=f"{timestamp}",
                ),
                "endpoint": {
                    "endpointId": device_id,
                    "devicePairedInfo": {"did": device_id, "devSession": dev_session},
                    "cookie": {},
                },
                "payload": {},
            }
        }

    @create_retry_decorator()
    async def _act_device_params(
        self,
        device: dict[str, Any],
        act: str,
        params: list[str] | None = None,
        vals: list[list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Act on device parameters with retry."""
        params = params or []
        vals = vals or []
        _LOGGER.debug(
            "Acting on device parameters for device: %s, action: %s", device, act
        )

        if act == "set" and len(params) != len(vals):
            msg = "Params and Vals must have the same length"
            raise ValueError(msg)

        session = await self._get_session()
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
                "header": self._get_directive_header(
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
                    "act": act,
                    "params": params,
                    "vals": vals,
                },
            }
        }

        if self._is_ambient_mode(params):
            data["directive"]["payload"]["did"] = device["endpointId"]
            data["directive"]["payload"]["vals"] = [[{"val": 0, "idx": 1}]]

        async with session.post(
            f"{self.url}/device/control/v2/sdkcontrol",
            params={"license": LICENSE},
            data=json.dumps(data, separators=(",", ":")),
            headers=self._get_headers(),
        ) as resp:
            response_text = await resp.text()
            json_data = json.loads(response_text)

            if all(
                key in json_data.get("event", {}).get("payload", {})
                for key in ("data",)
            ):
                response = json.loads(json_data["event"]["payload"]["data"])
                _LOGGER.debug(
                    "Acted on device parameters for device %s: %s", device, response
                )
                return {
                    response["params"][i]: response["vals"][i][0]["val"]
                    for i in range(len(response["params"]))
                }

            msg = f"Failed to {act} device parameters: {response_text}"
            raise ValueError(msg)

    def _is_ambient_mode(self, params: list[str]) -> bool:
        """
        Check if the parameters indicate ambient mode.

        Args:
            params: List of parameter names

        Returns:
            True if parameters indicate ambient mode, False otherwise

        """
        return len(params) == 1 and params[0] == "mode"

    async def refresh(self) -> None:
        """
        Refresh all family and device data.

        Raises:
            Exception: If refresh operation fails

        """
        _LOGGER.debug("Refreshing all data")
        try:
            family_data = await self.list_families()
            tasks = [
                self.list_devices(family["familyid"]) for family in family_data
            ] + [
                self.list_devices(family["familyid"], shared=True)
                for family in family_data
            ]
            await asyncio.gather(*tasks)
        except Exception:
            _LOGGER.exception("Error refreshing data")
            raise
