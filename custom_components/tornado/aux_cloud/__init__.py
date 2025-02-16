"""AuxCloud API client for Tornado AC."""
from __future__ import annotations

import base64
import hashlib
import json
import aiohttp
import asyncio
import time
import logging
from typing import Any, List, Dict
from io import BytesIO
from pathlib import Path
from .util import encrypt_aes_cbc_zero_padding

_LOGGER = logging.getLogger(__name__)

# Constants from original implementation
TIMESTAMP_TOKEN_ENCRYPT_KEY = 'kdixkdqp54545^#*'
PASSWORD_ENCRYPT_KEY = '4969fj#k23#'
BODY_ENCRYPT_KEY = 'xgx3d*fe3478$ukx'
AES_INITIAL_VECTOR = bytes([(b + 256) % 256 for b in [-22, -86, -86, 58, -69, 88, 98, -94, 25, 24, -75, 119, 29, 22, 21, -86]])
LICENSE = 'PAFbJJ3WbvDxH5vvWezXN5BujETtH/iuTtIIW5CE/SeHN7oNKqnEajgljTcL0fBQQWM0XAAAAAAnBhJyhMi7zIQMsUcwR/PEwGA3uB5HLOnr+xRrci+FwHMkUtK7v4yo0ZHa+jPvb6djelPP893k7SagmffZmOkLSOsbNs8CAqsu8HuIDs2mDQAAAAA='
LICENSE_ID = '3c015b249dd66ef0f11f9bef59ecd737'
COMPANY_ID = '48eb1b36cf0202ab2ef07b880ecda60d'
SPOOF_APP_VERSION = "2.2.10.456537160"
SPOOF_USER_AGENT = 'Dalvik/2.1.0 (Linux; U; Android 12; SM-G991B Build/SP1A.210812.016)'
SPOOF_SYSTEM = 'android'
SPOOF_APP_PLATFORM = 'android'
API_SERVER_URL_EU = "https://app-service-deu-f0e9ebbb.smarthomecs.de"
API_SERVER_URL_USA = "https://app-service-usa-fd7cc04c.smarthomecs.com"

class AuxCloudAPI:
    """API Client for AuxCloud."""
    
    def __init__(self, email: str, password: str, region: str = 'eu', session_file: str = None) -> None:
        """Initialize the API client."""
        self.session_file = session_file
        self.url = API_SERVER_URL_EU if region == 'eu' else API_SERVER_URL_USA
        self.email = email
        self.password = password
        self.data = {}  # Store family and device data
        _LOGGER.debug("Initialized AuxCloudAPI with email: %s, region: %s", email, region)

    def _get_headers(self, **kwargs: str) -> dict:
        """Get request headers."""
        headers = {
            "Content-Type": "application/x-java-serialized-object",
            "licenseId": LICENSE_ID,
            "lid": LICENSE_ID,
            "language": "en",
            "appVersion": SPOOF_APP_VERSION,
            "User-Agent": SPOOF_USER_AGENT,
            "system": SPOOF_SYSTEM,
            "appPlatform": SPOOF_APP_PLATFORM,
            "loginsession": getattr(self, 'loginsession', ''),
            "userid": getattr(self, 'userid', ''),
            **kwargs
        }
        _LOGGER.debug("Generated headers: %s", headers)
        return headers

    def _get_directive_header(self, namespace: str, name: str, message_id_prefix: str, **kwargs: str) -> dict:
        """Get directive header for device control."""
        timestamp = int(time.time())
        header = {
            "namespace": namespace,
            "name": name,
            "interfaceVersion": "2",
            "senderId": "sdk",
            "messageId": f'{message_id_prefix}-{timestamp}',
            **kwargs
        }
        _LOGGER.debug("Generated directive header: %s", header)
        return header

    async def login(self, email: str = None, password: str = None) -> bool:
        """Login to AuxCloud."""
        email = email if email is not None else self.email
        password = password if password is not None else self.password
        _LOGGER.info("Attempting to login with email: %s", email)

        if self.session_file and Path(self.session_file).exists():
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    self.userid = session_data['userid']
                    self.loginsession = session_data['loginsession']
                _LOGGER.info("Loaded session from file: %s", self.session_file)
                return True
            except Exception as ex:
                _LOGGER.warning("Failed to load session file: %s", str(ex))

        try:
            currentTime = time.time()
            shaPassword = hashlib.sha1(f'{password}{PASSWORD_ENCRYPT_KEY}'.encode()).hexdigest()
            payload = {
                "email": email,
                "password": shaPassword,
                "companyid": COMPANY_ID,
                "lid": LICENSE_ID
            }
            jsonPayload = json.dumps(payload, separators=(',', ':'))
            token = hashlib.md5(f'{jsonPayload}{BODY_ENCRYPT_KEY}'.encode()).hexdigest()
            md5 = hashlib.md5(f'{currentTime}{TIMESTAMP_TOKEN_ENCRYPT_KEY}'.encode()).digest()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.url}/account/login',
                    data=encrypt_aes_cbc_zero_padding(AES_INITIAL_VECTOR, md5, jsonPayload.encode()),
                    headers=self._get_headers(timestamp=f'{currentTime}', token=token),
                ) as response:
                    data = await response.text()
                    json_data = json.loads(data)

                    if 'status' in json_data and json_data['status'] == 0:
                        self.loginsession = json_data['loginsession']
                        self.userid = json_data['userid']
                        if self.session_file:
                            with open(self.session_file, 'w', encoding='utf-8') as f:
                                json.dump({'userid': self.userid, 'loginsession': self.loginsession}, f)
                        _LOGGER.info("Login successful for email: %s", email)
                        return True
                    raise Exception(f"Login failed: {data}")

        except Exception as ex:
            _LOGGER.error("Login error: %s", str(ex))
            raise

    async def get_devices(self) -> List[Dict[str, Any]]:
        """Get all devices across all families."""
        _LOGGER.debug("Fetching all devices")
        all_devices = []
        try:
            # First get all families
            families = await self.list_families()
            _LOGGER.debug("Fetched families: %s", families)
            
            # Then get devices for each family
            for family in families:
                family_id = family['familyid']
                # Get regular devices
                devices = await self.list_devices(family_id)
                _LOGGER.debug("Fetched devices for family %s: %s", family_id, devices)
                all_devices.extend(devices)
                
                # Get shared devices
                shared_devices = await self.list_devices(family_id, shared=True)
                _LOGGER.debug("Fetched shared devices for family %s: %s", family_id, shared_devices)
                all_devices.extend(shared_devices)
                
        except Exception as ex:
            _LOGGER.error("Error getting devices: %s", str(ex))
            raise

        return all_devices

    async def list_families(self) -> List[Dict[str, Any]]:
        """Get list of all families."""
        _LOGGER.debug("Fetching list of families")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'{self.url}/appsync/group/member/getfamilylist',
                headers=self._get_headers(),
            ) as response:
                data = await response.text()
                json_data = json.loads(data)

                if 'status' in json_data and json_data['status'] == 0:
                    self.data = {}
                    for family in json_data['data']['familyList']:
                        self.data[family['familyid']] = {
                            'id': family['familyid'],
                            'name': family['name'],
                            'rooms': [],
                            'devices': []
                        }
                    _LOGGER.debug("Fetched family list: %s", json_data['data']['familyList'])
                    return json_data['data']['familyList']
                raise Exception(f"Failed to get families list: {data}")

    async def list_devices(self, family_id: str, shared: bool = False) -> List[Dict[str, Any]]:
        """Get devices for a specific family."""
        _LOGGER.debug("Fetching devices for family_id: %s, shared: %s", family_id, shared)
        async with aiohttp.ClientSession() as session:
            device_endpoint = 'dev/query?action=select' if not shared else 'sharedev/querylist?querytype=shared'
            async with session.post(
                f'{self.url}/appsync/group/{device_endpoint}',
                data='{"pids":[]}' if not shared else '{"endpointId":""}',
                headers=self._get_headers(familyid=family_id),
            ) as response:
                data = await response.text()
                json_data = json.loads(data)

                if 'status' in json_data and json_data['status'] == 0:
                    if 'endpoints' in json_data['data']:
                        devices = json_data['data']['endpoints']
                    elif 'shareFromOther' in json_data['data']:
                        devices = [dev['devinfo'] for dev in json_data['data']['shareFromOther']]

                    for dev in devices:
                        # Get device state
                        dev_state = await self.query_device_state(dev['endpointId'], dev['devSession'])
                        dev['state'] = dev_state['data'][0]['state']
                        
                        # Get device parameters
                        dev_params = await self.get_device_params(dev)
                        dev['params'] = dev_params

                        # Get device ambient mode
                        ambient_mode = await self.get_device_params(dev, ['mode'])
                        _LOGGER.debug("Ambient mode for device %s: %s", dev['endpointId'], ambient_mode)
                        dev['params']['envtemp'] = ambient_mode['envtemp']
                        
                        if not any(d['endpointId'] == dev['endpointId'] for d in self.data[family_id]['devices']):
                            self.data[family_id]['devices'].append(dev)

                    _LOGGER.debug("Fetched devices for family_id %s: %s", family_id, devices)
                    return devices
                raise Exception(f"Failed to get devices: {data}")

    async def get_device_params(self, device: dict, params: list[str] = []) -> Dict[str, Any]:
        """Get device parameters."""
        _LOGGER.debug("Fetching device parameters for device: %s, params = %s", device, params)
        return await self._act_device_params(device, "get", params)

    async def set_device_params(self, device: dict, values: dict) -> Dict[str, Any]:
        """Set device parameters."""
        _LOGGER.info("Setting device parameters for device: %s with values: %s", device, values)
        params = list(values.keys())
        vals = [[{"val": val, "idx": 1}] for val in values.values()]
        return await self._act_device_params(device, "set", params, vals)

    async def query_device_state(self, device_id: str, dev_session: str) -> Dict[str, Any]:
        """Query device state."""
        _LOGGER.debug("Querying device state for device_id: %s", device_id)
        async with aiohttp.ClientSession() as session:
            timestamp = int(time.time())
            data = {
                "directive": {
                    "header": self._get_directive_header(
                        namespace="DNA.QueryState",
                        name="queryState",
                        messageType="controlgw.batch",
                        message_id_prefix=self.userid,
                        timestamp=f'{timestamp}'
                    ),
                    "payload": {
                        "studata": [{
                            "did": device_id,
                            "devSession": dev_session
                        }],
                        "msgtype": "batch"
                    }
                }
            }

            _LOGGER.debug("Sending query state request with data: %s", data)

            async with session.post(
                f'{self.url}/device/control/v2/querystate',
                data=json.dumps(data, separators=(',', ':')),
                headers=self._get_headers(),
            ) as response:
                data = await response.text()
                _LOGGER.debug("Received response: %s", data)
                json_data = json.loads(data)

                if ('event' in json_data and 
                    'payload' in json_data['event'] and 
                    json_data['event']['payload']['status'] == 0):
                    _LOGGER.debug("Queried device state for device_id %s: %s", device_id, json_data['event']['payload'])
                    return json_data['event']['payload']
                else:
                    _LOGGER.error("Failed to query device state: %s", data)
                    raise Exception(f"Failed to query device state: {data}")
    

    async def query_device_temperature(self, device_id: str, dev_session: str) -> Dict[str, Any]:
        """Query device temperature."""
        _LOGGER.debug("Querying device temperature for device_id: %s", device_id)
        async with aiohttp.ClientSession() as session:
            timestamp = int(time.time())
            data = {
                "directive": {
                    "header": self._get_directive_header(
                        namespace="DNA.TemperatureSensor",
                        name="ReportState",
                        message_id_prefix=self.userid,
                        timestamp=f'{timestamp}'
                    ),
                    "endpoint": {
                        "endpointId": device_id,
                        "devicePairedInfo": {
                            "did": device_id,
                            "devSession": dev_session
                        },
                        "cookie": {}
                    },
                    "payload": {}
                }
            }

            _LOGGER.debug("Sending query temperature request with data: %s", data)

            async with session.post(
                f'{self.url}/device/control/v2/temperaturesensor',
                data=json.dumps(data, separators=(',', ':')),
                headers=self._get_headers(),
            ) as response:
                data = await response.text()
                _LOGGER.debug("Received response: %s", data)
                json_data = json.loads(data)

                if ('event' in json_data and 
                    'payload' in json_data['event'] and 
                    json_data['event']['payload']['status'] == 0):
                    _LOGGER.debug("Queried device temperature for device_id %s: %s", device_id, json_data['event']['payload'])
                    return json_data['event']['payload']
                else:
                    _LOGGER.error("Failed to query device temperature: %s", data)
                    raise Exception(f"Failed to query device temperature: {data}")

    async def _act_device_params(self, device: dict, act: str, params: list[str] = [], vals: list[str] = []) -> Dict[str, Any]:
        """Internal method to get or set device parameters."""
        _LOGGER.debug("Acting on device parameters for device: %s, action: %s", device, act)
        if act == "set" and len(params) != len(vals):
            raise Exception("Params and Vals must have the same length")

        async with aiohttp.ClientSession() as session:
            cookie = json.loads(base64.b64decode(device['cookie'].encode()))
            mapped_cookie = base64.b64encode(json.dumps({
                "device": {
                    "id": cookie['terminalid'],
                    "key": cookie['aeskey'],
                    "devSession": device['devSession'],
                    "aeskey": cookie['aeskey'],
                    "did": device['endpointId'],
                    "pid": device['productId'],
                    "mac": device['mac'],
                }
            }, separators=(',', ':')).encode()).decode()

            data = {
                "directive": {
                    "header": self._get_directive_header(
                        namespace="DNA.KeyValueControl",
                        name="KeyValueControl",
                        message_id_prefix=device['endpointId']
                    ),
                    "endpoint": {
                        "devicePairedInfo": {
                            "did": device['endpointId'],
                            "pid": device['productId'],
                            "mac": device['mac'],
                            "devicetypeflag": device['devicetypeFlag'],
                            "cookie": mapped_cookie
                        },
                        "endpointId": device['endpointId'],
                        "cookie": {},
                        "devSession": device['devSession'],
                    },
                    "payload": {
                        "act": act,
                        "params": params,
                        "vals": vals
                    },
                }
            }
            # Special case for getting ambient mode
            if self._is_ambient_mode(params):
                data['directive']['payload']['did'] = device['endpointId']
                data['directive']['payload']['vals'] = [[{'val': 0, 'idx': 1}]]

            async with session.post(
                f'{self.url}/device/control/v2/sdkcontrol',
                params={"license": LICENSE},
                data=json.dumps(data, separators=(',', ':')),
                headers=self._get_headers(),
            ) as response:
                data = await response.text()
                json_data = json.loads(data)

                if ('event' in json_data and 
                    'payload' in json_data['event'] and 
                    'data' in json_data['event']['payload']):
                    response = json.loads(json_data['event']['payload']['data'])
                    _LOGGER.debug("Acted on device parameters for device %s: %s", device, response)
                    return {response['params'][i]: response['vals'][i][0]['val'] 
                            for i in range(len(response['params']))}
                raise Exception(f"Failed to {act} device parameters: {data}")

    def _is_ambient_mode(self, params: list[str]) -> bool:
        """Check if the parameters indicate ambient mode."""
        return len(params) == 1 and params[0] == 'mode'

    async def refresh(self) -> None:
        """Refresh all data."""
        _LOGGER.info("Refreshing all data")
        try:
            family_data = await self.list_families()
            tasks = []
            for family in family_data:
                tasks.append(self.list_devices(family['familyid']))
                tasks.append(self.list_devices(family['familyid'], shared=True))
            await asyncio.gather(*tasks)
        except Exception as ex:
            _LOGGER.error("Error refreshing data: %s", str(ex))
            raise
