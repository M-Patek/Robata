"""Small synchronous HTTP boundary for the Mage native-video endpoint.

The adapter owns canonical request construction.  This transport merely sends
those exact bytes to the endpoint route contained in the prepared invocation,
then validates the endpoint's durable response envelope.  It has no model,
Qwen, or media-decoding dependency and is usable with a local loopback URL or a
remote RunPod-style HTTP endpoint whose result artifact storage is mounted to the
consumer.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import suppress
from typing import Final, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

from robata.inference.mage_video_adapter import (
    MageVideoInferenceTransport,
    MageVideoObservationAdapterError,
    MageVideoObservationTransportRequest,
)
from robata.inference.mage_video_endpoint import (
    MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER,
    MageVideoEndpointResponse,
    MageVideoHealthResponse,
)

_DEFAULT_TIMEOUT_SECONDS: Final = 7_260.0
_MAX_ERROR_BODY_BYTES: Final = 8_192


class MageVideoHttpTransportError(MageVideoObservationAdapterError):
    """The declared Mage HTTP endpoint could not produce a valid envelope."""


class MageVideoHttpResponse(Protocol):
    """Tiny testable subset of a urllib response."""

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


MageVideoHttpOpener = Callable[..., MageVideoHttpResponse]


class MageVideoHttpTransport(MageVideoInferenceTransport):
    """Send an adapter-prepared native-video request to a declared HTTP endpoint."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        opener: MageVideoHttpOpener | None = None,
    ) -> None:
        self._base_url = _normalise_endpoint_base_url(endpoint_url)
        if not isinstance(timeout_seconds, (float, int)) or isinstance(timeout_seconds, bool):
            raise TypeError("timeout_seconds must be a positive finite number")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener or urlopen

    @property
    def endpoint_url(self) -> str:
        return self._base_url

    def infer(self, invocation: MageVideoObservationTransportRequest) -> MageVideoEndpointResponse:
        if not isinstance(invocation, MageVideoObservationTransportRequest):
            raise TypeError("invocation must be MageVideoObservationTransportRequest")
        path = invocation.endpoint_path
        if not isinstance(path, str) or not path.startswith("/"):
            raise MageVideoHttpTransportError("Mage endpoint invocation path must be absolute")
        response_bytes = _request_bytes(
            opener=self._opener,
            request=Request(
                url=f"{self._base_url}{path}",
                data=invocation.request_body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    MAGE_VIDEO_ENDPOINT_IDEMPOTENCY_HEADER: invocation.idempotency_key,
                },
                method="POST",
            ),
            timeout_seconds=self._timeout_seconds,
        )
        try:
            return MageVideoEndpointResponse.model_validate_json(response_bytes, strict=True)
        except (ValidationError, TypeError, ValueError) as error:
            raise MageVideoHttpTransportError(
                "Mage endpoint response is not a strict durable response envelope"
            ) from error


def fetch_mage_video_endpoint_health(
    *,
    endpoint_url: str,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    opener: MageVideoHttpOpener | None = None,
) -> MageVideoHealthResponse:
    """Read the ready endpoint's configured immutable model identity."""

    base_url = _normalise_endpoint_base_url(endpoint_url)
    if not isinstance(timeout_seconds, (float, int)) or isinstance(timeout_seconds, bool):
        raise TypeError("timeout_seconds must be a positive finite number")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    response_bytes = _request_bytes(
        opener=opener or urlopen,
        request=Request(
            url=f"{base_url}/healthz",
            headers={"Accept": "application/json"},
            method="GET",
        ),
        timeout_seconds=float(timeout_seconds),
    )
    try:
        return MageVideoHealthResponse.model_validate_json(response_bytes, strict=True)
    except (ValidationError, TypeError, ValueError) as error:
        raise MageVideoHttpTransportError(
            "Mage endpoint health response is not a strict readiness envelope"
        ) from error


def _normalise_endpoint_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("endpoint_url must be a nonempty HTTP URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint_url must use http or https and include a host")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint_url must not contain a query or fragment")
    return value.strip().rstrip("/")


def _request_bytes(
    *,
    opener: MageVideoHttpOpener,
    request: Request,
    timeout_seconds: float,
) -> bytes:
    response: MageVideoHttpResponse | None = None
    try:
        response = opener(request, timeout=timeout_seconds)
        payload = response.read()
    except HTTPError as error:
        detail = _error_detail(error.read(_MAX_ERROR_BODY_BYTES))
        raise MageVideoHttpTransportError(
            f"Mage endpoint returned HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise MageVideoHttpTransportError("Mage endpoint HTTP transport failed") from error
    except OSError as error:
        raise MageVideoHttpTransportError("Mage endpoint HTTP transport failed") from error
    finally:
        if response is not None:
            with suppress(OSError):
                response.close()
    if not isinstance(payload, bytes) or not payload:
        raise MageVideoHttpTransportError("Mage endpoint returned an empty response body")
    return payload


def _error_detail(payload: bytes) -> str:
    if not payload:
        return "empty error body"
    try:
        parsed = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "non-JSON error body"
    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return "JSON error body"


__all__ = [
    "MageVideoHttpOpener",
    "MageVideoHttpResponse",
    "MageVideoHttpTransport",
    "MageVideoHttpTransportError",
    "fetch_mage_video_endpoint_health",
]
