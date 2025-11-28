"""HTTP client wrapper for the Langfuse public API."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


class LangfuseError(Exception):
    """Raised when a Langfuse API call fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class LangfuseCredentials:
    """Credentials and host information used to talk to Langfuse."""

    url: str
    public_key: str
    secret_key: str
    project_id: str | None = None


@dataclass(frozen=True)
class DatasetStatus:
    """Information about a dataset lookup/create operation."""

    name: str
    id: str | None
    existed: bool
    created: bool
    dry_run_creation: bool = False


class LangfuseHttpClient:
    """Thin HTTPX wrapper focused on the endpoints needed for trace import."""

    def __init__(
        self,
        credentials: LangfuseCredentials,
        *,
        timeout: float = 10.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = credentials.url.rstrip("/") + "/api/public"
        headers = {}
        if credentials.project_id:
            headers["X-Langfuse-Project"] = credentials.project_id

        self._client = httpx.Client(
            base_url=self._base_url,
            auth=httpx.BasicAuth(credentials.public_key, credentials.secret_key),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    def list_traces(
        self,
        *,
        from_timestamp: str,
        to_timestamp: str,
        environment: str | None,
        limit: int = 100,
    ) -> Iterable[dict[str, Any]]:
        """Yield trace summaries within the requested time range."""
        page = 1
        params: dict[str, Any] = {
            "fromTimestamp": from_timestamp,
            "toTimestamp": to_timestamp,
            "limit": limit,
            "page": page,
            "orderBy": "timestamp.desc",
            "fields": "core,io",
        }
        if environment:
            params["environment"] = [environment]

        while True:
            response = self._request("GET", "/traces", params=params)
            body = response.json()
            data = body.get("data", [])
            if not isinstance(data, list):
                raise LangfuseError("Unexpected traces response shape")
            for trace in data:
                if not isinstance(trace, dict):
                    raise LangfuseError("Unexpected trace entry in response")
                yield trace
            meta = body.get("meta") or {}
            total_pages = int(meta.get("totalPages") or 1)
            if page >= total_pages:
                break
            page += 1
            params["page"] = page

    def list_traces_with_total(
        self,
        *,
        from_timestamp: str,
        to_timestamp: str,
        environment: str | None,
        limit: int = 100,
    ) -> tuple[Iterable[dict[str, Any]], int]:
        """Yield trace summaries and return the reported total count."""
        page = 1
        params: dict[str, Any] = {
            "fromTimestamp": from_timestamp,
            "toTimestamp": to_timestamp,
            "limit": limit,
            "page": page,
            "orderBy": "timestamp.desc",
            "fields": "core,io",
        }
        if environment:
            params["environment"] = [environment]

        first_response = self._request("GET", "/traces", params=params)
        body = first_response.json()
        data = body.get("data", [])
        if not isinstance(data, list):
            raise LangfuseError("Unexpected traces response shape")
        meta = body.get("meta") or {}
        total_pages = int(meta.get("totalPages") or 1)
        total_items = int(meta.get("totalItems") or len(data))

        def generator() -> Iterable[dict[str, Any]]:
            for trace in data:
                if not isinstance(trace, dict):
                    raise LangfuseError("Unexpected trace entry in response")
                yield trace

            params_inner = dict(params)
            params_inner["page"] = 2
            current_page = 2
            while current_page <= total_pages:
                response = self._request("GET", "/traces", params=params_inner)
                page_body = response.json()
                page_data = page_body.get("data", [])
                if not isinstance(page_data, list):
                    raise LangfuseError("Unexpected trace entry in response")
                for trace in page_data:
                    if not isinstance(trace, dict):
                        raise LangfuseError("Unexpected trace entry in response")
                    yield trace
                current_page += 1
                params_inner["page"] = current_page

        return generator(), total_items

    def get_trace(self, trace_id: str) -> dict[str, Any]:
        """Fetch a trace with full observation details."""
        response = self._request("GET", f"/traces/{trace_id}")
        body = response.json()
        if not isinstance(body, dict):
            raise LangfuseError("Unexpected trace payload")
        return body

    def ensure_dataset(self, dataset_name: str, *, dry_run: bool = False) -> DatasetStatus:
        """Ensure the dataset exists, creating it when necessary."""
        exists = self._request(
            "GET",
            f"/v2/datasets/{dataset_name}",
            allow_404=True,
        )
        if exists is not None:
            body = exists.json()
            dataset_id = body.get("id") if isinstance(body, dict) else None
            return DatasetStatus(
                name=dataset_name,
                id=str(dataset_id) if dataset_id else None,
                existed=True,
                created=False,
            )

        if dry_run:
            return DatasetStatus(
                name=dataset_name,
                id=None,
                existed=False,
                created=False,
                dry_run_creation=True,
            )

        created = self._request(
            "POST",
            "/datasets",
            json={"name": dataset_name},
        )
        created_body = created.json() if created is not None else {}
        dataset_id = created_body.get("id") if isinstance(created_body, dict) else None
        return DatasetStatus(
            name=dataset_name,
            id=str(dataset_id) if dataset_id else None,
            existed=False,
            created=True,
        )

    def dataset_item_exists(
        self, dataset_name: str, *, trace_id: str | None = None, item_id: str | None = None
    ) -> bool:
        """Check whether a dataset item already exists."""
        if item_id:
            response = self._request(
                "GET",
                f"/dataset-items/{item_id}",
                allow_404=True,
            )
            return response is not None

        params: dict[str, Any] = {"datasetName": dataset_name, "limit": 1}
        if trace_id:
            params["sourceTraceId"] = trace_id
        response = self._request("GET", "/dataset-items", params=params, allow_404=True)
        if response is None:
            return False
        body = response.json()
        meta = body.get("meta") or {}
        total_items = int(meta.get("totalItems") or 0)
        return total_items > 0

    def create_dataset_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create or upsert a dataset item."""
        response = self._request("POST", "/dataset-items", json=payload)
        body = response.json()
        if not isinstance(body, dict):
            raise LangfuseError("Unexpected dataset item response")
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> httpx.Response | None:
        """Execute an HTTP request with exponential backoff."""
        delay = self._backoff_seconds
        last_error: LangfuseError | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.request(method, path, params=params, json=json)
            except httpx.HTTPError as exc:  # network or protocol issues
                last_error = LangfuseError(str(exc))
            else:
                if allow_404 and response.status_code == 404:
                    return None
                if response.status_code == 200:
                    return response
                if response.status_code in {429} or response.status_code >= 500:
                    last_error = LangfuseError(
                        f"Temporary error {response.status_code}: {response.text}",
                        status_code=response.status_code,
                    )
                else:
                    raise LangfuseError(
                        f"Langfuse request failed ({response.status_code}): {response.text}",
                        status_code=response.status_code,
                    )

            if attempt == self._max_attempts:
                assert last_error is not None
                raise last_error
            time.sleep(delay)
            delay *= 2
        return None  # pragma: no cover - loop always returns or raises

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "LangfuseHttpClient":  # pragma: no cover - convenience
        return self

    def __exit__(self, *_: Any) -> None:  # pragma: no cover - convenience
        self.close()
