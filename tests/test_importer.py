import json

import httpx
import pytest
import typer
from typer.testing import CliRunner

from copy_traces import cli as copy_cli
from copy_traces.cli import _build_credentials
from copy_traces.client import LangfuseCredentials, LangfuseError, LangfuseHttpClient
from copy_traces.importer import (
    ImportStats,
    TraceFilter,
    TraceImporter,
    _decode_if_json,
    _first_model,
)
from langfuse_utils import cli as root_cli

META_EMPTY = {"page": 1, "limit": 1, "totalItems": 0, "totalPages": 1}
META_ONE = {"page": 1, "limit": 1, "totalItems": 1, "totalPages": 1}
CHAT_INPUT = [
    {"role": "system", "content": "Summarize the conversation."},
    {"role": "user", "content": {"question": "What is the status?"}},
]
CHAT_OUTPUT = {"role": "assistant", "content": {"answer": "All good"}}


def _make_client(responder: httpx.MockTransport) -> LangfuseHttpClient:
    credentials = LangfuseCredentials(
        url="http://langfuse.test",
        public_key="pk",
        secret_key="sk",
    )
    return LangfuseHttpClient(credentials, transport=responder)


def test_importer_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    created_items: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(404)
        if request.url.path == "/api/public/datasets" and request.method == "POST":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [
                    {
                        "id": "t1",
                        "input": CHAT_INPUT,
                        "output": CHAT_OUTPUT,
                        "environment": "production",
                    },
                ],
                "meta": {"page": 1, "limit": 100, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items" and request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        if request.url.path == "/api/public/dataset-items" and request.method == "POST":
            body = json.loads(request.content)
            created_items.append(body)
            return httpx.Response(200, json={"id": body["id"]})
        if request.url.path.startswith("/api/public/dataset-items/"):
            return httpx.Response(200, json={"id": request.url.path.split("/")[-1]})
        raise AssertionError(f"Unhandled request {request.method} {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1))
    stats = importer.run()

    assert stats.imported == 1
    assert stats.skipped == 0
    assert stats.failed == 0
    assert created_items[0]["sourceTraceId"] == "t1"
    assert created_items[0]["datasetName"] == "demo"
    assert created_items[0]["input"] == {"user_content": CHAT_INPUT[1]["content"]}
    assert created_items[0]["expectedOutput"] == CHAT_OUTPUT["content"]
    assert created_items[0]["metadata"] == {
        "traceId": "t1",
        "environment": "production",
        "prompt": CHAT_INPUT[0]["content"],
    }


def test_dry_run_never_writes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            raise AssertionError("dry-run should not perform POST requests")
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": CHAT_INPUT, "output": CHAT_OUTPUT}],
                "meta": {"page": 1, "limit": 100, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        raise AssertionError(f"Unhandled {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1), dry_run=True)
    stats = importer.run()
    assert stats.imported == 1
    assert stats.failed == 0
    assert stats.skipped == 0


def test_dry_run_missing_dataset_succeeds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(404)
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": CHAT_INPUT, "output": CHAT_OUTPUT}],
                "meta": {"page": 1, "limit": 100, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items":
            return httpx.Response(404)
        raise AssertionError(f"Unhandled {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1), dry_run=True)
    stats = importer.run()

    assert stats.imported == 1
    assert stats.failed == 0


def test_model_filter_fetches_details_and_decodes_payload() -> None:
    posted: dict | None = None
    encoded_input = json.dumps(json.dumps(CHAT_INPUT))
    encoded_output = json.dumps(json.dumps(CHAT_OUTPUT))

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posted
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": encoded_input, "output": encoded_output}],
                "meta": {"page": 1, "limit": 100, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/traces/t1":
            detail = {
                "id": "t1",
                "input": encoded_input,
                "output": encoded_output,
                "observations": [{"model": "gpt-m1"}],
            }
            return httpx.Response(200, json=detail)
        if request.url.path == "/api/public/dataset-items" and request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        if request.url.path == "/api/public/dataset-items" and request.method == "POST":
            posted = json.loads(request.content)
            return httpx.Response(200, json={"id": posted["id"]})
        if request.url.path.startswith("/api/public/dataset-items/"):
            return httpx.Response(200, json={"id": request.url.path.split("/")[-1]})
        raise AssertionError(f"Unhandled {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1, model="gpt-m1"))
    stats = importer.run()

    assert stats.imported == 1
    assert posted["input"] == {"user_content": CHAT_INPUT[1]["content"]}
    assert posted["expectedOutput"] == CHAT_OUTPUT["content"]
    assert posted["metadata"]["prompt"] == CHAT_INPUT[0]["content"]
    assert posted["metadata"]["model"] == "gpt-m1"


def test_skips_when_trace_already_imported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": CHAT_INPUT, "output": CHAT_OUTPUT}],
                "meta": {"page": 1, "limit": 100, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_ONE},
            )
        raise AssertionError(f"Unhandled {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1))
    stats = importer.run()
    assert stats.imported == 0
    assert stats.skipped == 1
    assert stats.failed == 0


def test_retry_on_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: dict[str, int] = {"create": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            page = request.url.params.get("page", "1")
            input_value = [
                {"role": "system", "content": f"prompt-{page}"},
                {"role": "user", "content": {"page": int(page)}},
            ]
            payload = {
                "data": [
                    {
                        "id": f"t{page}",
                        "input": input_value,
                        "output": {"content": {"status": "ok", "page": int(page)}},
                    },
                ],
                "meta": {"page": int(page), "limit": 1, "totalPages": 2, "totalItems": 2},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items" and request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        if request.url.path == "/api/public/dataset-items" and request.method == "POST":
            attempts["create"] += 1
            if attempts["create"] == 1:
                return httpx.Response(500, json={"error": "temporary"})
            return httpx.Response(200, json={"id": json.loads(request.content)["id"]})
        if request.url.path.startswith("/api/public/dataset-items/"):
            return httpx.Response(200, json={"id": request.url.path.split("/")[-1]})
        raise AssertionError(f"Unhandled {request.url}")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1))
    stats = importer.run()

    assert attempts["create"] == 3
    assert stats.imported == 2
    assert stats.failed == 0


def test_non_retriable_error_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": CHAT_INPUT, "output": CHAT_OUTPUT}],
                "meta": {"page": 1, "limit": 100, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items" and request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        if request.url.path == "/api/public/dataset-items" and request.method == "POST":
            return httpx.Response(400, json={"error": "bad payload"})
        raise AssertionError(f"Unhandled {request.url}")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1))
    stats = importer.run()
    assert stats.imported == 0
    assert stats.failed == 1
    assert stats.skipped == 0


def test_credentials_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LANGFUSE_BASE_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_PROJECT_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(typer.BadParameter):
        _build_credentials(None, None, None)

    monkeypatch.setenv("LANGFUSE_PROJECT_ID", "demo-project")
    creds = _build_credentials(
        url="http://host",
        public_key="pk",
        secret_key="sk",
    )
    assert creds.url == "http://host"
    assert creds.project_id == "demo-project"


def test_environment_parameter_and_missing_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            assert request.url.params.get("environment") == "staging"
            payload = {
                "data": [{"id": None, "input": "{}", "output": "{invalid"}],
                "meta": {"page": 1, "limit": 1, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        raise AssertionError(f"Unhandled {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1, environment="staging"))
    stats = importer.run()
    assert stats.failed == 1
    assert stats.imported == 0
    assert stats.skipped == 0


def test_build_dataset_item_strips_empty_metadata() -> None:
    importer = TraceImporter(
        _make_client(httpx.MockTransport(lambda request: httpx.Response(404))),
        "demo",
        TraceFilter(range_days=1),
    )
    trace = {
        "id": "t1",
        "input": CHAT_INPUT,
        "output": {"content": {"answer": "ok"}},
        "name": "",
        "environment": None,
        "timestamp": None,
        "latency": None,
        "observations": [],
    }

    payload = importer._build_dataset_item(trace)

    assert payload["metadata"] == {"traceId": "t1", "prompt": CHAT_INPUT[0]["content"]}
    assert payload["input"] == {"user_content": CHAT_INPUT[1]["content"]}
    assert payload["expectedOutput"] == {"answer": "ok"}


def test_model_filter_mismatch_skips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": CHAT_INPUT, "output": CHAT_OUTPUT}],
                "meta": {"page": 1, "limit": 1, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/traces/t1":
            detail = {
                "id": "t1",
                "input": "{}",
                "output": "ok",
                "observations": [{"model": "other"}],
            }
            return httpx.Response(200, json=detail)
        if request.url.path == "/api/public/dataset-items":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        raise AssertionError(f"Unhandled {request.url}")

    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1, model="target"))
    stats = importer.run()
    assert stats.skipped == 1
    assert stats.imported == 0
    assert stats.failed == 0


def test_http_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.TransportError("boom")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(LangfuseError):
        client._request("GET", "/traces")  # type: ignore[attr-defined]


def test_project_header_is_sent() -> None:
    seen_header: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_header
        seen_header = request.headers.get("X-Langfuse-Project")
        return httpx.Response(404)

    credentials = LangfuseCredentials(
        url="http://langfuse.test",
        public_key="pk",
        secret_key="sk",
        project_id="demo-project",
    )
    client = LangfuseHttpClient(credentials, transport=httpx.MockTransport(handler))
    with pytest.raises(LangfuseError):
        client._request("GET", "/traces")  # type: ignore[attr-defined]
    assert seen_header == "demo-project"


def test_root_cli_exposes_copy_traces_command() -> None:
    runner = CliRunner()
    result = runner.invoke(root_cli.app, ["copy-traces", "--help"])

    assert result.exit_code == 0
    assert "Copy traces into a dataset" in result.stdout


def test_root_cli_parses_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, credentials: LangfuseCredentials) -> None:
            captured["creds"] = credentials

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_: object) -> None:
            captured["exited"] = True

    class FakeImporter:
        def __init__(
            self,
            client: FakeClient,
            dataset_name: str,
            trace_filter: TraceFilter,
            dry_run: bool,
        ) -> None:
            captured["dataset"] = dataset_name
            captured["trace_filter"] = trace_filter
            captured["dry_run"] = dry_run

        def run(self) -> ImportStats:
            captured["ran"] = True
            return ImportStats(imported=0, skipped=0, failed=0)

    monkeypatch.setattr(copy_cli, "LangfuseHttpClient", FakeClient)
    monkeypatch.setattr(copy_cli, "TraceImporter", FakeImporter)

    env = {
        "LANGFUSE_BASE_URL": "http://host",
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
        "LANGFUSE_PROJECT_ID": "proj",
    }

    result = runner.invoke(
        root_cli.app,
        [
            "copy-traces",
            "assorted_summaries",
            "--range",
            "7",
            "--environment",
            "production",
            "--dry-run",
        ],
        env=env,
    )

    assert result.exit_code == 0
    assert captured["dataset"] == "assorted_summaries"
    assert captured["trace_filter"].range_days == 7
    assert captured["trace_filter"].environment == "production"
    assert captured["dry_run"] is True


def test_cli_invokes_importer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, credentials: LangfuseCredentials) -> None:
            captured["creds"] = credentials

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_: object) -> None:
            captured["exited"] = True

    class FakeImporter:
        def __init__(
            self,
            client: FakeClient,
            dataset_name: str,
            trace_filter: TraceFilter,
            dry_run: bool,
        ) -> None:
            captured["dataset"] = dataset_name
            captured["trace_filter"] = trace_filter
            captured["dry_run"] = dry_run

        def run(self) -> ImportStats:
            return ImportStats(imported=2, skipped=1, failed=0)

    monkeypatch.setattr(copy_cli, "LangfuseHttpClient", FakeClient)
    monkeypatch.setattr(copy_cli, "TraceImporter", FakeImporter)

    copy_cli.copy_traces(
        dataset="demo",
        range_days=3,
        environment=None,
        model=None,
        dry_run=True,
        url="http://host",
        public_key="pk",
        secret_key="sk",
    )

    assert isinstance(captured["creds"], LangfuseCredentials)
    assert captured["dataset"] == "demo"
    assert captured["dry_run"] is True


def test_import_stats_total_property() -> None:
    stats = ImportStats(imported=1, skipped=2, failed=3)
    assert stats.total == 6


def test_list_traces_invalid_shapes_raise() -> None:
    def handler_invalid_data(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": "oops", "meta": {"totalPages": 1}})

    client = _make_client(httpx.MockTransport(handler_invalid_data))
    with pytest.raises(LangfuseError):
        list(client.list_traces(from_timestamp="now", to_timestamp="later", environment=None))

    def handler_invalid_entry(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [123], "meta": {"totalPages": 1}})

    client = _make_client(httpx.MockTransport(handler_invalid_entry))
    with pytest.raises(LangfuseError):
        list(client.list_traces(from_timestamp="now", to_timestamp="later", environment=None))


def test_list_traces_happy_path_with_environment() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        assert request.url.params.get("environment") == "prod"
        payload = {
            "data": [{"id": f"t{page}"}],
            "meta": {"page": page, "limit": 1, "totalPages": 2, "totalItems": 2},
        }
        return httpx.Response(200, json=payload)

    client = _make_client(httpx.MockTransport(handler))
    traces = list(
        client.list_traces(from_timestamp="start", to_timestamp="end", environment="prod"),
    )
    assert [trace["id"] for trace in traces] == ["t1", "t2"]


def test_get_trace_requires_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/traces/abc":
            return httpx.Response(200, json=["not-a-dict"])
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(LangfuseError):
        client.get_trace("abc")


def test_dataset_item_exists_without_trace_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/dataset-items":
            assert "sourceTraceId" not in request.url.params
            return httpx.Response(
                200,
                json={"data": [], "meta": META_ONE},
            )
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    assert client.dataset_item_exists("demo", trace_id=None) is True


def test_dataset_item_exists_with_item_id_and_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/dataset-items/item-1":
            return httpx.Response(404)
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    assert client.dataset_item_exists("demo", item_id="item-1") is False


def test_dataset_item_exists_handles_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/dataset-items":
            return httpx.Response(404)
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    assert client.dataset_item_exists("demo", trace_id="missing") is False


def test_create_dataset_item_requires_dict() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not-a-dict"])

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(LangfuseError):
        client.create_dataset_item({"datasetName": "demo"})
    client.close()


def test_verification_failure_counts_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"name": "demo"})
        if request.url.path == "/api/public/traces":
            payload = {
                "data": [{"id": "t1", "input": CHAT_INPUT, "output": CHAT_OUTPUT}],
                "meta": {"page": 1, "limit": 1, "totalPages": 1, "totalItems": 1},
            }
            return httpx.Response(200, json=payload)
        if request.url.path == "/api/public/dataset-items" and request.method == "GET":
            return httpx.Response(
                200,
                json={"data": [], "meta": META_EMPTY},
            )
        if request.url.path == "/api/public/dataset-items" and request.method == "POST":
            return httpx.Response(200, json={"id": "item-1"})
        if request.url.path.startswith("/api/public/dataset-items/"):
            return httpx.Response(404)
        raise AssertionError(f"Unhandled {request.url}")

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _make_client(httpx.MockTransport(handler))
    importer = TraceImporter(client, "demo", TraceFilter(range_days=1))
    stats = importer.run()
    assert stats.failed == 1


def test_list_traces_with_total_paginates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        payload = {
            "data": [{"id": f"t{page}"}],
            "meta": {"page": page, "limit": 1, "totalPages": 2, "totalItems": 2},
        }
        return httpx.Response(200, json=payload)

    client = _make_client(httpx.MockTransport(handler))
    traces, total = client.list_traces_with_total(
        from_timestamp="start",
        to_timestamp="end",
        environment=None,
    )
    assert total == 2
    assert [trace["id"] for trace in traces] == ["t1", "t2"]


def test_list_traces_with_total_invalid_shapes_raise() -> None:
    def handler_invalid_data(request: httpx.Request) -> httpx.Response:
        payload = {"data": "oops", "meta": {"totalPages": 1, "totalItems": 1}}
        return httpx.Response(200, json=payload)

    client = _make_client(httpx.MockTransport(handler_invalid_data))
    with pytest.raises(LangfuseError):
        list(
            client.list_traces_with_total(
                from_timestamp="now",
                to_timestamp="later",
                environment=None,
            )[0],
        )

    def handler_invalid_second_page(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            payload = {"data": [{"id": "t1"}], "meta": {"totalPages": 2, "totalItems": 2}}
            return httpx.Response(200, json=payload)
        payload = {"data": "oops", "meta": {"totalPages": 2, "totalItems": 2}}
        return httpx.Response(200, json=payload)

    client = _make_client(httpx.MockTransport(handler_invalid_second_page))
    traces, _ = client.list_traces_with_total(
        from_timestamp="now",
        to_timestamp="later",
        environment=None,
    )
    with pytest.raises(LangfuseError):
        list(traces)

    def handler_invalid_second_page_entry(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        if page == 1:
            payload = {"data": [{"id": "t1"}], "meta": {"totalPages": 2, "totalItems": 2}}
            return httpx.Response(200, json=payload)
        payload = {"data": [123], "meta": {"totalPages": 2, "totalItems": 2}}
        return httpx.Response(200, json=payload)

    client = _make_client(httpx.MockTransport(handler_invalid_second_page_entry))
    traces, _ = client.list_traces_with_total(
        from_timestamp="now",
        to_timestamp="later",
        environment=None,
    )
    with pytest.raises(LangfuseError):
        list(traces)

    def handler_invalid_entry(request: httpx.Request) -> httpx.Response:
        payload = {"data": [123], "meta": {"totalPages": 1, "totalItems": 1}}
        return httpx.Response(200, json=payload)

    client = _make_client(httpx.MockTransport(handler_invalid_entry))
    with pytest.raises(LangfuseError):
        list(
            client.list_traces_with_total(
                from_timestamp="now",
                to_timestamp="later",
                environment=None,
            )[0],
        )


def test_ensure_dataset_existing_and_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(200, json={"id": "d1"})
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    status = client.ensure_dataset("demo")
    assert status.existed is True
    assert status.id == "d1"
    assert status.created is False


def test_ensure_dataset_dry_run_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(404)
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    status = client.ensure_dataset("demo", dry_run=True)
    assert status.dry_run_creation is True
    assert status.existed is False


def test_ensure_dataset_creates_when_missing() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/datasets/demo":
            return httpx.Response(404)
        if request.url.path == "/api/public/datasets" and request.method == "POST":
            seen["created"] = request.content.decode()
            return httpx.Response(200, json={"id": "d2"})
        raise AssertionError("unexpected request")

    client = _make_client(httpx.MockTransport(handler))
    status = client.ensure_dataset("demo", dry_run=False)
    assert status.created is True
    assert status.id == "d2"
    assert "created" in seen


def test_decode_if_json_encoded_scalar() -> None:
    encoded = '"plain"'
    assert _decode_if_json(encoded) == encoded


def test_decode_if_json_exhausts_loop_without_break() -> None:
    encoded = '"\\"inner\\""'
    assert _decode_if_json(encoded) == encoded


def test_decode_if_json_handles_decode_error() -> None:
    invalid = '{"foo":'  # looks like JSON but is malformed
    assert _decode_if_json(invalid) == invalid


def test_first_model_handles_missing_model() -> None:
    assert _first_model({"observations": [{"model": None}]}) is None
