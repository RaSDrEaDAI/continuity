"""T13 contract regression tests — ComfySdkEndpoint against a stubbed SDK.

Offline and deterministic: the SDK classes are replaced by stubs that record
calls, so these tests pin OUR contract (workflow dict -> WorkflowFactory,
submit -> job id, poll maps terminal statuses, outputs described without
bytes, downloads go through Output.to_file) without network or GPU.

The REAL end-to-end proof (comfy-api-proxy 0.1.6 + comfy-sdk 0.5.0 driving a
live ComfyUI workflow) is a separate trace artifact, not these tests.
"""

from __future__ import annotations

import pytest

from continuity.api.sdk_endpoint import ComfySdkEndpoint


class _Out:
    def __init__(self, node_id="9", name="img.png", type="image",
                 oid="asset-1", size_bytes=1234, content_type="image/png"):
        self.node_id = node_id
        self.name = name
        self.type = type
        self.id = oid
        self.size_bytes = size_bytes
        self.content_type = content_type
        self.saved_to = None

    def to_file(self, path):
        self.saved_to = path
        with open(path, "wb") as fh:
            fh.write(b"png-bytes")


class _LowJob:
    def __init__(self, status, outputs):
        self.status = status
        self.outputs = outputs
        self.error = {"message": "boom"} if status == "failed" else None


class _Job:
    def __init__(self, low, model):
        self._low = low
        self._model = model

    @property
    def status(self):
        return self._model.status

    @property
    def outputs(self):
        return self._model.outputs

    @property
    def error(self):
        return self._model.error


class _SubmittedJob:
    id = "job-42"


class _Client:
    def __init__(self, *, api_key=None):
        self.api_key = api_key
        self.submitted = None

    def submit(self, wf):
        self.submitted = wf
        return _SubmittedJob()

    def close(self):
        pass


class _Low:
    def __init__(self, base_url, api_key=None, **kw):
        self.base_url = base_url
        self.api_key = api_key
        self.jobs = {"job-42": _LowJob("success", [_Out()])}

    def list_jobs(self):
        return []

    def get_job(self, jid):
        return self.jobs[jid]


class _WF:
    def __init__(self, graph):
        self.graph = graph


class _WFFactory:
    @staticmethod
    def from_json(graph):
        return _WF(graph)


def _endpoint(tmp_path, status="success"):
    ep = ComfySdkEndpoint(base_url="http://127.0.0.1:8190", api_key="k-TEST",
                          client=_Client, low_client=_Low, workflow_factory=_WFFactory,
                          job_cls=_Job)
    return ep


def test_queue_converts_dict_via_workflow_factory_and_returns_job_id():
    ep = _endpoint(None)
    jid = ep.queue({"1": {"class_type": "EmptyLatentImage",
                          "inputs": {"width": 64, "height": 64, "batch_size": 1}}})
    assert jid == "job-42"
    assert ep._client.submitted.graph["1"]["class_type"] == "EmptyLatentImage"


def test_health_reports_transport_and_url():
    ep = _endpoint(None)
    h = ep.health()
    assert h["ok"] is True
    assert h["transport"] == "comfy-api-v2"
    assert h["base_url"] == "http://127.0.0.1:8190"


def test_poll_maps_terminal_success_and_describes_outputs_without_bytes():
    ep = _endpoint(None)
    r = ep.poll("job-42", timeout_s=2, poll_s=0.01)
    assert r["status"] == "success"
    assert r["job_id"] == "job-42"
    assert r["outputs"] == [{
        "node_id": "9", "name": "img.png", "type": "image",
        "id": "asset-1", "size_bytes": 1234, "content_type": "image/png",
    }]


def test_poll_maps_failure_and_carries_error():
    ep = _endpoint(None)
    ep._ensure()
    ep._low.jobs["job-42"] = _LowJob("failed", [])
    r = ep.poll("job-42", timeout_s=2, poll_s=0.01)
    assert r["status"] == "error"
    assert r["error"] == {"message": "boom"}


def test_poll_times_out_on_nonterminal():
    ep = _endpoint(None)
    ep._ensure()
    ep._low.jobs["job-42"] = _LowJob("running", [])
    with pytest.raises(TimeoutError):
        ep.poll("job-42", timeout_s=0.2, poll_s=0.05)


def test_download_outputs_uses_output_tofile(tmp_path):
    ep = _endpoint(None)
    paths = ep.download_outputs("job-42", str(tmp_path))
    assert len(paths) == 1
    assert paths[0].endswith("img.png")
    with open(paths[0], "rb") as fh:
        assert fh.read() == b"png-bytes"
