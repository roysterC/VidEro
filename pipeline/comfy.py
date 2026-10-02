"""Minimal client for the ComfyUI HTTP API (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 900):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _request(self, path: str, body: dict | None = None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"} if data else {}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"cannot reach ComfyUI at {self.base_url} ({e.reason}). Is it running?") from e

    def _json(self, path: str, body: dict | None = None):
        return json.loads(self._request(path, body))

    def list_options(self, node_class: str, input_name: str) -> list[str]:
        """Values ComfyUI offers for a dropdown, e.g. installed checkpoints or LoRAs."""
        info = self._json(f"/object_info/{node_class}")
        spec = info[node_class]["input"]["required"][input_name]
        options = spec[0]
        if isinstance(options, str) and len(spec) > 1:  # newer "COMBO" format
            options = spec[1].get("options", [])
        return list(options)

    def queue(self, graph: dict) -> str:
        resp = self._json("/prompt", {"prompt": graph, "client_id": self.client_id})
        if resp.get("node_errors"):
            raise ComfyError(f"workflow rejected: {json.dumps(resp['node_errors'])[:2000]}")
        return resp["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 1.0) -> dict:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            hist = self._json(f"/history/{prompt_id}")
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(f"render failed: {json.dumps(status.get('messages', []))[:2000]}")
                if status.get("completed", True):
                    return entry
            time.sleep(poll)
        raise ComfyError(f"timed out after {self.timeout}s waiting for {prompt_id}")

    @staticmethod
    def output_images(entry: dict) -> list[dict]:
        images = []
        for out in entry.get("outputs", {}).values():
            images += [i for i in out.get("images", []) if i.get("type") == "output"]
        return images

    def download(self, image: dict) -> bytes:
        q = urllib.parse.urlencode(
            {"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")}
        )
        return self._request(f"/view?{q}")
