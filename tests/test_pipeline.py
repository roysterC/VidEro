import csv
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from pipeline import dataset, export, generate
from pipeline.config import Persona, Preset, Shot, load_persona, load_shots
from pipeline.prompts import GuardrailError, build_prompts, check_persona, check_prompt
from pipeline.workflows import fill_template, sdxl_graph

ROOT = Path(__file__).resolve().parent.parent


def persona(**kw):
    base = dict(slug="t", name="T", age=26, identity="26 year old woman, green eyes", style="candid photo",
                trigger="tstx woman")
    base.update(kw)
    return Persona(**base)


class GuardrailTests(unittest.TestCase):
    def test_blocks_youth_terms(self):
        for text in ["a teen at the beach", "school uniform", "young girl smiling", "Teenage look",
                     "petite woman", "17 year old woman", "19yo", "18-year-old", "pre-teen"]:
            with self.subTest(text=text), self.assertRaises(GuardrailError):
                check_prompt(text)

    def test_blocks_likeness_terms(self):
        for text in ["looks like a famous singer", "celebrity lookalike", "resembling someone"]:
            with self.subTest(text=text), self.assertRaises(GuardrailError):
                check_prompt(text)

    def test_allows_adult_prompts(self):
        for text in ["26 year old woman", "woman in a cafe", "30yo woman", "sunglasses, girlfriend-free zone"]:
            with self.subTest(text=text):
                check_prompt(text)

    def test_persona_age_floor(self):
        with self.assertRaises(GuardrailError):
            check_persona(persona(age=19))

    def test_public_tier_adds_nsfw_negative(self):
        pos, neg = build_prompts(persona(), Shot(id="a", prompt="cafe"), use_lora=False)
        self.assertIn("nude", neg)
        self.assertIn("child", neg)
        self.assertNotIn("tstx", pos)
        _, neg_private = build_prompts(persona(), Shot(id="a", prompt="cafe", tier="private"), use_lora=True)
        self.assertNotIn("nude", neg_private)
        self.assertIn("child", neg_private)

    def test_trigger_only_with_lora(self):
        pos, _ = build_prompts(persona(), Shot(id="a", prompt="cafe"), use_lora=True)
        self.assertTrue(pos.startswith("tstx woman, 26 year old woman"))


class ConfigFileTests(unittest.TestCase):
    def test_repo_files_load_and_pass_guardrails(self):
        for pf in sorted((ROOT / "personas").glob("*.yaml")):
            p = load_persona(pf)
            check_persona(p)
            for f in sorted((ROOT / "shots").glob("*.yaml")):
                for shot in load_shots(f):
                    with self.subTest(persona=pf.name, file=f.name, shot=shot.id):
                        build_prompts(p, shot, use_lora=True)


class WorkflowTests(unittest.TestCase):
    preset = Preset(checkpoint="m.safetensors", steps=6, cfg=1.5, sampler="dpmpp_sde", scheduler="karras")

    def test_graph_links_resolve(self):
        hires = Preset(**{**self.preset.__dict__, "hires_scale": 1.5})
        for p, lora in [(self.preset, None), (hires, ("l.safetensors", 0.8))]:
            g = sdxl_graph(preset=p, positive="a", negative="b", width=832, height=1216, seed=5, prefix="x", lora=lora)
            for node in g.values():
                for v in node["inputs"].values():
                    if isinstance(v, list):
                        self.assertIn(v[0], g)
            self.assertEqual("hires" in g, p.hires_scale is not None)
            self.assertEqual("lora" in g, lora is not None)

    def test_repo_templates_load_and_link(self):
        for f in sorted((ROOT / "workflows").glob("*.json")):
            with self.subTest(file=f.name):
                g = fill_template(load_template(f), positive="P", negative="N", seed=7,
                                  width=768, height=1344, prefix="x")
                for node in g.values():
                    for v in node["inputs"].values():
                        if isinstance(v, list):
                            self.assertIn(v[0], g)
                texts = [n["inputs"]["text"] for n in g.values() if n["class_type"] == "CLIPTextEncode"]
                self.assertEqual(sorted(texts), ["N", "P"])
                self.assertIn(7, [n["inputs"].get("seed") for n in g.values()])

    def test_fill_template(self):
        t = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "%PROMPT%", "clip": ["9", 1]}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "%NEGATIVE%, extra", "clip": ["9", 1]}},
            "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 6}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI"}},
        }
        g = fill_template(t, positive="P", negative="N", seed=42, width=832, height=1216, prefix="pre")
        self.assertEqual(g["1"]["inputs"]["text"], "P")
        self.assertEqual(g["2"]["inputs"]["text"], "N, extra")
        self.assertEqual(g["3"]["inputs"]["seed"], 42)
        self.assertEqual((g["4"]["inputs"]["width"], g["4"]["inputs"]["height"]), (832, 1216))
        self.assertEqual(g["5"]["inputs"]["filename_prefix"], "pre")
        self.assertEqual(t["1"]["inputs"]["text"], "%PROMPT%")  # template untouched


def png_bytes(size=(832, 1216)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 90, 80)).save(buf, "PNG")
    return buf.getvalue()


class FakeComfy(BaseHTTPRequestHandler):
    queued = []

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/object_info/CheckpointLoaderSimple":
            self._send({"CheckpointLoaderSimple": {"input": {"required": {
                "ckpt_name": [["RealVisXL_V5.0_Lightning_fp16.safetensors"]]}}}})
        elif url.path.startswith("/history/"):
            pid = url.path.rsplit("/", 1)[1]
            self._send({pid: {"status": {"status_str": "success", "completed": True},
                              "outputs": {"save": {"images": [
                                  {"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}}})
        elif url.path == "/view":
            assert parse_qs(url.query)["filename"][0].endswith(".png")
            self._send(png_bytes(), "image/png")
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeComfy.queued.append(body["prompt"])
        self._send({"prompt_id": f"p{len(FakeComfy.queued)}", "number": 0, "node_errors": {}})


class EndToEndTests(unittest.TestCase):
    def test_generate_dataset_export(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = (ROOT / "config/render.yaml").read_text().replace(
                "http://127.0.0.1:8188", f"http://127.0.0.1:{server.server_address[1]}")
            (tmp / "render.yaml").write_text(cfg)
            out = tmp / "output"
            rc = generate.main(["--persona", str(ROOT / "personas/ava.yaml"),
                                "--shots", str(ROOT / "shots/01-casting.yaml"),
                                "--config", str(tmp / "render.yaml"), "--count", "2", "--out", str(out)])
            self.assertEqual(rc, 0)
            pngs = sorted((out / "ava/01-casting").glob("*.png"))
            self.assertEqual(len(pngs), 4)
            self.assertEqual(len(FakeComfy.queued), 4)
            with open(out / "ava/log.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertTrue(rows[0]["scene"].startswith("head and shoulders portrait"))

            ds = tmp / "ds"
            self.assertEqual(dataset.main(["--persona", str(ROOT / "personas/ava.yaml"),
                                           "--src", str(out / "ava/01-casting"), "--out", str(ds),
                                           "--log", str(out / "ava/log.csv")]), 0)
            caps = sorted((ds / "img/10_avlx woman").glob("*.txt"))
            self.assertEqual(len(caps), 4)
            self.assertTrue(caps[0].read_text().startswith("avlx woman, head and shoulders portrait"))

            ex = tmp / "ex"
            self.assertEqual(export.main(["--src", str(out / "ava/01-casting"), "--format", "feed", "story",
                                          "--out", str(ex)]), 0)
            with Image.open(next((ex / "feed").glob("*.jpg"))) as im:
                self.assertEqual(im.size, (1080, 1350))
                self.assertFalse(im.info.get("exif"))
            with Image.open(next((ex / "story").glob("*.jpg"))) as im:
                self.assertEqual(im.size, (1080, 1920))

    def test_guardrail_refuses_before_queueing(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("shots:\n  - {id: x, prompt: 'teen at the beach'}\n")
            before = len(FakeComfy.queued)
            rc = generate.main(["--persona", str(ROOT / "personas/ava.yaml"), "--shots", str(bad),
                                "--config", str(ROOT / "config/render.yaml"), "--out", tmp])
            self.assertEqual(rc, 2)
            self.assertEqual(len(FakeComfy.queued), before)


if __name__ == "__main__":
    unittest.main()
