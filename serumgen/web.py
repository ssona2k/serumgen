"""Local web UI: latent-space galaxy + live generation for demos."""
import base64
import binascii
import hmac
import io
import json
import os
import threading
import uuid
import webbrowser
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from sklearn.decomposition import PCA

from serumgen import generate as gen
from serumgen import importer
from serumgen.map import _COLORS, _encode_matrix
from serumgen.train import load_features

_STATIC = Path(__file__).resolve().parent / "static"

# Per-type projection + model cache, built lazily on first request.
# type -> {model, scaler, keys, meta, templates, pca, points}
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()

# download_id -> path of a generated .SerumPreset
_DOWNLOADS: dict[str, Path] = {}


def available_types(library_dir, models_dir) -> list[dict]:
    """Types that have both library presets and a trained model."""
    index = importer.load_index(Path(library_dir))
    counts: dict[str, int] = {}
    for entry in index.values():
        counts[entry["type"]] = counts.get(entry["type"], 0) + 1
    out = []
    for ptype in sorted(counts):
        if (Path(models_dir) / ptype / "meta.json").exists():
            out.append({"type": ptype, "count": counts[ptype],
                        "color": _COLORS.get(ptype, "#2a78d6"), "trained": True})
    return out


def get_projection(ptype, library_dir, models_dir) -> dict:
    """Encode a type's library, fit PCA(2) once, cache everything."""
    with _CACHE_LOCK:
        if ptype in _CACHE:
            return _CACHE[ptype]
        model, scaler, keys, meta = gen.load_model(models_dir, ptype)
        features = load_features(library_dir, ptype)
        names, latents, _ = _encode_matrix(features, model, scaler, keys)
        pca = PCA(n_components=2).fit(latents)
        coords = pca.transform(latents)
        points = [{"name": names[i],
                   "x": float(coords[i, 0]), "y": float(coords[i, 1])}
                  for i in range(len(names))]
        templates = gen._templates(library_dir, ptype)
        entry = {"model": model, "scaler": scaler, "keys": keys, "meta": meta,
                 "templates": templates,
                 "template_mus": gen._template_latents(templates, model, scaler, keys),
                 "pca": pca, "points": points,
                 "color": _COLORS.get(ptype, "#2a78d6")}
        _CACHE[ptype] = entry
        return entry


def generate_one(ptype, library_dir, models_dir, out_dir, temperature) -> dict:
    """Sample one preset, write it, project its latent through the cached PCA."""
    entry = get_projection(ptype, library_dir, models_dir)
    model, scaler, keys, meta = (entry["model"], entry["scaler"],
                                 entry["keys"], entry["meta"])
    z = gen.sample_near_library(entry["template_mus"], temperature)
    features = gen.decode_to_features(z, model, scaler, keys)

    out_dir = Path(out_dir) / ptype
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{ptype.capitalize()}_{uuid.uuid4().hex[:6]}"
    out_path = out_dir / f"{name}.SerumPreset"
    donor = gen._nearest_template(z.detach().numpy()[0], entry["templates"],
                                  entry["template_mus"])
    gen._write_preset(features, donor, out_path)

    coords = entry["pca"].transform(z.detach().numpy())
    download_id = uuid.uuid4().hex
    _DOWNLOADS[download_id] = out_path
    return {"name": name, "x": float(coords[0, 0]), "y": float(coords[0, 1]),
            "download_id": download_id, "donor": donor.stem}


_ZIP_MAX_IDS = 500


def zip_downloads(download_ids) -> tuple[bytes, int, set[str]]:
    """Bundle generated presets into one zip; returns (bytes, count, types).

    A single-type batch (the normal case: one section's presets) is flat;
    a mixed batch gets one folder per type. Only ids this server handed out
    are honoured (unknown ones are skipped), so the endpoint can't be used to
    read arbitrary files. Presets are already compressed binaries, so entries
    are stored, not deflated.
    """
    paths = []
    for download_id in list(download_ids)[:_ZIP_MAX_IDS]:
        path = _DOWNLOADS.get(download_id)
        if path and path.exists() and path not in paths:
            paths.append(path)
    types = {p.parent.name for p in paths}                    # out_dir/<type>/<name>
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for path in paths:
            z.write(path, path.name if len(types) == 1 else f"{path.parent.name}/{path.name}")
    return buf.getvalue(), len(paths), types


def zip_filename(types: set[str], stamp: str) -> str:
    """serumgen-bass-<stamp>.zip for one section, serumgen-session-… if mixed."""
    label = next(iter(types)) if len(types) == 1 else "session"
    return f"serumgen-{label}-{stamp}.zip"


def _static_svg(rel: str) -> Path | None:
    """Resolve a /static/ request to an .svg inside _STATIC, or None.

    Only SVG brand assets are exposed; anything that resolves outside the
    static folder (../ tricks) or isn't an existing .svg is refused.
    """
    root = _STATIC.resolve()
    path = (root / rel).resolve()
    if path.suffix != ".svg" or not path.is_file() or not path.is_relative_to(root):
        return None
    return path


_AUTH_USER = "serumgen"


class _Handler(BaseHTTPRequestHandler):
    # Injected by serve() before the server starts.
    library_dir = "data/library"
    models_dir = "data/models"
    out_dir = "data/generated"
    password = None  # None = auth disabled (local dev); set = required for every request

    def log_message(self, *args):  # keep the demo console quiet
        pass

    def _authorized(self) -> bool:
        """True if this request may proceed. Auth is off entirely when no
        password is configured (local dev stays exactly as before); once a
        password is set (e.g. on a VPS), every request needs it.
        """
        if self.password is None:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[len("Basic "):]).decode()
            user, _, pwd = decoded.partition(":")
        except (binascii.Error, UnicodeDecodeError):
            return False
        # constant-time comparison — a naive == leaks how many leading
        # characters matched via response-time differences
        return hmac.compare_digest(user, _AUTH_USER) and hmac.compare_digest(pwd, self.password)

    def _require_auth(self) -> bool:
        """Send 401 + WWW-Authenticate and return False if unauthorized."""
        if self._authorized():
            return True
        body = b"authentication required"
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="serumgen"')
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type, download_name=None):
        self._send_bytes(path.read_bytes(), content_type, download_name)

    def _send_bytes(self, data: bytes, content_type, download_name=None):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if download_name:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._require_auth():
            return
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/" or route == "/index.html":
                self._send_file(_STATIC / "index.html", "text/html; charset=utf-8")
            elif route.startswith("/static/"):
                path = _static_svg(unquote(route[len("/static/"):]))
                if path is None:
                    self._send_json({"error": "not found"}, 404)
                    return
                self._send_file(path, "image/svg+xml")
            elif route == "/api/types":
                self._send_json(available_types(self.library_dir, self.models_dir))
            elif route == "/api/library":
                ptype = parse_qs(parsed.query).get("type", [""])[0]
                entry = get_projection(ptype, self.library_dir, self.models_dir)
                self._send_json({"type": ptype, "color": entry["color"],
                                 "points": entry["points"]})
            elif route.startswith("/api/download/"):
                download_id = route.rsplit("/", 1)[-1]
                path = _DOWNLOADS.get(download_id)
                if not path or not path.exists():
                    self._send_json({"error": "unknown download"}, 404)
                    return
                self._send_file(path, "application/octet-stream", path.name)
            elif route == "/api/download-all":
                ids = parse_qs(parsed.query).get("ids", [""])[0].split(",")
                data, count, types = zip_downloads(i for i in ids if i)
                if not count:
                    self._send_json({"error": "nothing to download"}, 404)
                    return
                stamp = datetime.now().strftime("%Y%m%d-%H%M")
                self._send_bytes(data, "application/zip", zip_filename(types, stamp))
            else:
                self._send_json({"error": "not found"}, 404)
        except SystemExit as exc:  # user-facing one-liners from load_model etc.
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001 — demo server, report and stay up
            self._send_json({"error": repr(exc)}, 500)

    def do_POST(self):
        if not self._require_auth():
            return
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/generate":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                result = generate_one(
                    body["type"], self.library_dir, self.models_dir,
                    self.out_dir, body.get("temperature", 0.8))
                self._send_json(result)
            else:
                self._send_json({"error": "not found"}, 404)
        except SystemExit as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": repr(exc)}, 500)


def serve(library_dir="data/library", models_dir="data/models", out_dir="data/generated",
          port=8765, host="127.0.0.1", open_browser=True, password=None) -> None:
    """Start the web server.

    host: bind address. "127.0.0.1" (default) is local-machine-only —
    correct for `serumgen web` on a laptop. Pass "0.0.0.0" to accept
    connections from other machines (a VPS reachable from the internet)
    — do this only with `password` set, or SERUMGEN_PASSWORD in the
    environment, since the API is otherwise wide open to anyone who
    finds the address.
    password: shared Basic Auth password (username is always "serumgen").
    Falls back to the SERUMGEN_PASSWORD environment variable. Leave both
    unset to disable auth entirely (fine for 127.0.0.1-only local use;
    do not do this with host="0.0.0.0").
    """
    _Handler.library_dir = library_dir
    _Handler.models_dir = models_dir
    _Handler.out_dir = out_dir
    _Handler.password = password if password is not None else os.environ.get("SERUMGEN_PASSWORD")

    if host not in ("127.0.0.1", "localhost") and not _Handler.password:
        print(f"warning: binding to {host} with no password set — "
              f"the API is open to anyone who can reach this address. "
              f"Set SERUMGEN_PASSWORD or pass --password.")

    httpd = ThreadingHTTPServer((host, port), _Handler)
    local_url = f"http://127.0.0.1:{port}/"
    print(f"serumgen web running on {host}:{port}"
          + (f"  (open {local_url} locally)" if host != "127.0.0.1" else "")
          + "  (ctrl-c to stop)")
    if _Handler.password:
        print("  auth: required (Basic, user 'serumgen')")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(local_url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
