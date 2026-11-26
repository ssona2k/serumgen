"""Generate presets from a trained per-type VAE."""
import json
import pickle
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from serumgen import extract, packager
from serumgen.vae import VAE


def load_model(models_dir: Path, ptype: str):
    model_dir = Path(models_dir) / ptype
    if not (model_dir / "meta.json").exists():
        raise SystemExit(f"no model for '{ptype}' — run: serumgen train {ptype}")
    meta = json.loads((model_dir / "meta.json").read_text())
    keys = json.loads((model_dir / "keys.json").read_text())
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = VAE(input_dim=meta["input_dim"], latent_dim=meta["latent_dim"])
    model.load_state_dict(torch.load(model_dir / "vae.pth"))
    model.eval()
    return model, scaler, keys, meta


def inject_features(data: dict, features: dict[str, float]) -> None:
    """Write feature values back into the nested module tree.

    Keys are ``/``-joined module paths (see extract.extract_numeric_features):
    walk the module chain through nested dicts, then set the leaf param inside
    that node's ``plainParams`` — only where the full path already exists.
    """
    root = data.get("data", {})
    for name, value in features.items():
        segments = name.split("/")
        node = root
        for seg in segments[:-1]:
            node = node.get(seg) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                break
        else:
            params = node.get("plainParams") if isinstance(node, dict) else None
            leaf = segments[-1]
            if isinstance(params, dict) and leaf in params:
                params[leaf] = float(value)


def _features_to_vector(features: dict[str, float], keys, scaler) -> torch.Tensor:
    raw = np.array([[features.get(k, 0.0) for k in keys]], dtype=np.float32)
    return torch.FloatTensor(scaler.transform(raw))


def encode_preset(preset_path: Path, model, scaler, keys) -> torch.Tensor:
    data = packager.load(preset_path)
    features = extract.extract_numeric_features(data)
    with torch.no_grad():
        mu, _ = model.encode(_features_to_vector(features, keys, scaler))
    return mu


def decode_to_features(z, model, scaler, keys) -> dict[str, float]:
    with torch.no_grad():
        normalized = model.decode(z).numpy()
    raw = scaler.inverse_transform(normalized)[0]
    return {keys[i]: float(raw[i]) for i in range(len(keys))}


def _templates(library_dir: Path, ptype: str) -> list[Path]:
    templates = sorted((Path(library_dir) / ptype).glob("*.SerumPreset"))
    if not templates:
        raise SystemExit(
            f"no '{ptype}' presets in library — run: serumgen import <folder>")
    return templates


def _template_latents(templates, model, scaler, keys):
    """Encode each template to its latent mean, from cached features if present."""
    vectors = []
    for t in templates:
        cache = t.with_suffix(".json")
        if cache.exists():
            feats = json.loads(cache.read_text())
        else:
            feats = extract.extract_numeric_features(packager.load(t))
        vectors.append([feats.get(k, 0.0) for k in keys])
    x = torch.FloatTensor(scaler.transform(np.array(vectors, dtype=np.float32)))
    with torch.no_grad():
        mu, _ = model.encode(x)
    return mu.numpy()


def _nearest_template(z_np, templates, template_mus):
    """Template whose latent mean is closest to z — inherits a congruent wavetable."""
    dists = ((template_mus - z_np) ** 2).sum(axis=1)
    return templates[int(dists.argmin())]


# Noise per unit temperature, as a fraction of the library's own per-dim spread.
# 0.5 → T=0.2 is a close variation of a real preset, T=1.8 wanders ~1 spread out.
_ANCHOR_SPREAD = 0.5


def sample_near_library(template_mus, temperature, rng=None) -> torch.Tensor:
    """Sample a latent near the real presets rather than from the N(0, I) prior.

    The trained VAEs show heavy posterior collapse: most latent dims barely
    vary across the library (std ~0.03) while 1-3 dims carry nearly all the
    signal. `randn * T` puts prior-sized noise into every dim, so even T=0.2
    lands ~80x farther from any real preset than presets sit from each other.
    Instead: pick a random library preset as an anchor (so dense regions are
    sampled more, matching the data) and add noise scaled per dim by the
    library's own spread, times temperature.
    """
    rng = rng if rng is not None else np.random.default_rng()
    mus = np.asarray(template_mus, dtype=np.float64)
    anchor = mus[rng.integers(len(mus))]
    noise = rng.standard_normal(mus.shape[1]) * mus.std(axis=0)
    z = anchor + noise * _ANCHOR_SPREAD * float(temperature)
    return torch.from_numpy(z.astype(np.float32)).unsqueeze(0)


_AUTHOR = "audiomimi experimental labs"


def _sanitize_metadata(metadata: dict, name: str) -> None:
    """Strip the donor template's identity out of a generated preset's
    metadata, so nothing from the original file survives — only the
    format/schema fields (fileType/product/productVersion/version) are
    left alone, since Serum needs those to load the file.
    """
    metadata["presetAuthor"] = _AUTHOR
    metadata["presetName"] = name
    metadata["presetDescription"] = f"Experimental Neural Network Sound Design — {_AUTHOR}"
    metadata["vendor"] = _AUTHOR
    metadata["url"] = ""
    metadata["tags"] = []
    metadata["hash"] = ""


def _write_preset(features, template: Path, out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        packager.unpack(template, tmp_path)
        data = json.loads(tmp_path.read_text())
        inject_features(data, features)
        _sanitize_metadata(data["metadata"], out_path.stem)
        tmp_path.write_text(json.dumps(data))
        packager.pack(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def generate(ptype, library_dir, models_dir, out_dir, n=5, temperature=0.8,
             template=None, like=None, spread=0.3) -> list[Path]:
    model, scaler, keys, meta = load_model(models_dir, ptype)
    templates = library = _templates(library_dir, ptype)
    if template:
        matches = [t for t in templates if template.lower() in t.name.lower()]
        if not matches:
            raise SystemExit(f"no template matching '{template}' in library/{ptype}")
        templates = matches
    center = None
    if like:
        center = encode_preset(Path(like), model, scaler, keys)

    library_mus = _template_latents(library, model, scaler, keys)
    # --template only narrows which presets may donate wavetables; sampling
    # still anchors on (and takes its spread from) the whole library.
    template_mus = (library_mus if templates is library
                    else _template_latents(templates, model, scaler, keys))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(1, n + 1):
        if center is not None:
            z = center + torch.randn(1, meta["latent_dim"]) * spread
        else:
            z = sample_near_library(library_mus, temperature)
        features = decode_to_features(z, model, scaler, keys)
        donor = _nearest_template(z.detach().numpy()[0], templates, template_mus)
        out_path = out_dir / (
            f"{ptype.capitalize()}_{i:03d}_{uuid.uuid4().hex[:4]}.SerumPreset")
        _write_preset(features, donor, out_path)
        print(f"wrote {out_path}")
        written.append(out_path)
    return written


def interpolate(ptype, library_dir, models_dir, out_dir,
                preset_a, preset_b, steps=5) -> list[Path]:
    model, scaler, keys, meta = load_model(models_dir, ptype)
    mu_a = encode_preset(Path(preset_a), model, scaler, keys)
    mu_b = encode_preset(Path(preset_b), model, scaler, keys)
    templates = _templates(library_dir, ptype)
    template_mus = _template_latents(templates, model, scaler, keys)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(steps):
        t = i / max(1, steps - 1)
        z = mu_a * (1 - t) + mu_b * t
        features = decode_to_features(z, model, scaler, keys)
        donor = _nearest_template(z.detach().numpy()[0], templates, template_mus)
        out_path = out_dir / (
            f"{ptype.capitalize()}_morph_{i + 1:02d}of{steps}.SerumPreset")
        _write_preset(features, donor, out_path)
        print(f"wrote {out_path}  (t={t:.2f})")
        written.append(out_path)
    return written
