"""Latent-space HTML map: training presets, generated overlay, outliers."""
import html
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from serumgen import extract, packager
from serumgen.generate import load_model
from serumgen.train import load_features

_COLORS = {"bass": "#2a78d6", "lead": "#1baf7a", "pad": "#eda100",
           "pluck": "#4a3aa7", "keys": "#e87ba4", "arp": "#0ca30c",
           "seq": "#eb6834", "chord": "#d55181", "vox": "#199e70",
           "fx": "#888780", "drum": "#854f0b", "unknown": "#b4b2a9"}


def _encode_matrix(features_by_name, model, scaler, keys):
    raw = np.array(
        [[features_by_name[n].get(k, 0.0) for k in keys]
         for n in sorted(features_by_name)],
        dtype=np.float32,
    )
    x = torch.FloatTensor(scaler.transform(raw))
    with torch.no_grad():
        mu, _ = model.encode(x)
        recon = model.decode(mu)
        errors = ((recon - x) ** 2).mean(dim=1).numpy()
    return sorted(features_by_name), mu.numpy(), errors


def build_map(ptype, library_dir, models_dir, generated_dir, out_path,
              method="pca") -> Path:
    model, scaler, keys, meta = load_model(models_dir, ptype)
    features = load_features(library_dir, ptype)
    names, latents, errors = _encode_matrix(features, model, scaler, keys)

    gen_names, gen_latents = [], np.zeros((0, meta["latent_dim"]))
    gen_dir = Path(generated_dir)
    if gen_dir.is_dir():
        gen_feats = {}
        for p in sorted(gen_dir.glob("*.SerumPreset")):
            gen_feats[p.stem] = extract.extract_numeric_features(packager.load(p))
        if gen_feats:
            gen_names, gen_latents, _ = _encode_matrix(
                gen_feats, model, scaler, keys)

    stacked = np.vstack([latents, gen_latents])
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30, max(2, len(stacked) - 1))
        coords = TSNE(n_components=2, perplexity=perplexity,
                      random_state=0).fit_transform(stacked)
    else:
        coords = PCA(n_components=2).fit_transform(stacked)

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    scaled = 40 + (coords - lo) / span * np.array([720, 420])

    color = _COLORS.get(ptype, "#2a78d6")
    dots = []
    for i, name in enumerate(names):
        x, y = scaled[i]
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
            f'opacity="0.8"><title>{html.escape(name)}</title></circle>')
    for j, name in enumerate(gen_names):
        x, y = scaled[len(names) + j]
        dots.append(
            f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" '
            f'fill="none" stroke="#e34948" stroke-width="2" '
            f'transform="rotate(45 {x:.1f} {y:.1f})" class="generated">'
            f'<title>{html.escape(name)} (generated)</title></rect>')

    worst = sorted(zip(names, errors), key=lambda p: -p[1])[:10]
    rows = "".join(
        f"<tr><td>{html.escape(n)}</td><td>{e:.4f}</td></tr>"
        for n, e in worst)

    out_path = Path(out_path)
    out_path.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>serumgen map — {ptype}</title>
<style>
 body {{ font: 14px/1.5 -apple-system, sans-serif; margin: 2rem; color: #222; }}
 svg {{ background: #f7f7f5; border-radius: 8px; }}
 table {{ border-collapse: collapse; margin-top: .5rem; }}
 td {{ padding: 2px 14px 2px 0; font-family: ui-monospace, monospace; font-size: 12px; }}
</style></head><body>
<h2>serumgen map — {ptype}</h2>
<p>{len(names)} presets · {len(gen_names)} generated · {method} of
{meta["latent_dim"]}D latent · hover a dot for the preset name</p>
<svg viewBox="0 0 800 500" width="100%">{"".join(dots)}</svg>
<h3>Worst reconstruction — review these presets</h3>
<table>{rows}</table>
</body></html>""")
    print(f"map written to {out_path}")
    return out_path
