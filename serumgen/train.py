"""Train a per-type VAE from library param caches."""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from serumgen.vae import VAE, vae_loss


def load_features(library_dir: Path, ptype: str) -> dict[str, dict[str, float]]:
    type_dir = Path(library_dir) / ptype
    files = sorted(type_dir.glob("*.json")) if type_dir.is_dir() else []
    if not files:
        raise SystemExit(
            f"no '{ptype}' presets in library — run: serumgen import <folder>")
    return {f.stem: json.loads(f.read_text()) for f in files}


def build_matrix(features_by_name: dict[str, dict[str, float]]):
    """Union of keys across presets, missing values filled with 0.0."""
    all_keys = sorted({k for feats in features_by_name.values() for k in feats})
    names = sorted(features_by_name)
    matrix = np.array(
        [[features_by_name[n].get(k, 0.0) for k in all_keys] for n in names],
        dtype=np.float32,
    )
    return names, all_keys, matrix


def train_type(ptype: str, library_dir: Path, models_dir: Path,
               epochs: int = 500, latent_dim: int = 16,
               min_presets: int = 40, force: bool = False) -> Path:
    features = load_features(library_dir, ptype)
    if len(features) < min_presets and not force:
        raise SystemExit(
            f"only {len(features)} '{ptype}' presets (< {min_presets}) — "
            f"import more or use --force")

    _, all_keys, raw = build_matrix(features)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(raw)
    tensors = torch.FloatTensor(normalized)
    loader = DataLoader(TensorDataset(tensors), batch_size=16, shuffle=True)

    model = VAE(input_dim=len(all_keys), latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        beta = min(1.0, epoch / max(1, epochs * 0.5))
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch}/{epochs}  loss {total / len(tensors):.4f}")

    out_dir = Path(models_dir) / ptype
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "vae.pth")
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (out_dir / "keys.json").write_text(json.dumps(all_keys, indent=1))
    (out_dir / "meta.json").write_text(json.dumps({
        "latent_dim": latent_dim,
        "input_dim": len(all_keys),
        "epochs": epochs,
        "num_presets": len(features),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=1))
    print(f"model saved to {out_dir}/")
    return out_dir
