"""Scan preset folders, classify by type, build the library."""
import hashlib
import json
import re
import shutil
from pathlib import Path

from serumgen import extract, packager

_ALIASES = {
    "bass": "bass", "basses": "bass", "808": "bass", "sub": "bass",
    "lead": "lead", "leads": "lead",
    "pad": "pad", "pads": "pad",
    "pluck": "pluck", "plucks": "pluck",
    "keys": "keys", "key": "keys", "keyboard": "keys", "piano": "keys",
    "e piano": "keys", "organ": "keys", "bell": "keys", "mallet": "keys",
    "arp": "arp", "arps": "arp",
    "seq": "seq", "sequence": "seq",
    "chord": "chord", "chords": "chord",
    "vox": "vox", "vocal": "vox", "voice": "vox",
    "fx": "fx", "sfx": "fx",
    "drum": "drum", "drums": "drum", "perc": "drum",
}

_PREFIXES = {
    "ba": "bass", "bs": "bass", "ld": "lead", "pd": "pad", "pl": "pluck",
    "ky": "keys", "ar": "arp", "sq": "seq", "ch": "chord", "vx": "vox",
    "fx": "fx", "dr": "drum",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def classify(path: Path, metadata: dict | None = None) -> str:
    # 1. folder names, nearest parent first
    for parent in path.parents:
        name = parent.name.lower()
        if name in _ALIASES:
            return _ALIASES[name]
    stem_tokens = _tokens(path.stem)
    # 2. filename prefix (BA, LD, ...)
    if stem_tokens and stem_tokens[0] in _PREFIXES:
        return _PREFIXES[stem_tokens[0]]
    # 3. filename keyword anywhere
    for tok in stem_tokens:
        if tok in _ALIASES:
            return _ALIASES[tok]
    # 3b. prefix code anywhere (e.g. XX_YY_BS_..., where the type code is
    #     not the first token)
    for tok in stem_tokens:
        if tok in _PREFIXES:
            return _PREFIXES[tok]
    # 4. metadata tags / description
    if metadata:
        meta_text = " ".join(
            str(v) for v in [metadata.get("presetDescription", "")]
            + list(metadata.get("tags") or [])
        )
        for tok in _tokens(meta_text):
            if tok in _ALIASES:
                return _ALIASES[tok]
    return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(library_dir: Path) -> dict:
    index_path = library_dir / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {}


def import_folder(src_dir: Path, library_dir: Path) -> dict:
    src_dir, library_dir = Path(src_dir), Path(library_dir)
    library_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(library_dir)
    stats = {"imported": 0, "skipped_dupes": 0, "failed": 0,
             "fxp_found": 0, "by_type": {}}

    stats["fxp_found"] = sum(1 for _ in src_dir.rglob("*.fxp"))

    for preset_path in sorted(src_dir.rglob("*.SerumPreset")):
        digest = _sha256(preset_path)
        if digest in index:
            stats["skipped_dupes"] += 1
            continue
        try:
            data = packager.load(preset_path)
            features = extract.extract_numeric_features(data)
        except Exception as exc:  # corrupt preset: warn, keep going
            print(f"warning: could not unpack {preset_path.name}: {exc}")
            stats["failed"] += 1
            continue
        ptype = classify(preset_path.relative_to(src_dir), data.get("metadata"))
        author = extract.extract_author(data, preset_path.name)

        type_dir = library_dir / ptype
        type_dir.mkdir(exist_ok=True)
        dest = type_dir / preset_path.name
        if dest.exists():  # same name, different content
            dest = type_dir / f"{preset_path.stem}_{digest[:8]}.SerumPreset"
        shutil.copy2(preset_path, dest)
        params_path = dest.with_suffix(".json")
        params_path.write_text(json.dumps(features))

        index[digest] = {
            "name": preset_path.stem,
            "type": ptype,
            "author": author,
            "source": str(preset_path),
            "preset": str(dest.relative_to(library_dir)),
            "params": str(params_path.relative_to(library_dir)),
        }
        stats["imported"] += 1
        stats["by_type"][ptype] = stats["by_type"].get(ptype, 0) + 1

    (library_dir / "index.json").write_text(json.dumps(index, indent=1))
    return stats
