"""Flatten packager JSON into numeric ML features."""
import re


def extract_numeric_features(data: dict) -> dict[str, float]:
    """Flatten every numeric ``plainParams`` value in the module tree.

    Recurses through nested dict sub-modules so oscillator-engine params
    (wavetable position, warp, ...) are captured, not just the top level.
    Keys are the module path joined with ``/`` and the leaf param name, e.g.
    ``Oscillator0/WTOsc0/kParamTablePos``. Lists (FX racks) are skipped —
    their positional, type-keyed params balloon dimensionality without
    proportional benefit.
    """
    features: dict[str, float] = {}

    def walk(node, path: list[str]) -> None:
        if not isinstance(node, dict):
            return
        params = node.get("plainParams")
        if isinstance(params, dict):
            for param_name, value in params.items():
                if isinstance(value, (int, float, bool)):
                    features["/".join(path + [param_name])] = float(value)
        for key, child in node.items():
            if key == "plainParams":
                continue
            if isinstance(child, dict):
                walk(child, path + [key])

    walk(data.get("data", {}), [])
    return features


def extract_author(data: dict, filename: str) -> str:
    author = data.get("metadata", {}).get("presetAuthor", "")
    if author:
        return author
    match = re.search(r"\[([^\]]+)\]", filename)
    return match.group(1) if match else ""
