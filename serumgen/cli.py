"""serumgen command-line interface."""
import argparse
import json
import webbrowser
from pathlib import Path

from serumgen import generate as gen
from serumgen import importer, train
from serumgen import map as latent_map

DEFAULT_LIBRARY = Path("data/library")
DEFAULT_MODELS = Path("data/models")
DEFAULT_GENERATED = Path("data/generated")


def _cmd_import(args) -> int:
    stats = importer.import_folder(args.folder, args.library_dir)
    print(f"imported {stats['imported']}  "
          f"dupes {stats['skipped_dupes']}  failed {stats['failed']}")
    for ptype, count in sorted(stats["by_type"].items()):
        print(f"  {ptype:8s} {count}")
    if stats["fxp_found"]:
        print(f"note: {stats['fxp_found']} legacy .fxp files found — "
              f"re-save them as .SerumPreset via the Serum UI to import")
    return 0


def _cmd_library(args) -> int:
    lib = Path(args.library_dir)
    index = importer.load_index(lib)
    if not index:
        print("library is empty — run: serumgen import <folder>")
        return 0
    by_type: dict[str, int] = {}
    for entry in index.values():
        by_type[entry["type"]] = by_type.get(entry["type"], 0) + 1
    print(f"{'type':10s} {'presets':>8s} {'model':>12s}")
    for ptype in sorted(by_type):
        meta_path = Path(args.models_dir) / ptype / "meta.json"
        if meta_path.exists():
            trained = json.loads(meta_path.read_text())["trained_at"][:10]
        else:
            trained = "—"
        print(f"{ptype:10s} {by_type[ptype]:>8d} {trained:>12s}")
    print(f"{'total':10s} {len(index):>8d}")
    return 0


def _cmd_extract(args) -> int:
    """Rebuild param JSON caches for a type (after extraction logic changes)."""
    from serumgen import extract as ex
    from serumgen import packager
    type_dir = Path(args.library_dir) / args.type
    presets = sorted(type_dir.glob("*.SerumPreset"))
    if not presets:
        raise SystemExit(f"no '{args.type}' presets in library — "
                         f"run: serumgen import <folder>")
    for p in presets:
        feats = ex.extract_numeric_features(packager.load(p))
        p.with_suffix(".json").write_text(json.dumps(feats))
    print(f"rebuilt {len(presets)} param caches in {type_dir}/")
    return 0


def _cmd_train(args) -> int:
    train.train_type(args.type, args.library_dir, args.models_dir,
                     epochs=args.epochs, latent_dim=args.latent_dim,
                     force=args.force)
    return 0


def _cmd_generate(args) -> int:
    out_dir = Path(args.out_dir) / args.type
    gen.generate(args.type, args.library_dir, args.models_dir, out_dir,
                 n=args.n, temperature=args.temperature,
                 template=args.template, like=args.like, spread=args.spread)
    return 0


def _cmd_interpolate(args) -> int:
    out_dir = Path(args.out_dir) / args.type
    gen.interpolate(args.type, args.library_dir, args.models_dir, out_dir,
                    args.preset_a, args.preset_b, steps=args.steps)
    return 0


def _cmd_map(args) -> int:
    out = latent_map.build_map(
        args.type, args.library_dir, args.models_dir,
        Path(args.out_dir) / args.type,
        args.output or f"map_{args.type}.html",
        method="tsne" if args.tsne else "pca")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _cmd_web(args) -> int:
    from serumgen import web
    web.serve(library_dir=args.library_dir, models_dir=args.models_dir,
              out_dir=args.out_dir, port=args.port, host=args.host,
              open_browser=not args.no_open, password=args.password)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="serumgen",
        description="Serum preset dataset builder and ML generator")
    # Parent parser with global arguments
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--library-dir", default=DEFAULT_LIBRARY)
    global_parser.add_argument("--models-dir", default=DEFAULT_MODELS)
    global_parser.add_argument("--out-dir", default=DEFAULT_GENERATED,
                               help="root folder for generated presets")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="scan a folder into the library",
                       parents=[global_parser])
    p.add_argument("folder")
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("library", help="show library and model status",
                       parents=[global_parser])
    p.set_defaults(func=_cmd_library)

    p = sub.add_parser("extract", help="rebuild param caches for a type",
                       parents=[global_parser])
    p.add_argument("type")
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("train", help="train the VAE for a type",
                       parents=[global_parser])
    p.add_argument("type")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--force", action="store_true",
                   help="train even with < 40 presets")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("generate", help="generate new presets",
                       parents=[global_parser])
    p.add_argument("type")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="how far from real library presets to wander: ~0.2 is a "
                        "close variation, ~1.8 explores beyond the clusters "
                        "(default 0.8)")
    p.add_argument("--template", help="substring match on template preset name")
    p.add_argument("--like", help="path to a preset to generate variations of")
    p.add_argument("--spread", type=float, default=0.3,
                   help="variation amount around --like")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("interpolate", help="morph between two presets",
                       parents=[global_parser])
    p.add_argument("type")
    p.add_argument("preset_a")
    p.add_argument("preset_b")
    p.add_argument("--steps", type=int, default=5)
    p.set_defaults(func=_cmd_interpolate)

    p = sub.add_parser("map", help="latent-space HTML map",
                       parents=[global_parser])
    p.add_argument("type")
    p.add_argument("--output", help="output HTML path")
    p.add_argument("--tsne", action="store_true", help="t-SNE instead of PCA")
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=_cmd_map)

    p = sub.add_parser("web", help="launch the browser demo UI",
                       parents=[global_parser])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address — use 0.0.0.0 to accept connections "
                        "from other machines (e.g. a VPS); set a password "
                        "when you do")
    p.add_argument("--password",
                   help="Basic Auth password (user 'serumgen'); falls back "
                        "to the SERUMGEN_PASSWORD environment variable")
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=_cmd_web)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
