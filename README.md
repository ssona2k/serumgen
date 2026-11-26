# SerumGen

Train a neural network on **your own** [Serum 2](https://xferrecords.com/products/serum-2)
presets and generate new, playable ones.

SerumGen imports a folder of `.SerumPreset` files, sorts them by sound type
(bass, lead, pad, pluck, …), trains a small Variational Autoencoder (VAE) per
type, and samples that model to write brand-new presets you can load straight
into Serum. A browser UI shows the whole library as a latent-space map and
lets you generate presets live and watch where they land.

> **No presets are included.** This repository contains code only. You bring
> your own preset collection, and it stays on your machine under `data/`,
> which is gitignored. See [Presets, copyright and this repo](#presets-copyright-and-this-repo).

## How it works

```
 .SerumPreset ──extract──▶ numeric params ──train──▶ per-type VAE
                                                         │
 playable .SerumPreset ◀──repack── donor preset ◀──decode── sampled latent point
```

1. **Import** (`serumgen/importer.py`). Scans a folder recursively, classifies
   each preset by folder name, filename prefix (`BA`, `LD`, `PD`, …), keyword or
   metadata tags, de-duplicates by SHA-256 and copies it into
   `data/library/<type>/`.
2. **Extract** (`serumgen/extract.py`). Unpacks each binary preset to JSON with
   [serum-preset-packager](https://github.com/KennethWussmann/serum-preset-packager)
   and flattens every numeric parameter into a feature vector keyed by module
   path, e.g. `Oscillator0/WTOsc0/kParamTablePos`.
3. **Train** (`serumgen/train.py`, `serumgen/vae.py`). Fits one PyTorch VAE per
   type: it encodes a preset into a 16-dimensional latent space and learns to
   decode any point back into a full parameter set.
4. **Generate** (`serumgen/generate.py`). Picks a point near the real presets
   (or near one preset, or between two), decodes it to parameters, and injects
   them into the library preset whose own latent position is closest. That
   "donor" supplies what the model doesn't learn (wavetables, routing, FX),
   so the result stays coherent. The donor's name, author and tags are
   stripped from the output.
5. **Explore** (`serumgen/web.py`, `serumgen/map.py`). A live browser demo, or
   a static HTML latent-space report.

## Requirements

- Python **3.11+**
- Git (the `.SerumPreset` codec is a git submodule)
- A collection of Serum 2 `.SerumPreset` files you own or are licensed to
  use. About **40 or more per type** are needed to train that type.
- Serum 2, to play the results

PyTorch runs on the CPU here; no GPU is needed.

## Install

```bash
git clone --recurse-submodules https://github.com/ssona2k/serumgen.git
cd serumgen
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you cloned without `--recurse-submodules`, fetch the codec with:

```bash
git submodule update --init
```

## Quick start

```bash
serumgen import ~/my-presets             # build data/library/ from your presets
serumgen library                          # presets per type, and which are trained
serumgen train bass                       # train the bass model (a few minutes on CPU)
serumgen generate bass -n 10              # write 10 new presets to data/generated/bass/
serumgen web                              # open the browser UI
```

Copy the generated files into your Serum presets folder, or drag them onto
Serum, and play.

## Command reference

Every command accepts `--library-dir` (default `data/library`),
`--models-dir` (default `data/models`) and `--out-dir` (default
`data/generated`).

| Command | What it does |
|---|---|
| `serumgen import <folder>` | Scan a folder (recursively) into the typed library. Duplicates are skipped and legacy `.fxp` files are counted but not imported. |
| `serumgen library` | Show preset counts per type and when each type was last trained. |
| `serumgen extract <type>` | Rebuild that type's feature caches after you change the extractor. |
| `serumgen train <type>` | Train the VAE for a type. Options: `--epochs` (500), `--latent-dim` (16), and `--force` to train with fewer than 40 presets. |
| `serumgen generate <type>` | Generate new presets. Options: `-n` (5), `--temperature` (0.8), `--like <preset>` with `--spread` (0.3), `--template <substring>`. |
| `serumgen interpolate <type> <a> <b>` | Morph between two presets in `--steps` (5) steps. |
| `serumgen map <type>` | Write a latent-space HTML report (`map_<type>.html`), with `--tsne` instead of PCA. |
| `serumgen web` | Start the browser UI. Options: `--port` (8765), `--host`, `--password`, `--no-open`. |

### Generation controls

- **`--temperature`** sets how far from real presets to wander. Around `0.2`
  you get close variations of existing sounds; around `1.8` the model explores
  beyond the clusters and gets wilder.
- **`--like path/to/preset.SerumPreset`** generates variations of one specific
  preset; `--spread` sets how much they vary.
- **`--template bright`** limits which library presets may act as the donor
  (substring match on the filename), for example to keep a family of
  wavetables.
- **`interpolate`** walks a straight line through latent space between two
  presets and writes one preset per step.

```bash
serumgen generate lead -n 20 --temperature 0.4
serumgen generate bass --like "data/library/bass/My Bass.SerumPreset" --spread 0.2
serumgen interpolate pad "data/library/pad/Soft.SerumPreset" "data/library/pad/Dark.SerumPreset" --steps 8
```

Legacy Serum 1 `.fxp` files can't be read. Open them in Serum 2 and re-save
them as `.SerumPreset` to include them.

### The web UI

`serumgen web` serves a local page at <http://127.0.0.1:8765>. It plots every
trained type's library as a density map, colored by type. Pick a type, set
the temperature, and generate: each new preset appears as a marker inside the
learned cluster and downloads as a playable `.SerumPreset`, singly or as a
zip per type.

To serve it beyond localhost (a VPS, a LAN), bind with `--host 0.0.0.0` and
always set a password with `--password` or `SERUMGEN_PASSWORD`. The
username is `serumgen`. [`deploy/README.md`](deploy/README.md) has a
step-by-step systemd setup.

### Generated preset metadata

Generated presets get a fresh name and description, and the donor's author,
vendor, URL and tags are cleared. The author and vendor fields are set by
`_AUTHOR` in [`serumgen/generate.py`](serumgen/generate.py); change it to your
own name or label.

## Where your data lives

```text
data/                    # gitignored: everything here is local to you
├── library/<type>/      # imported presets + their .json feature caches, index.json
├── models/<type>/       # trained VAE weights, scaler and meta.json
└── generated/<type>/    # presets SerumGen wrote
```

Models depend on the extractor. If you change what
`extract_numeric_features()` returns, run `serumgen extract <type>` and then
`serumgen train <type>` again.

## Development

```bash
pytest
```

Unit tests run anywhere. The integration tests run the real pipeline (import,
train a tiny model, generate, repack) on real preset files. Since none ship
with the repo, they read from `$SERUMGEN_TEST_PRESETS`, falling back to
`data/library/bass/`, and are **skipped** when fewer than 45 presets are
found:

```bash
SERUMGEN_TEST_PRESETS=~/my-presets/bass pytest
```

[`AGENTS.md`](AGENTS.md) covers the architecture and code conventions (path
injection, the feature-key format, nearest-donor generation, near-library
sampling). The design notes behind individual features are in
[`docs/superpowers/specs/`](docs/superpowers/specs/).

### Project layout

```text
serumgen/                 # the package: CLI, pipeline stages, web server
├── static/               # the web UI (one HTML file) and brand SVGs
serum-preset-packager/    # git submodule: .SerumPreset unpack/pack
tests/                    # pytest suite (tests/presets.py finds real presets)
deploy/                   # systemd unit + VPS guide
scripts/                  # maintenance scripts (regenerating the logo SVGs)
docs/superpowers/specs/   # design notes
```

## Presets, copyright and this repo

Presets made by others, whether commercial packs, free downloads or Serum's
factory content, are their authors' work. SerumGen is built so that none of
it has to leave your machine:

- `data/`, `*.SerumPreset` and `*.fxp` are gitignored. Please keep it that
  way in forks and pull requests.
- Tests, docs and examples use made-up names only.
- Generated presets are new parameter sets, but they inherit wavetables and
  other non-modelled data from a donor preset in your library. Check the
  licence of the presets you trained on before you distribute or sell
  anything you generate.

Only train on presets you're entitled to use.

## Credits

SerumGen relies on other people's work, in particular for reading and
writing the `.SerumPreset` format:

- **[serum-preset-packager](https://github.com/KennethWussmann/serum-preset-packager)**
  by [Kenneth Wussmann](https://github.com/KennethWussmann) unpacks
  `.SerumPreset` files into JSON and packs them back. It's included as a git
  submodule and is not redistributed here. Every read and write SerumGen
  makes goes through it ([`serumgen/packager.py`](serumgen/packager.py) is a
  thin wrapper). The upstream repository does not declare a licence, so its
  terms are its author's.
- **[@0xdevalias](https://github.com/0xdevalias)** published the
  [reverse-engineering notes](https://gist.github.com/0xdevalias/5a06349b376d01b2a76ad27a86b08c1b)
  on the Serum 2 preset format that the packager builds on.
- The SerumGen wordmark is outlined from
  **[Funnel Display](https://github.com/google/fonts/tree/main/ofl/funneldisplay)**
  (SIL Open Font License 1.1) by `scripts/outline_logo.py`; the font itself
  isn't shipped.
- Built with [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/),
  [NumPy](https://numpy.org/), [cbor2](https://github.com/agronholm/cbor2),
  [python-zstandard](https://github.com/indygreg/python-zstandard) and
  [msgpack](https://msgpack.org/).

Serum is a trademark of Xfer Records. SerumGen is an independent project,
not affiliated with or endorsed by Xfer Records.

## License

[MIT](LICENSE) © 2026 ssona2k. The MIT licence covers SerumGen's own code.
The submodule and any presets you use keep their own terms.
