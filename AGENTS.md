# AGENTS.md

Guidance for contributors — human or AI agent — picking up this repo. For
setup and day-to-day CLI usage, see [README.md](README.md). For the reasoning
behind each feature, see the dated design docs in
[docs/superpowers/specs/](docs/superpowers/specs/).

## Architecture in one paragraph

serumgen is a four-stage pipeline: **extract** (`serumgen/extract.py`) turns a
binary `.SerumPreset` into a flat dict of numeric parameters, **train**
(`serumgen/train.py` + `serumgen/vae.py`) fits a per-type Variational
Autoencoder on those parameters, **generate** (`serumgen/generate.py`) samples
the trained latent space and repacks a new playable preset, and **map/web**
(`serumgen/map.py`, `serumgen/web.py`) visualize the result — a static HTML
report or a live browser demo. `serumgen/importer.py` and
`serumgen/packager.py` are shared infrastructure: classifying and
deduplicating incoming presets into `data/library/`, and wrapping the
`serum-preset-packager` submodule that does the actual binary unpack/pack.
`serumgen/cli.py` is the only place that wires all of this into subcommands.

## Conventions worth knowing before you touch code

**Path injection, not path constants.** Every function that touches the
filesystem — `train_type()`, `generate()`, `build_map()`, `import_folder()`,
`available_types()`, and so on — takes `library_dir`/`models_dir`/`out_dir` as
an explicit parameter. Nothing below `serumgen/cli.py` imports a hardcoded
path. This is why moving the default data location from `library/` to
`data/library/` touched only 3 constants in `cli.py` (plus matching defaults
in `web.py`) and zero other files. Keep this pattern: a new function that
needs a data directory takes it as a parameter, never reaches for a module-
level constant. It's also what makes every module testable in isolation with
pytest's `tmp_path`/`tmp_path_factory` fixtures instead of mocking the
filesystem.

**The feature-key format.** Extracted parameters are keyed by their full
module path, `/`-joined, e.g. `Oscillator0/WTOsc0/kParamTablePos`.
`extract_numeric_features()` recurses into **dict** sub-modules (an
oscillator's wavetable/granular/sample engine nests one level deeper than the
top-level module) but deliberately does **not** recurse into **lists** — FX
racks store their effects as a positional list, and including them blows up
the feature-vector dimension by ~3x with mostly-zero, low-signal data (this
was measured, not assumed — see
`docs/superpowers/specs/2026-07-14-richer-features-design.md`).
`inject_features()` in `serumgen/generate.py` walks the same `/`-split path
forward through the nested dicts and writes the leaf value into that module's
`plainParams` — only where the full path already exists in the target (a
silent no-op on any path the template preset doesn't have, which is
deliberate).

**Retrain after any extractor change.** `data/models/` is derived, gitignored,
and never hand-edited. If you change what `extract_numeric_features()`
returns (a new param, a different recursion rule), the per-type `.json`
feature caches in `data/library/<type>/` are now stale relative to the code.
Rebuild them and retrain:

    serumgen extract <type>   # rebuild the .json caches from the .SerumPreset files
    serumgen train <type>     # retrain on the rebuilt caches

Do this for every type your change affects before trusting `serumgen
generate`/`map`/`web` output.

**Nearest-donor generation.** A generated preset's numeric parameters come
from the VAE decoder, but its wavetable data, routing, and anything else the
model doesn't model comes from a real library preset used as a "template" —
injected via `inject_features()`. `generate()`/`interpolate()` pick the
template whose *own* latent position is nearest the generated point
(`_nearest_template()` in `serumgen/generate.py`), not a random one, so the
inherited wavetable is sonically congruent with what the model actually
generated. Preserve this when touching generation — reverting to random
template selection measurably degrades output coherence.

**Near-library sampling, not the prior.** The trained VAEs show heavy
posterior collapse: most latent dims barely vary across the library (std
~0.03) while 1-3 dims carry nearly all the signal. Sampling `randn * T` from
the N(0, I) prior therefore lands far off the data even at low temperature
(~80x farther from any real preset than presets sit from each other).
`sample_near_library()` in `serumgen/generate.py` instead picks a random
library preset as an anchor and adds per-dim noise scaled by the library's
own spread times temperature. Both `serumgen generate` and `serumgen web`
use it; `--like` keeps its own centre + `--spread`. `--template` narrows only
the donors — anchors still come from the whole library. Don't go back to
prior sampling unless the model is retrained without the collapse.

**No presets, and no traces of them, in git.** Third-party presets are the
copyrighted work of their authors, and models and generated presets are
derived from them, so everything under `data/` is gitignored (and so is
`*.SerumPreset`/`*.fxp` anywhere). The same goes for text: test cases, docs
and examples use made-up names (`BA Deep [XY]`, `Jane Doe`), never a real
preset, pack, vendor or sound-designer name, and never a local path to
someone's collection.

## Testing

    pytest                    # scoped to tests/ via pyproject.toml's testpaths

Several fixtures (`real_library` in `test_generate.py`, `trained` in
`test_map.py`/`test_web.py`) copy real presets into a temp dir and train a
tiny real model (5 epochs, latent-dim 4) — these take 5-15 seconds each,
which is expected, not a hang. No presets ship with the repo:
`tests/presets.py` finds them in `$SERUMGEN_TEST_PRESETS` (any folder of
`.SerumPreset` files) or your local `data/library/bass/`, and skips those
tests when fewer than 45 are available. They exist to test the full
pipeline against real binary data rather than mocks; don't replace them with
synthetic fixtures without a strong reason.

## Things that are deliberately not built yet

These came up during design discussions and were parked, not forgotten —
don't start one without a fresh brainstorming pass, since each has open
questions that depend on decisions not yet made:

- **A preset-upload feature.** Should be a feature inside `serumgen web`, not
  a separate app. Basic Auth + `--host` (see `deploy/README.md`) cover the
  transport-level access control; still needs its own design for what
  happens to an uploaded file (public upload endpoint vs. invite-only,
  auto-train vs. review queue).
- **Morph/interpolate view in the web UI.** The `interpolate()` backend
  already exists (`serumgen/generate.py`); only the browser-side visualization
  is missing.
- **Conditional VAE for style-targeted generation** (e.g. "generate a bass in
  the style of a particular sound designer"). Would need a labeled-style training signal the
  current per-type-only models don't have.
- **Smarter, non-positional FX-parameter inclusion.** FX racks are currently
  skipped entirely in extraction (see the feature-key format note above) —
  including them without the dimensionality blowup needs a different
  encoding than the plain path-join used for oscillator params.
- **Audio audition in the web UI.** Blocked on headless Serum rendering —
  no way to render audio from a `.SerumPreset` without the actual plugin.
