# Richer features + smart donor selection

**Date:** 2026-07-14
**Status:** Approved, building on branch `richer-features`

## Problem

Two quality limits in the generator, found by inspecting a real preset's cache:

1. **Shallow extraction.** `extract.extract_numeric_features` only reads
   `data[module].plainParams` one level deep. Serum nests the sonically decisive
   oscillator controls one level deeper — e.g.
   `Oscillator0.WTOsc0.plainParams.kParamTablePos` (wavetable position) and
   `kParamWarp`/`kParamWarp2`. The model never sees them. Measured on a sample
   preset: 94 params captured, wavetable position and warp all missing.
2. **Random donor.** Generation injects AI parameters into a **random** library
   preset (`random.choice`) to inherit its wavetable/routing. A bass built on a
   random donor can inherit an incongruous wavetable, so the output is
   incoherent for its type.

## Decision — measured

Three extraction depths were measured on the real library:

| Approach | bass per-preset | bass union dim | captures osc internals |
|---|---|---|---|
| shallow (current) | 94 | ~155 | no |
| **dict-recursive (chosen)** | ~103 | ~669 | **yes** |
| full incl. FX lists | ~103 | 2298 (95% zeros) | yes, but explodes |

Dict-recursive is the sweet spot: it captures the oscillator sub-modules
(wavetable position, warp, sub-osc) without the positional FX-slot list
explosion that makes the full version too sparse to train well.

## Change 1 — recursive dict extraction

`extract.extract_numeric_features` walks `data` recursively through **dict**
sub-modules (lists skipped), and whenever it finds a `plainParams` dict emits
each numeric param keyed by its full path, joined with `/`:

```
Env0/kParamAttack                       (was Env0_kParamAttack)
Oscillator0/WTOsc0/kParamTablePos       (newly captured)
Oscillator0/WTOsc0/kParamWarp           (newly captured)
```

The `plainParams` literal is omitted from the key; the path is the module chain,
and the leaf always lives in that node's `plainParams`. Separator changed from
`_` to `/` because nested paths need an unambiguous separator — verified no
module or param name contains `/`. This changes the key format, so **all models
must be retrained** (done in this branch).

`generate.inject_features` is rewritten to match: split the key on `/`, walk the
module chain through nested dicts, and set the leaf inside that node's
`plainParams` — only when the full path and leaf already exist in the template
(unknown paths are skipped, as before).

Lists are intentionally not traversed: FX-rack params are positional and type-
keyed, which balloons dimensionality without proportional benefit. Documented as
a deliberate scope line.

## Change 2 — nearest-donor selection

Instead of `random.choice(templates)`, generation picks the donor whose latent
position is nearest to the generated point, so the inherited wavetable comes
from a preset the model considers sonically similar.

- Encode every template of the type to its latent mean once (from the cached
  `.json` features, falling back to unpacking if a cache is missing).
- For each generated `z`, choose the template minimizing `‖mu_template − z‖²`.
- Applies to `generate` (temperature and `--like`) and `interpolate`.

## Files

- Modify `serumgen/extract.py` — recursive `extract_numeric_features`.
- Modify `serumgen/generate.py` — nested `inject_features`; `_template_latents`
  + `_nearest_template` helpers; use them in `generate`/`interpolate`.
- Modify `tests/test_extract.py`, `tests/test_generate.py` — new key format,
  nested capture, nested injection, donor-selection behavior.
- Rebuild library `.json` caches (`serumgen extract <type>`) and retrain all 10
  models on the richer features.
- No change to `train.py`, `map.py`, `web.py`, `cli.py` — they consume feature
  keys generically.

## Verification

- Unit: nested param round-trips (extract → inject writes it back); donor
  selection picks the nearest, not random.
- Live: retrain bass, generate via `serumgen web`, confirm generated presets
  still load (packager round-trip) and that a generated preset's wavetable-
  position param now varies from the donor's.

## Out of scope

FX-list parameters, latent-dim tuning (stays 16), per-type feature selection.
