# serumgen web — live demo UI

**Date:** 2026-07-14
**Status:** Approved design, building on branch `serumgen-web-ui`

## Goal

A local web UI that makes serumgen easy to understand and impressive to demo to
other audio professionals. Core loop: show the whole preset library as a
latent-space "galaxy" colored by type, then generate a new preset live and watch
it land inside the learned cluster — provably part of the distribution — and
download it for Serum.

## Demo narrative (what the UI must deliver)

1. The galaxy — the library as a field of dots, colored by sound type.
2. Spotlight — pick a type; its cluster is the active view.
3. Generate — one click drops a new preset as a bright marker inside the cluster.
   The temperature slider visibly pushes generated points outward as it rises.
4. Download — the generated preset's name + a download button → load in Serum.

Morph/interpolate between two presets is **phase 2**, not v1.

## Architecture

- **`serumgen web` subcommand** starts a local server (default port 8765) and
  opens the browser.
- **Backend: Python stdlib `http.server` (ThreadingHTTPServer)** — zero new
  dependencies, consistent with the codebase's minimalist choices (argparse over
  click). ~5 routes in `serumgen/web.py`. Reuses existing functions:
  `generate.load_model`, `train.load_features`, `map._encode_matrix`,
  `generate.decode_to_features`, `generate._write_preset`, `generate._templates`.
- **Projection cache (the one real design decision):** on first request for a
  type, encode all library presets of that type to their latent means, **fit a
  PCA(2) once**, and cache the fitted projector + the library's 2-D coordinates +
  model artifacts + templates in a module-level dict. A generated preset's
  sampled latent `z` is projected through that **same** cached PCA, so its dot is
  positioned in a frame directly comparable to the library. (The existing
  `map.py` refits PCA per call and cannot place a live point consistently — the
  server caches instead.)
- **Frontend: one file `serumgen/static/index.html`, vanilla JS canvas** — no
  chart library, so the "dot pulses in" animation and hover tooltips are hand-
  drawn. Dark galaxy aesthetic. Works offline; served by the backend.

## Endpoints

- `GET /` → `index.html`
- `GET /api/types` → `[{type, count, color, trained: bool}]` for types that have
  a trained model; drives the type chips.
- `GET /api/library?type=bass` → `{type, color, points: [{name, x, y}]}` where
  x/y are raw PCA coordinates. Cached after first compute.
- `POST /api/generate` `{type, temperature}` → `{name, x, y, download_id}`.
  Samples `z ~ N(0, temperature²)`, decodes to features, writes a real
  `.SerumPreset` into `generated/<type>/`, projects `z` through the cached PCA.
- `GET /api/download/<id>` → serves the generated `.SerumPreset` as an
  attachment. `download_id → path` kept in a module dict.

The frontend maps raw PCA coordinates to canvas pixels itself (fit library
bounds + padding), so generated points outside the library's bounds render at
the edge — which is the visible meaning of high temperature.

## Files

- Create `serumgen/web.py` — projection cache, generation helper, request handler, `serve()`.
- Create `serumgen/static/index.html` — the page.
- Modify `serumgen/cli.py` — add `web` subcommand (`--port`, `--no-open`).
- Modify `pyproject.toml` — include `serumgen/static/*` as package data.
- Create `tests/test_web.py` — smoke tests on the core helpers against a tiny
  trained model (reusing the `real_library` fixture pattern from
  `test_generate.py`): projection returns one point per library preset with
  finite coords; a live generate produces a parseable preset and finite
  projected coords; a generated download path resolves.
- Modify `README.md` — document `serumgen web`.

## Non-goals (v1)

Morph/interpolate view, auth, remote hosting, audio preview (blocked on headless
Serum rendering), persistence of generated dots across sessions.
