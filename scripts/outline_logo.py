"""Regenerate the SerumGen brand SVGs in serumgen/static/brand/.

Locked spec: Funnel Display (OFL), "SerumGen", weight 500, plain, letter
spacing -0.025em, no mark. Favicon: "SG" at weight 620 on a rounded tile.

Letters are converted to outlines, so the web app never loads the font.
Not a runtime dependency — needs `pip install fonttools` and the variable
font file FunnelDisplay[wght].ttf from github.com/google/fonts (ofl/funneldisplay).

usage: python scripts/outline_logo.py path/to/FunnelDisplay[wght].ttf serumgen/static/brand
"""
import sys
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

FONT, OUT = sys.argv[1], Path(sys.argv[2])


def instance(wght):
    return instantiateVariableFont(TTFont(FONT), {"wght": wght}, inplace=False)


def kern_table(font):
    """Pair-adjust x-advances from GPOS PairPos lookups (formats 1 and 2)."""
    pairs = {}
    if "GPOS" not in font:
        return pairs
    gpos = font["GPOS"].table
    for lookup in gpos.LookupList.Lookup:
        subs = lookup.SubTable
        if lookup.LookupType == 9:
            subs = [s.ExtSubTable for s in subs]
        elif lookup.LookupType != 2:
            continue
        for st in subs:
            cov = st.Coverage.glyphs
            if st.Format == 1:
                for i, first in enumerate(cov):
                    for pvr in st.PairSet[i].PairValueRecord:
                        v = getattr(pvr.Value1, "XAdvance", 0) or 0
                        pairs.setdefault((first, pvr.SecondGlyph), v)
            elif st.Format == 2:
                c1 = st.ClassDef1.classDefs
                c2 = st.ClassDef2.classDefs
                for first in cov:
                    k1 = c1.get(first, 0)
                    rec = st.Class1Record[k1]
                    for second, k2 in list(c2.items()):
                        v = getattr(rec.Class2Record[k2].Value1, "XAdvance", 0) or 0
                        if v:
                            pairs.setdefault((first, second), v)
    return pairs


def set_text(font, text, tracking_em):
    """Return (svg path d, bounds) for text laid out with kerning + tracking."""
    upm = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    hmtx = font["hmtx"]
    kern = kern_table(font)
    names = [cmap[ord(c)] for c in text]
    svg = SVGPathPen(gs, ntos=fmt)
    bp = BoundsPen(gs)
    x = 0
    for i, n in enumerate(names):
        # font units are y-up; SVG is y-down, so flip
        gs[n].draw(TransformPen(svg, (1, 0, 0, -1, x, 0)))
        gs[n].draw(TransformPen(bp, (1, 0, 0, 1, x, 0)))
        x += hmtx[n][0]
        if i < len(names) - 1:
            x += kern.get((n, names[i + 1]), 0) + tracking_em * upm
    xmin, ymin, xmax, ymax = bp.bounds
    return svg.getCommands(), (xmin, -ymax, xmax - xmin, ymax - ymin)


def fmt(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


# ---- wordmark: Funnel Display 500, "SerumGen", plain, -0.025em ----
font = instance(500)
d, (x, y, w, h) = set_text(font, "SerumGen", -0.025)
pad = 0
(OUT / "serumgen-wordmark.svg").write_text(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{fmt(x - pad)} {fmt(y - pad)} '
    f'{fmt(w + 2 * pad)} {fmt(h + 2 * pad)}" role="img" aria-label="SerumGen">'
    f'<title>SerumGen</title><path fill="currentColor" d="{d}"/></svg>\n')
print("wordmark", w, h)

# ---- favicon: "SG" on a rounded tile, heavier so it holds at 16 px ----
font = instance(620)
d, (x, y, w, h) = set_text(font, "SG", -0.04)
S = 64
scale = (S * 0.56) / w if w / h > 1 else (S * 0.5) / h
tx = (S - w * scale) / 2 - x * scale
ty = (S - h * scale) / 2 - y * scale
(OUT / "favicon.svg").write_text(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}">'
    '<style>.t{fill:#f5f5f7}.g{fill:#1d1d1f}'
    '@media (prefers-color-scheme:dark){.t{fill:#1d1d1f}.g{fill:#f5f5f7}}</style>'
    f'<rect class="t" width="{S}" height="{S}" rx="14"/>'
    f'<path class="g" transform="translate({fmt(tx)} {fmt(ty)}) scale({scale:.5f})" d="{d}"/></svg>\n')
print("favicon ok")
