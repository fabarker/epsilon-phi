# Portfolio Analytics UI — working prototype

One page, both steps, laid out to the host dashboard's page-folder convention.

Open **`proposalTool/proposalTool.html`** directly — no server, and no network.

See **`PORTING.md`** for dropping it into `cyrus_pmg.dashboard`: copy the folder, add one Flask
route, add one nav link.

## The flow

**Landing → mandate dialog → workspace.** The page opens on a dark navy front door carrying the
title *Portfolio Management Group Implementation Tool*, a **Start Here** button, and a hero that is
**a real output of the tool** — three allocations drawn as composition bands from the same model and
the same category colours the allocation table uses, on a white sheet floating over the navy. The
bands grow on entry, suppressed under `prefers-reduced-motion`.

The rail is hidden here, because every control in it depends on a mandate existing.

**Start Here** opens the mandate card &mdash; a modal with the three mandate fields, live
validation, and a proper combobox for Primary PWA (arrow keys, `aria-activedescendant`, a
two-character threshold). The same dialog is reused for **Edit** from the rail later.

On Continue the workspace appears with the rail in place and focus on the Allocation control.

## The two steps

Switch between them with the tabs at the top of the document.

### 1 — Asset allocation

- **Mandate** with live validation: mandate size **≥ $5,000,000** is a hard block, and it cannot
  exceed the top account size
- **Scenario basis** (currency, hedging) — fixed across every column; changing either rebuilds all
- **Base portfolio**: allocation select plus **Exclude Real Estate** (enabled only for Full and
  Ex HFs) and **Exclude Tactical Asset Allocation** tick boxes
- Add up to three comparisons from the **`+` column** at the right of the allocation table. Adding
  happens in the popover — the column lands immediately and the popover stays open
- Allocation table, two summary charts, and the three-section risk dashboard with its mixed
  column span

### 2 — Implementation

Only the **base portfolio** is implemented. Attach one PMG-authored sleeve per category; the
category's weight is distributed proportionally across the sleeve's products.

Thirteen columns: Categories & Asset Classes · Products · Allocation (%) · Ticker · Style ·
Vehicle · Source · Liquidity · Exposure ccy · Cost · Mgmt fee · Wtd fee · Notional. Both identity
columns are pinned, so category, asset class and product name stay visible while the rest scrolls.

The download unlocks only when every category carries a sleeve.

## The rounding rule

Every printed figure derives from the **printed** weight, so the document hand-reconciles:

1. Weights round to 2dp by largest remainder — the column sums to exactly `100.00%`
2. Notional derives from that rounded weight, then rounds to the nearest $100
3. The weighted fee derives from it too

All of these hold at once: every line is a round hundred, lines sum to the mandate exactly,
category subtotals reconcile, weights total `100.00%`, and **weight × mandate equals the notional
shown** — zero drift.

## Typography

House faces, local, no external request.

| Role | Face | Notes |
|---|---|---|
| Wordmark | **Goldman Sans** | Regular only — used at 400 and never asked for bold |
| Headings & body | **GS Sans** | Variable, `wght 250–700` |
| Tables & data | **GS Sans Condensed** | Variable, `wght 300–900`. Narrow enough for the thirteen-column table, and it echoes Aptos Narrow in the Excel risk sheet |
| Fallback | **Roboto** | 1250 glyphs against GS Sans's 500 |

All five carry `tnum`, so `font-variant-numeric: tabular-nums` works throughout.

## The design

Porcelain restraint on the **mpo-ui `:root` palette**, with a navy `#shellnav` rail. All eighteen
house tokens are in use. Two are adjusted for contrast and documented in the specification:

| | House value | Used | Why |
|---|---|---|---|
| Rail input border | `#2e4468` — 1.59:1 | `#52739c` — 3.18:1 | Below 3:1 the field boundary is invisible |
| Rail primary button | `#1f5fbf` — 2.56:1 vs rail | `#2a6ad0` — 3.02:1 | Only candidate clearing text 4.5 *and* non-text 3.0 |

`#1f5fbf` remains the accent throughout the document body.

## Editing

```bash
python3 build_styles.py     # rewrites proposalTool/
```

Output is `proposalTool/` — html, css, js and fonts as separate files, the way the host
serves them.

| Source | Contains |
|---|---|
| `build_styles.py` | Base stylesheet, `@font-face`, page template, assembly |
| `themes.py` | The token set |
| `pickers.py` | Data model, state, table and chart renderers, the `+` column picker |
| `implementation.py` | Sleeve library, implementation table, fee and rounding logic |

Do not hand-edit anything under `proposalTool/` — it is overwritten on every run.

## Reference

- `../ui-spec/spec.html` — the production build specification (asset allocation step)
- `../ui-design-qa.md` — the full decision log, Q1 to Q33
