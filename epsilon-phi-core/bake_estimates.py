"""Write the Proposal Tool's ``assetEstimates.json``.

Loops every currency against every hedging assumption, builds one SAAPortfolio
per pair holding all the assets, reads each asset's long-term estimates off it,
and writes the one file the workbook's assumptions sheet is made of: each
asset's risk premium with its estimated range, its volatility, Sharpe ratio,
total return, the hedging ratio in force, and the window it was estimated over.

TO RUN IT IN PYCHARM: open this file, set the four constants under CONFIGURE
ME, make BUILD ONE PORTFOLIO below match how you build an SAAPortfolio, and
press Run. Nothing is imported from the Proposal Tool, so it can live in the
Cyrus repo; the only import that matters is your own analytics library, and it
happens inside buildPortfolio.

Leave DRY_RUN = True for the first run and it uses a stand-in portfolio instead
of your library, so you can see the file it writes before wiring anything up.

WHY ONE PORTFOLIO PER PAIR IS ENOUGH. These are properties of an ASSET under a
(currency, hedging) basis, not of a portfolio: the weights do not enter any of
them. So the portfolio here exists only to carry the assets and the basis, and
it is built equally weighted.

UNITS ARE FRACTIONS, not percent: 0.0111 means 1.11%. The bake's slice files
are the other way round. That is the tool's own split, and the workbook formats
each accordingly.
"""

from __future__ import annotations

import datetime
import json
import math
import os

# ===========================================================================
#  CONFIGURE ME
# ===========================================================================

#: Where assetEstimates.json goes. The service looks for it beside the bake's
#: slice files, or wherever SCENARIO_ASSET_ESTIMATES points.
OUT_DIR = './baked'

#: Run without your analytics library, against a stand-in, to see the shape.
DRY_RUN = True

CURRENCIES = ['USD', 'GBP', 'CHF', 'EUR']

#: The four the tool offers. A block is keyed "USD|ISG Hedged" - with the
#: space, unlike the slice filenames, which drop it.
HEDGING = ['Hedged', 'ISG Hedged', 'Unhedged', 'Equity Not Hedged']

#: Every asset, as (ticker, reporting name, category). This is the Proposal
#: Tool's own ASSET_METADATA, from pmgService/scenario/portfolio_weights.py,
#: and it must stay in step with it: the assumptions sheet matches its rows to
#: the strategic sheets BY REPORTING NAME, and the categories drive the hedge
#: ratios below.
ASSETS = [
    ('LHTRYIN', 'US Dollar Debt', 'Investment Grade Fixed Income'),
    ('LHYIELD', 'US High Yield', 'Other Fixed Income'),
    ('FRUS1GR', 'US Large Cap Growth Equity', 'Public Equity'),
    ('FRUS1VA', 'US Large Cap Value Equity', 'Public Equity'),
    ('FRUSS2L', 'US Small Cap Equity', 'Public Equity'),
    ('MSEXUKL', 'Europe ex-UK Equity', 'Public Equity'),
    ('MSUTDKL', 'UK Equity', 'Public Equity'),
    ('MSJPANL', 'Japanese Equity', 'Public Equity'),
    ('MSPXJPL', 'Asia-Pacific Equity', 'Public Equity'),
    ('MSEMKF$', 'Emerging Market Equity', 'Public Equity'),
    ('CSTEVDH', 'Event Driven', 'Hedge Funds'),
    ('CSTLNSH', 'Equity Long/Short', 'Hedge Funds'),
    ('CSFBMTT', 'Tactical Trading', 'Hedge Funds'),
    ('PE_BUYOUT', 'Buyout', 'Private Equity'),
    ('PE_GROWTH', 'Growth', 'Private Equity'),
    ('PE_VENTURE', 'Venture', 'Private Equity'),
    ('PRIVATE_CREDIT', 'Private Credit', 'Other Private Assets'),
    ('PA_REAL_ESTATE', 'Core Real Estate', 'Other Private Assets'),
    ('LHUT1T3', 'Tactical Tilt Fund', 'Asset Allocation Strategies'),
]

#: A currency whose analytics must run in ANOTHER currency, because the
#: database has no config for its own. The block records the substitution and
#: the workbook then says so. Empty when every currency is native.
ANALYTICS_CURRENCY = {}          # e.g. {'GBP': 'USD', 'CHF': 'USD', 'EUR': 'USD'}

#: The estimation window the context is built over.
CONTEXT_START_DATE = '30-Nov-1983'
CONTEXT_END_DATE = '31-Dec-2022'

#: Hedging policy -> the FX hedge ratio for an asset, by its category. This is
#: the tool's own table; the ISG Hedged row is a stand-in awaiting PMG's house
#: numbers. The ratio is READ BACK onto every row, so a change here changes the
#: file.
HEDGE_RATIOS = {
    'Hedged': lambda category: 1.0,
    'Unhedged': lambda category: 0.0,
    'Equity Not Hedged': lambda category: 0.0 if category == 'Public Equity' else 1.0,
    'ISG Hedged': lambda category: 0.5 if category == 'Public Equity' else 1.0,
}


# ===========================================================================
#  BUILD ONE PORTFOLIO  -  the one function to make yours
# ===========================================================================

def buildPortfolio(currency, hedging, assets):
    """Return an SAAPortfolio in *currency* on the *hedging* basis, holding
    every asset in *assets* - a list of (ticker, reportingName, category).

    What follows is how the Proposal Tool builds one, so it is a working
    starting point rather than a sketch. Replace the body with your own if you
    build them differently; everything after this function only reads.

    Three things the rest of the script depends on:

      * every asset carries its REPORTING NAME and CATEGORY, which is what
        `set_reporting_info` is for. Without them the estimates cannot be
        matched to the sheet's rows;
      * the hedge ratios are applied, because `hedging_ratio` is read back off
        each asset and printed;
      * the portfolio is not shared. SAAPortfolio is mutable and memoises its
        own analytics, so a cached instance handed out twice with different
        hedging would answer with one basis while claiming the other. Build a
        fresh one per pair, which is what the loop below does.
    """
    if DRY_RUN:
        return _StubPortfolio(assets, hedging)

    # ---- your analytics library ------------------------------------------
    from epsilonPhi.core.portfolio.SAAPortfolio import SAAPortfolio
    from epsilonPhi.core.schema.Schema import ContextCreator

    context = ContextCreator(currency=currency,
                             start_date=CONTEXT_START_DATE,
                             end_date=CONTEXT_END_DATE).create_context()

    # Equal weights: the estimates are per asset and no figure read below
    # depends on them, so the portfolio is here to carry the assets and the
    # basis, nothing more.
    weight = 1.0 / len(assets)
    weights = {ticker: weight for ticker, _, _ in assets}
    portfolio = SAAPortfolio.from_dict('Estimates', weights, context)

    lookup = {ticker: (name, category) for ticker, name, category in assets}
    for assetName in portfolio.get_asset_names():
        found = lookup.get(assetName)
        if found:
            portfolio.get_asset(assetName).set_reporting_info(
                reporting_name=found[0], category=found[1])

    ratioFor = HEDGE_RATIOS.get(hedging)
    if ratioFor is None:
        raise ValueError('Unknown hedging assumption {!r}; expected one of: {}'.format(
            hedging, ', '.join(HEDGE_RATIOS)))
    portfolio.set_hedging_ratios([
        ratioFor(lookup.get(name, ('', ''))[1]) for name in portfolio.get_asset_names()])
    return portfolio


# ===========================================================================
#  Reading the estimates off a portfolio
# ===========================================================================

def assetRows(portfolio):
    """One row per asset, every figure read off the asset object.

    The risk premium arrives with an uncertainty either side of it, which the
    sheet prints as an estimated range, so it is stored as three numbers.
    """
    known = {name: category for _, name, category in ASSETS}
    rows = []
    for asset in portfolio.get_assets():
        name = str(asset.reporting_name)
        category = str(asset.category)
        if name not in known:
            raise ValueError(
                '{!r} is not an asset the Proposal Tool reports on. Add it to ASSETS above '
                'AND to ASSET_METADATA in the tool, or the assumptions sheet will drop the '
                'row without saying so.'.format(name))
        if category != known[name]:
            raise ValueError(
                '{!r} is in {!r} here but {!r} in the Proposal Tool; the assumptions sheet '
                'groups by category and the two must agree.'.format(name, category, known[name]))
        premia = float(asset.get_risk_premia())
        uncertainty = float(asset.get_uncertainty())
        rows.append({
            'reportingName': name,
            'category': category,
            'lower': premia - uncertainty,
            'mean': premia,
            'upper': premia + uncertainty,
            'volatility': float(asset.get_volatility()),
            'sharpe': float(asset.get_sharpe_ratio()),
            'totalReturn': float(asset.get_total_return()),
            'hedgingRatio': float(asset.hedging_ratio),
            'from': str(asset.index.min())[:10],
            'to': str(asset.index.max())[:10],
        })
    if not rows:
        raise ValueError('the portfolio reported on no assets')
    missing = sorted(known) if not rows else sorted(set(known) - {r['reportingName'] for r in rows})
    if missing:
        raise ValueError('the portfolio is missing {}: every asset must be in every '
                         'block'.format(', '.join(missing)))
    for row in rows:                       # no NaN may reach the sheet
        for field, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('{}: {} is {!r}'.format(row['reportingName'], field, value))
    return rows


# ===========================================================================
#  The loop, and the file
# ===========================================================================

def main():
    blocks, failures = {}, []
    for currency in CURRENCIES:
        for hedging in HEDGING:
            key = '{}|{}'.format(currency, hedging)
            analytics = ANALYTICS_CURRENCY.get(currency, currency)
            try:
                portfolio = buildPortfolio(analytics, hedging, ASSETS)
                rows = assetRows(portfolio)
            except Exception as exc:       # noqa: BLE001 - reported, and the rest still run
                failures.append((key, '{}: {}'.format(type(exc).__name__, exc)))
                print('  {:<24} FAILED  {}: {}'.format(key, type(exc).__name__, exc))
                continue
            blocks[key] = {'analyticsCurrency': analytics, 'assets': rows}
            print('  {:<24} {:>2} assets, computed in {}{}'.format(
                key, len(rows), analytics,
                '  (substituted)' if analytics != currency else ''))

    if not blocks:
        raise SystemExit('\nNothing was baked. Nothing written.')

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    path = os.path.join(OUT_DIR, 'assetEstimates.json')

    # Merged, not replaced: the file holds every slice, so a partial run adds
    # to what is already there rather than throwing it away.
    out = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            out = json.load(handle)
    out.setdefault('_note', "Per-asset long-term estimates behind the export's assumptions "
                            'sheet, recorded once per (currency, hedging) slice.')
    out['updatedAt'] = datetime.datetime.now().isoformat(timespec='seconds')
    out.setdefault('slices', {}).update(blocks)

    # whole file, temp name, rename - an interrupted run leaves nothing half
    # written for the service to read
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=1, sort_keys=False)
    os.replace(temporary, path)

    print('\n{}: {} slice(s) in the file'.format(path, len(out['slices'])))
    expected = ['{}|{}'.format(c, h) for c in CURRENCIES for h in HEDGING]
    absent = [k for k in expected if k not in out['slices']]
    print('   ' + ('every slice is covered' if not absent
                   else 'still missing: ' + ', '.join(absent)))
    if failures:
        print('\n{} slice(s) failed:'.format(len(failures)))
        for key, why in failures:
            print('   {:<24} {}'.format(key, why))
    if DRY_RUN:
        print('\nDRY_RUN is on: these figures came from a stand-in, not your library.')


# ===========================================================================
#  The stand-in, for DRY_RUN
# ===========================================================================

class _StubAsset:
    def __init__(self, name, category, ratio, seed):
        self.reporting_name, self.category = name, category
        self.hedging_ratio, self._seed = ratio, seed
        self.index = _StubIndex()

    def get_risk_premia(self):
        return 0.005 + self._seed * 0.002

    def get_uncertainty(self):
        return 0.004 + self._seed * 0.0005

    def get_volatility(self):
        return 0.03 + self._seed * 0.01

    def get_sharpe_ratio(self):
        return 0.30 + self._seed * 0.02

    def get_total_return(self):
        return 0.03 + self._seed * 0.002


class _StubIndex:
    def min(self):
        return '1983-11-30 00:00:00'

    def max(self):
        return '2022-12-31 00:00:00'


class _StubPortfolio:
    def __init__(self, assets, hedging):
        ratioFor = HEDGE_RATIOS[hedging]
        self._assets = [_StubAsset(name, category, ratioFor(category), i)
                        for i, (_, name, category) in enumerate(assets)]

    def get_assets(self):
        return self._assets


if __name__ == '__main__':
    main()
