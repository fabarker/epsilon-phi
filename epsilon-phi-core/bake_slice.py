"""Write the Proposal Tool's baked JSON for ONE (currency, hedging) SLICE.

A slice is one file - `USD_Hedged.json` - holding EVERY portfolio for that
currency and hedging basis: all eight risk levels crossed with the allocation
types the universe offers, plus the ex-RAs variants. Forty-three of them in the
packaged universe. Sixteen such files (4 currencies x 4 hedging bases) are a
complete bake.

Standalone on purpose: it imports nothing from the Proposal Tool, so it can sit
in the Cyrus repo and run wherever your analytics library is importable.

    from bake_slice import Portfolio, bakeSlice

    portfolios = [
        Portfolio(riskLevel='LowVol', allocationType='Full', excludeRealAssets=False,
                  weights={'LHTRYIN': 0.62, ...},
                  # an SAAPortfolio, or a callable returning one. A CALLABLE is
                  # better: it is invoked inside the loop, so 43 portfolios are
                  # built one at a time rather than all held at once.
                  portfolio=lambda w=weights: buildSaaPortfolio('USD', w, 'Hedged')),
        ...
    ]
    bakeSlice('USD', 'Hedged', portfolios, outDir='/data/pmg/proposalTool/baked')

The whole file is written at once, replacing whatever was there, so a slice
never carries a stale key from an earlier run. A manifest entry is merged in
beside it so the page's footer can say when the bake was made.

    python bake_slice.py --demo          # a whole slice from stub portfolios

TWO TABLES MUST STAY IN STEP with the Proposal Tool, or a payload will be
served that the page cannot render:

* ASSET_METADATA is a copy of the tool's own
  (pmgService/scenario/portfolio_weights.py). Ticker, reporting name, category
  - and the order sets the order categories appear in.
* RISK_LEVEL_LABELS is a copy of the tool's (scenario/rules.py). It affects the
  display name only, never the key.

Neither has changed since the tool was built. If either does, copy it again.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os

# --------------------------------------------------------------------------
# Copies of the tool's own tables. See the note above.
# --------------------------------------------------------------------------

#: (ticker, reporting name, category), in the order categories should appear.
ASSET_METADATA = (
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
)

#: How a risk level reads in a display name. The KEY keeps the database's own
#: spelling; only the name uses these.
RISK_LEVEL_LABELS = {
    'LowVol': 'Low Vol',
    'Conservative': 'Conservative',
    'ConsMod': 'Conservative-Moderate',
    'Moderate': 'Moderate',
    'ModAgg': 'Moderate-Aggressive',
    'Agg': 'Aggressive',
    'Higher Risk': 'Higher Risk',
    'All Equity': 'All Equity',
}

#: The four hedging bases the tool offers. One slice file each, per currency.
HEDGING_BASES = ('Hedged', 'ISG Hedged', 'Unhedged', 'Equity Not Hedged')

#: The three var/pol horizons, as (index passed to the engine, printed label).
VAR_HORIZONS = ((0, 'Over 1 Month'), (11, 'Over 1 Year'), (33, 'Over 3 Years'))

#: Weights are checked against this before anything is written. Real extract
#: weights sum to 1.0 to machine epsilon, so a bigger gap is bad data - and the
#: portfolios sheet prints a hardcoded TOTAL of 100%, which would then be a lie.
WEIGHT_TOLERANCE = 1e-6

NA = 'NA'


#: One portfolio to bake. *portfolio* is an SAAPortfolio, or a callable
#: returning one - prefer the callable, so 43 of them are built one at a time.
Portfolio = collections.namedtuple(
    'Portfolio', 'riskLevel allocationType excludeRealAssets weights portfolio')


# --------------------------------------------------------------------------
# The key: four fields, one canonical string
# --------------------------------------------------------------------------

def keyString(currency, riskLevel, allocationType, excludeRealAssets) -> str:
    """`USD|Moderate|Full|0`. An all-equity book has neither of the last two,
    and both print as NA: `USD|All Equity|NA|NA`."""
    if allocationType is None or allocationType == NA:
        return '|'.join([currency, riskLevel, NA, NA])
    return '|'.join([currency, riskLevel, allocationType,
                     '1' if excludeRealAssets else '0'])


def header(riskLevel, allocationType, excludeRealAssets) -> str:
    """The column header: the name without its currency."""
    risk = RISK_LEVEL_LABELS.get(riskLevel, riskLevel)
    if allocationType is None or allocationType == NA:
        return risk
    return '{} {}{}'.format(risk, allocationType, ' ex-RAs' if excludeRealAssets else '')


def displayName(currency, riskLevel, allocationType, excludeRealAssets) -> str:
    return '{} {}'.format(currency, header(riskLevel, allocationType, excludeRealAssets))


def sliceFilename(currency, hedging) -> str:
    """`USD_ISGHedged.json` - the space is dropped, nothing else."""
    return '{}_{}.json'.format(currency, hedging.replace(' ', ''))


# --------------------------------------------------------------------------
# One payload
# --------------------------------------------------------------------------

def categoryRows(weights, tolerance=WEIGHT_TOLERANCE) -> list:
    """Categories and their assets, in ASSET_METADATA order.

    *weights* is {ticker: decimal}; the payload is in PERCENT. A zero or
    missing weight is LEFT OUT: the page reads an absent category as one the
    portfolio does not hold, where a zero row would print as a real 0.0%.
    """
    meta = {code: (name, category) for code, name, category in ASSET_METADATA}
    unknown = [t for t in weights if t not in meta]
    if unknown:
        raise ValueError(
            'not assets the Proposal Tool knows: {}. Add them to ASSET_METADATA '
            'here AND in the tool, or the payload cannot be rendered.'.format(
                ', '.join(sorted(unknown))))
    total = sum(float(w or 0.0) for w in weights.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            'weights sum to {:.10f}, not 1.0. They are decimals, not percents, '
            'and the portfolios sheet prints a TOTAL of 100% without checking.'
            .format(total))
    categories, byName = [], {}
    for ticker, _, _ in ASSET_METADATA:          # universe order, not dict order
        weight = float(weights.get(ticker, 0.0) or 0.0)
        if weight == 0.0:
            continue
        name, category = meta[ticker]
        entry = byName.get(category)
        if entry is None:
            entry = {'name': category, 'weightPct': 0.0, 'assets': []}
            byName[category] = entry
            categories.append(entry)
        entry['weightPct'] += weight * 100.0
        entry['assets'].append({'reportingName': name, 'weightPct': weight * 100.0})
    if not categories:
        raise ValueError('every weight is zero: there is nothing to bake')
    return categories


def readMetrics(portfolio) -> dict:
    """The three headline figures, as percents (the Sharpe ratio is a ratio)."""
    return {
        'estimatedReturnPct': float(portfolio.get_total_return()) * 100.0,
        'volatilityPct': float(portfolio.get_risk()) * 100.0,
        'sharpe': float(portfolio.get_sharpe_ratio()),
    }


def readStress(portfolio) -> list:
    """One row per stress period, in whatever order the engine returns them.
    The tool renders the rows it is given and never assumes a fixed list."""
    return [{'period': str(period),
             'nominalPct': float(entry.total) * 100.0,
             'realPct': float(entry.real) * 100.0}
            for period, entry in portfolio.get_factor_stress_tests().items()]


def readPremia(portfolio) -> list:
    """VaR, CVaR and probability of loss, three horizons each: nine rows.

    NOTE THE SIGN. The engine returns VaR and CVaR as POSITIVE magnitudes; they
    are NEGATED here, because the screen reads them as losses and the workbook
    flips them back to positive when it prints. A correct payload therefore has
    a NEGATIVE nominalPct on both VaR rows and a POSITIVE one on probability of
    loss. Copy this as it stands or the two surfaces will disagree.
    """
    varPol = portfolio.get_portfolio_var_pol(confidence=0.99, loss=0)
    measures = (
        ('Value at Risk with 99% Confidence', varPol.get_VaR, 'loss', -1.0),
        ('Conditional Value at Risk with 99% Confidence', varPol.get_CVaR, 'loss', -1.0),
        ('Probability of Loss', varPol.get_PoL, 'probability', 1.0),
    )
    rows = []
    for group, read, kind, sign in measures:
        for horizon, label in VAR_HORIZONS:
            nominal, real = read(horizon)
            rows.append({
                'group': group,
                'horizon': label,
                'label': group + ' · ' + label,
                'nominalPct': sign * float(nominal) * 100.0,
                'realPct': sign * float(real) * 100.0,
                'kind': kind,
            })
    return rows


def assertFinite(payload) -> None:
    """No NaN and no infinity may reach a rendered surface. The tool refuses a
    payload carrying one; better to fail here, where you can see which."""
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, '{}.{}'.format(path, k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, '{}[{}]'.format(path, i))
        elif isinstance(node, float) and not math.isfinite(node):
            raise ValueError('non-finite value at {}'.format(path))
    walk(payload, '$')


def buildPayload(entry, currency, hedging, analyticsCurrency=None,
                 tolerance=WEIGHT_TOLERANCE) -> dict:
    """One PortfolioResult, exactly as the tool serves it."""
    portfolio = entry.portfolio() if callable(entry.portfolio) else entry.portfolio
    if portfolio is None:
        raise ValueError('{}: no SAAPortfolio and no callable to build one'.format(
            keyString(currency, entry.riskLevel, entry.allocationType,
                      entry.excludeRealAssets)))
    payload = {
        'key': {
            'currency': currency,
            'riskLevel': entry.riskLevel,
            'allocationType': (None if entry.allocationType in (None, NA)
                               else entry.allocationType),
            'excludeRealAssets': (None if entry.allocationType in (None, NA)
                                  else bool(entry.excludeRealAssets)),
        },
        'keyStr': keyString(currency, entry.riskLevel, entry.allocationType,
                            entry.excludeRealAssets),
        'name': displayName(currency, entry.riskLevel, entry.allocationType,
                            entry.excludeRealAssets),
        'header': header(entry.riskLevel, entry.allocationType, entry.excludeRealAssets),
        'categories': categoryRows(entry.weights, tolerance),
        'metrics': readMetrics(portfolio),
        'stress': readStress(portfolio),
        'premia': readPremia(portfolio),
    }
    if analyticsCurrency and analyticsCurrency != currency:
        # the figures were computed in another currency; the workbook says so
        payload['analyticsCurrency'] = analyticsCurrency
    assertFinite(payload)
    return payload


# --------------------------------------------------------------------------
# The slice
# --------------------------------------------------------------------------

def _writeJson(path, data) -> None:
    """Whole file, temp name, rename - so an interrupted run leaves no
    half-written slice for the service to read."""
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=1, sort_keys=False)
    os.replace(temporary, path)


def bakeSlice(currency, hedging, portfolios, outDir, analyticsCurrency=None,
              dataversion=None, tolerance=WEIGHT_TOLERANCE, quiet=False) -> dict:
    """Bake every portfolio of one (currency, hedging) slice into one file.

    Returns the slice: {keyStr: payload}. The file is REPLACED, not merged, so
    a key dropped from *portfolios* disappears rather than lingering.

    *analyticsCurrency* is for a book computed in another currency (the tool's
    own bake ran GBP, CHF and EUR in a USD context); every payload is stamped
    with it and the workbook then says so. *dataversion* is free text recorded
    in the manifest and shown in the page's footer.
    """
    if hedging not in HEDGING_BASES:
        raise ValueError('{!r} is not a hedging basis the tool offers: {}'.format(
            hedging, ', '.join(HEDGING_BASES)))
    portfolios = list(portfolios)
    if not portfolios:
        raise ValueError('nothing to bake')

    started = datetime.datetime.now()
    slice_ = {}
    for entry in portfolios:
        payload = buildPayload(entry, currency, hedging, analyticsCurrency, tolerance)
        key = payload['keyStr']
        if key in slice_:
            raise ValueError('{} appears twice in this slice'.format(key))
        slice_[key] = payload
        if not quiet:
            print('   %-30s %s' % (key, payload['name']))

    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    name = sliceFilename(currency, hedging)
    _writeJson(os.path.join(outDir, name), slice_)
    _updateManifest(outDir, name, currency, hedging, slice_, analyticsCurrency,
                    dataversion, (datetime.datetime.now() - started).total_seconds())
    if not quiet:
        print('\n%s: %d portfolios' % (name, len(slice_)))
        levels = []
        for key in slice_:
            level = key.split('|')[1]
            if level not in levels:
                levels.append(level)
        print('   risk levels     : %s' % ', '.join(levels))
        print('   allocation types: %s' % ', '.join(
            sorted({key.split('|')[2] for key in slice_})))
    return slice_


def _updateManifest(outDir, name, currency, hedging, slice_, analyticsCurrency,
                    dataversion, seconds) -> None:
    """Merge this slice's entry into manifest.json and recount the store.

    The service reads a store with no manifest perfectly well - it counts the
    slice files instead - but then the page's footer has no bake date. This
    writes the few fields the footer and the console actually read.
    """
    path = os.path.join(outDir, 'manifest.json')
    manifest = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            manifest = json.load(handle)
    stamp = datetime.datetime.now().isoformat(timespec='seconds')
    slices = manifest.setdefault('slices', {})
    slices[name] = {
        'currency': currency,
        'hedging': hedging,
        'baked': len(slice_),
        'complete': True,
        'analyticsCurrency': analyticsCurrency or currency,
        'currencySubstituted': bool(analyticsCurrency and analyticsCurrency != currency),
        'seconds': round(seconds, 1),
        'describe': {'adapter': 'live', 'source': 'SAA analytics',
                     'dataversion': dataversion or 'dataversion unrecorded',
                     'asOf': stamp[:10]},
        'bakedAt': stamp,
    }
    manifest['updatedAt'] = stamp
    manifest['portfoliosBaked'] = sum(e.get('baked', 0) for e in slices.values())
    manifest['currencies'] = sorted({e['currency'] for e in slices.values()})
    manifest['currencySubstitutions'] = {
        e['currency']: e['analyticsCurrency'] for e in slices.values()
        if e.get('currencySubstituted')}
    _writeJson(path, manifest)


# --------------------------------------------------------------------------
# A dry run, so the shape can be seen without the analytics library
# --------------------------------------------------------------------------

class _StubPortfolio:
    """Answers the five calls a bake makes, with obvious made-up numbers."""

    class _Stress:
        def __init__(self, total, real):
            self.total, self.real = total, real

    class _VarPol:
        # POSITIVE magnitudes, which is what the real engine returns; readPremia
        # negates them, so the payload ends up with the negative VaR the tool
        # expects. Getting this backwards is the easiest mistake to make here.
        def get_VaR(self, horizon):
            return (0.05 + horizon / 1000.0, 0.06 + horizon / 1000.0)

        def get_CVaR(self, horizon):
            return (0.07 + horizon / 1000.0, 0.08 + horizon / 1000.0)

        def get_PoL(self, horizon):
            return (0.30 - horizon / 1000.0, 0.32 - horizon / 1000.0)

    def __init__(self, riskiness=0.5):
        self._r = riskiness

    def get_total_return(self):
        return 0.03 + 0.05 * self._r

    def get_risk(self):
        return 0.02 + 0.11 * self._r

    def get_sharpe_ratio(self):
        return 0.40 + 0.10 * self._r

    def get_factor_stress_tests(self):
        return {'Oil Embargo': self._Stress(-0.18 * self._r, -0.30 * self._r),
                'Financial Crisis': self._Stress(-0.24 * self._r, -0.22 * self._r)}

    def get_portfolio_var_pol(self, confidence, loss):
        return self._VarPol()


def _demo(outDir):
    """Bake a whole slice from stub portfolios: three risk levels, the
    allocation types under each, and an all-equity book."""
    bonds, equity = 'LHTRYIN', 'FRUS1GR'
    alts = ('CSTEVDH', 'PE_BUYOUT', 'PA_REAL_ESTATE')

    def mix(equityShare, withAlts=True, withRealAssets=True):
        weights = {}
        share = equityShare
        if withAlts:
            weights['CSTEVDH'] = 0.05
            weights['PE_BUYOUT'] = 0.05
            share -= 0.10
        if withRealAssets:
            weights['PA_REAL_ESTATE'] = 0.04
            share -= 0.04
        weights[equity] = round(share, 6)
        weights[bonds] = round(1.0 - sum(weights.values()), 6)
        return weights

    entries = []
    for level, equityShare in (('LowVol', 0.30), ('Moderate', 0.55), ('Agg', 0.80)):
        riskiness = equityShare
        for allocation, withAlts, withRA in (('Full', True, True),
                                             ('Full', True, False),
                                             ('Core', True, True),
                                             ('ex-Alts', False, True),
                                             ('ex-HFs', True, True)):
            weights = mix(equityShare, withAlts, withRA)
            entries.append(Portfolio(
                riskLevel=level, allocationType=allocation,
                excludeRealAssets=not withRA, weights=weights,
                portfolio=lambda r=riskiness: _StubPortfolio(r)))
    entries.append(Portfolio(riskLevel='All Equity', allocationType=None,
                             excludeRealAssets=None, weights={equity: 1.0},
                             portfolio=lambda: _StubPortfolio(1.0)))
    bakeSlice('USD', 'Hedged', entries, outDir, dataversion='demo, not real analytics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--demo', action='store_true',
                        help='bake a whole slice from stub portfolios')
    parser.add_argument('--out', default='./baked', help='the bake store directory')
    args = parser.parse_args()
    if args.demo:
        _demo(args.out)
    else:
        parser.print_help()
