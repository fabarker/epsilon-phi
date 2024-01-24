import pandas as pd
import numpy as np


class OptionAttributer(object):

    _DATA_PATH = '/Users/francisbarker/Desktop/SPX Options 1.csv'
    _DAYS_PER_YEAR = 365.25
    def __init__(self, start_date, end_date):
        self._process_data(start_date,
                           end_date)

    @property
    def dates(self):
        return self._ds.index.get_level_values('date').unique()
    @property
    def unique_strikes(self):
        return np.array(self._ds.index.get_level_values('Z').unique()).reshape(1, -1)
    @property
    def unique_maturities(self):
        return np.array(self._ds.index.get_level_values('T').unique()).reshape(1, -1)
    @property
    def spot(self):
        return self._ds.get('s').droplevel([1, 2]).drop_duplicates().reindex(self.dates).ffill().to_frame().copy()
    @property
    def sig(self):
        return self._ds.get('sig').to_frame().unstack([1, 2]).get('sig').reindex(self.dates).sort_index(axis=1, level=1).copy()
    @property
    def dsig(self):
        return self._ds.get('dsig').to_frame().unstack([1, 2]).get('dsig').reindex(self.dates).sort_index(axis=1, level=1).copy()
    @property
    def sig_sq(self):
        return np.power(self.sig, 2).copy()
    @property
    def atm_sig(self):
        return self.sig.get(0).sort_index(axis=1).copy()
    @property
    def atm_dsig(self):
        return self.dsig.get(0).sort_index(axis=1).copy()

    @property
    def atm_sig_sq(self):
        return np.power(self.sig, 2).get(0).copy()

    def _process_data(self, start_date, end_date):

        _df = pd.read_csv('/Users/francisbarker/Desktop/SPX Options 1.csv')
        _df['date'] = pd.to_datetime(_df['date'])
        _df = _df.set_index(['date', 'Z', 'T'], drop=True)

        keep_rows = np.logical_and(_df.index.get_level_values('date') >= pd.to_datetime(start_date),
                                   _df.index.get_level_values('date') <= pd.to_datetime(end_date))

        self._ds = _df.iloc[keep_rows].copy()
        self._start_date = self._ds.index.get_level_values('date').min()
        self._end_date = self._ds.index.get_level_values('date').max()

    def get_mu(self, maturity=None):

        if not hasattr(self, '_mu'):

            Asq = self.atm_sig_sq
            Asq_tau = Asq * Asq.columns.to_numpy().reshape(1, -1)

            mu_ = ((1/2) * Asq.diff(axis=1) / Asq_tau.diff(axis=1)).dropna(axis=1)
            X = 0.5 * Asq_tau.columns.values[0:-1] + 0.5 * Asq_tau.columns.values[1:]
            mu_.columns = X

            mu_[[Asq_tau.columns]] = np.nan
            mu = mu_.sort_index(axis=1).interpolate(axis=1)
            mu[mu.columns[0]] = mu_[mu_.columns[0]]
            self._mu = mu.get(Asq.columns)

        return self._mu.get(maturity, self._mu.copy())

    def run_implied_vol_prediction_regressions(self):

        # X is the implied vol drift term
        # Y is the change in implied vol oevr the next time step
        # Therefore these are prediction regressions over 1 day

        mu = self.get_mu()
        intercept = pd.DataFrame(np.ones(mu.shape[0]), columns=['constant'], index=mu.index)

        regstats = [pd.DataFrame()]
        for col in mu.columns:

            mu_col_ = pd.concat((intercept, mu.get(col)), axis=1)
            y_ = self.atm_dsig.get(col)

            common_dates = np.intersect1d(mu_col_.index, y_.index)
            reg = np.linalg.lstsq(mu_col_.loc[common_dates],
                                  y_.loc[common_dates], rcond=None)

            res = y_ - mu_col_ @ reg[0]
            regstats.extend([pd.DataFrame([reg[0][0], reg[0][1],  1 - np.var(res) / np.var(y_)],
                               index=['alpha', 'beta', 'rsq'], columns=[col])])
        return pd.concat(regstats, axis=1)

    def run_pca(self, start_date=None, end_date=None):

        # X is the implied vol drift term
        # Y is the change in implied vol oevr the next time step
        # Therefore these are prediction regressions over 1 day

        cov_mat = self.dsig[start_date:end_date].cov()
        L, V = np.linalg.eig(cov_mat)
        return pd.DataFrame(V, columns=L, index=self.dsig.columns).sort_index(axis=1, ascending=False)


if __name__ == "__main__":
    self = OptionAttributer('31-Dec-1990', '31-Dec-2025')
    self.A
