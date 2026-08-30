import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class HullWhiteCalibration:
    """
    Calibrated parameters of a single-factor Hull-White process.

    Attributes
    ----------
    mean_reversion : float
        Per-period mean reversion speed ``a``. The deviation process decays by a
        factor of ``(1 - a)`` each step, so ``a`` is expressed in the frequency of
        the calibration sample (monthly for the SAA simulation).
    sigma : float
        Standard deviation of the innovations.
    shocks : np.ndarray
        Centred innovation pool of shape ``(N,)``, aligned one-for-one with the
        observations the model was calibrated on so that bootstrap indices drawn
        against the historical sample can index it directly.
    long_run_mean : float
        Unconditional mean implied by the calibration. Used for diagnostics only:
        the simulated anchor is the supplied ``mu`` path, not this value.
    """

    mean_reversion: float
    sigma: float
    shocks: np.ndarray
    long_run_mean: float

    @property
    def decay(self):
        """Per-period persistence of the deviation process, ``1 - a``."""
        return 1.0 - self.mean_reversion

    @property
    def half_life(self):
        """Number of periods for a deviation from the anchor to halve."""
        return np.log(2.0) / -np.log(self.decay)

    @property
    def stationary_sd(self):
        """Limiting standard deviation of the deviation process."""
        return self.sigma / np.sqrt(1.0 - self.decay ** 2)

    def sd_term_structure(self, n_periods):
        """
        Closed-form standard deviation of the unfloored process by horizon.

        Provides an analytic target to validate the simulated panel against.

        Parameters
        ----------
        n_periods : int
            Number of periods to evaluate.

        Returns
        -------
        np.ndarray
            Standard deviations of shape ``(n_periods,)``, starting at zero.
        """
        t = np.arange(n_periods)
        return self.sigma * np.sqrt((1.0 - self.decay ** (2 * t)) / (1.0 - self.decay ** 2))


class ShiftedHullWhite:
    """
    Single-factor Hull-White short rate model written in shift decomposition form.

    The rate is split into a deterministic anchor and a zero-mean deviation:

    .. math::

        r(t) = \\varphi(t) + x(t), \\qquad
        x(t) = (1 - a)\\, x(t-1) + \\varepsilon(t), \\qquad x(0) = 0

    Because :math:`\\mathbb{E}[x(t)] = 0` for every ``t``, the cross-sectional mean of
    the simulated panel equals :math:`\\varphi(t)` by construction rather than by any
    post-hoc re-centring of the paths.

    Over a finite panel the realised mean of the deviation is zero only up to sampling
    error, so a deterministic shift ``c(t)``, common to every path, is solved at each
    step to hold the anchor exactly. Being common it leaves the cross-sectional
    dispersion untouched, and it can be stored alongside the calibration, after which
    the paths are independent draws from a well defined process.

    This is the standard shifted (extended Vasicek) formulation: the equivalent
    drift form is :math:`dr = (\\theta(t) - a r)\\,dt + \\sigma\\,dW` with
    :math:`\\theta(t) = \\varphi'(t) + a\\,\\varphi(t)`.

    Innovations are not assumed Gaussian. ``simulate`` consumes a panel of
    bootstrapped empirical residuals, so the block bootstrap and current
    environment conditioning applied upstream carry through unchanged and the
    simulated dispersion retains the skew and tail behaviour of the sample.

    A lower bound is applied as a shadow rate rather than by clipping the paths.
    Clipping raises the mean above the anchor whenever the bound binds; instead a
    deterministic shift ``c(t)`` is solved at each step so that

    .. math::

        \\mathbb{E}\\big[\\max(f,\\ \\varphi(t) + c(t) + x(t))\\big] = \\varphi(t)

    which holds the anchor exactly in a low rate regime where a large share of the
    cross-section sits on the floor.
    """

    MIN_MEAN_REVERSION = 1.0e-8
    MAX_MEAN_REVERSION = 2.0 - 1.0e-8
    SHIFT_TOLERANCE = 1.0e-13
    SHIFT_MAX_ITER = 64

    @staticmethod
    def calibrate(
            values,
            long_run_mean=None,
            pad='edge',
    ) -> HullWhiteCalibration:
        """
        Estimate the mean reversion speed and extract the empirical innovation pool.

        A single regression is run over the whole sample. The mean reversion speed
        is a property of the process rather than of the anchor it is simulated
        against, so it is deliberately not re-estimated per simulation step.

        Parameters
        ----------
        values : array_like
            Historical rate observations of shape ``(N,)``, in the units the
            simulation runs in (monthly rates for the SAA simulation).
        long_run_mean : float, optional
            Unconditional mean to impose. When omitted an intercept is included in
            the regression and the unconditional mean is estimated jointly with the
            mean reversion speed, which leaves the residuals exactly centred.
        pad : {'edge', 'zero'}, default 'edge'
            How to fill the innovation at index 0, which has no predecessor and so
            no innovation of its own. ``'edge'`` repeats the adjacent residual so
            that every bootstrap draw is a genuine empirical shock. ``'zero'``
            reproduces the legacy behaviour of inserting a structural zero, which
            puts a point mass on "no change" into the sampling pool.

        Returns
        -------
        HullWhiteCalibration
            Calibrated parameters and the centred innovation pool.

        Raises
        ------
        ValueError
            If fewer than three observations are supplied or ``pad`` is not
            recognised.
        """

        values = np.asarray(values, dtype=float).ravel()
        if values.size < 3:
            raise ValueError('At least three observations are required to calibrate')
        if pad not in ('edge', 'zero'):
            raise ValueError("pad must be one of {'edge', 'zero'}")

        r_t = values[:-1]
        r_tp1 = values[1:]

        if long_run_mean is None:
            # r(t+1) = alpha + rho * r(t) + e, with an intercept so that the
            # residuals are mean zero by construction
            design = np.column_stack([np.ones_like(r_t), r_t])
            (alpha, rho), *_ = np.linalg.lstsq(design, r_tp1, rcond=None)
            a_est = 1.0 - rho
            residuals = r_tp1 - (alpha + rho * r_t)
            dof = 2
            mu_bar = alpha / a_est if abs(a_est) > ShiftedHullWhite.MIN_MEAN_REVERSION else np.mean(values)
        else:
            # r(t+1) - r(t) = a * (mu_bar - r(t)) + e, regression through the origin
            mu_bar = float(long_run_mean)
            y = r_tp1 - r_t
            x = mu_bar - r_t
            a_est = np.dot(x, y) / np.dot(x, x)
            residuals = y - a_est * x
            dof = 1

        a_est = ShiftedHullWhite._validate_mean_reversion(a_est)

        fill = residuals[0] if pad == 'edge' else 0.0
        shocks = np.insert(residuals, 0, fill)

        # centre the sampled pool itself, after padding. The regression through the
        # origin leaves residuals orthogonal to the regressor but not mean zero, and
        # padding perturbs the mean of the intercept fit, so neither is centred for
        # free. E[x(t)] = 0 depends on this holding exactly.
        shocks = shocks - shocks.mean()

        return HullWhiteCalibration(
            mean_reversion=float(a_est),
            sigma=float(np.std(shocks, ddof=dof)),
            shocks=shocks,
            long_run_mean=float(mu_bar),
        )

    @staticmethod
    def simulate(
            mu_vals,
            shocks,
            mean_reversion,
            rate_floor=-np.inf,
            x0=0.0,
            return_shift=False,
    ):
        """
        Simulate rate paths anchored to ``mu_vals``.

        Parameters
        ----------
        mu_vals : array_like
            Anchor path :math:`\\varphi(t)` of shape ``(T,)``. The cross-sectional
            mean of the returned panel reproduces this path.
        shocks : np.ndarray
            Bootstrapped innovation panel of shape ``(T, nbstraps)``. Row ``t``
            holds the innovation applied moving into step ``t``; row 0 is unused
            because every path starts at the anchor.
        mean_reversion : float
            Per-period mean reversion speed ``a`` from :meth:`calibrate`.
        rate_floor : float, default ``-np.inf``
            Lower bound on the simulated rate. With no bound the anchor holds
            without any shift; otherwise a shift is solved per step.
        x0 : float or array_like, default 0.0
            Initial deviation from the anchor. Zero starts every path at
            ``mu_vals[0]``. Pass ``current_rate - mu_vals[0]`` to seed the simulation
            from the prevailing rate instead, in which case the panel means
            ``mu_vals[t] + x0 * (1 - a) ** t``: the anchor plus a start-state
            adjustment that decays at the mean reversion speed.
        return_shift : bool, default False
            Also return the solved shift path, for diagnostics.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            Panel of shape ``(T, nbstraps)``, and the shift of shape ``(T,)`` when
            ``return_shift`` is set.

        Raises
        ------
        ValueError
            If ``shocks`` is not two dimensional or its first axis does not match
            ``mu_vals``.
        """

        mu_vals = np.asarray(mu_vals, dtype=float).ravel()
        shocks = np.asarray(shocks, dtype=float)

        if shocks.ndim != 2:
            raise ValueError('shocks must be a (T, nbstraps) panel')
        if mu_vals.shape[0] != shocks.shape[0]:
            raise ValueError(
                'mu_vals has {} steps but the shock panel has {}'.format(
                    mu_vals.shape[0], shocks.shape[0])
            )

        decay = 1.0 - ShiftedHullWhite._validate_mean_reversion(mean_reversion)

        # zero-mean deviation process
        deviations = np.empty_like(shocks)
        deviations[0] = x0
        for t in range(1, deviations.shape[0]):
            deviations[t] = decay * deviations[t - 1] + shocks[t]

        paths = np.empty_like(deviations)
        shifts = np.zeros(mu_vals.shape[0])

        # a deliberate start-state offset decays deterministically towards the anchor.
        # The shift must remove the sampling error around that offset, not the offset
        # itself, or seeding from the prevailing rate would be cancelled at t = 0.
        targets = mu_vals + np.mean(x0) * decay ** np.arange(mu_vals.shape[0])

        for t in range(mu_vals.shape[0]):
            shadow = mu_vals[t] + deviations[t]
            shifts[t] = ShiftedHullWhite.solve_shift(shadow, targets[t], rate_floor)
            paths[t] = np.maximum(rate_floor, shadow + shifts[t])

        return (paths, shifts) if return_shift else paths

    @staticmethod
    def solve_shift(shadow, target, rate_floor):
        """
        Solve the shift that holds the floored cross-sectional mean on the anchor.

        Finds ``c`` such that ``mean(max(rate_floor, shadow + c)) == target``. The
        objective is continuous, convex and non-decreasing in ``c``, so a
        bracketed Newton iteration converges monotonically and terminates exactly
        once the set of paths above the floor settles.

        Parameters
        ----------
        shadow : np.ndarray
            Unfloored cross-section of shape ``(nbstraps,)``.
        target : float
            Anchor value the floored mean must reproduce.
        rate_floor : float
            Lower bound. A non-finite bound returns a zero shift.

        Returns
        -------
        float
            The solved shift.
        """

        if not np.isfinite(rate_floor):
            # unbounded: the objective is linear in c, so the shift is closed form.
            # It removes the sampling error in the realised cross-sectional mean of
            # the deviation, which is zero only in expectation over a finite panel.
            return float(target - shadow.mean())

        if target <= rate_floor:
            warnings.warn(
                'Anchor {} is not above the floor {}; the floored mean cannot reach it '
                'and every path is placed on the floor'.format(target, rate_floor),
                RuntimeWarning,
            )
            return rate_floor - shadow.max()

        # mean(max(floor, shadow + lo)) == floor <= target, and
        # mean(max(floor, shadow + hi)) == target + (target - shadow.min()) >= target
        lo = rate_floor - shadow.max()
        hi = target - shadow.min()
        tol = ShiftedHullWhite.SHIFT_TOLERANCE * max(1.0, abs(target))

        c = 0.0 if lo <= 0.0 <= hi else 0.5 * (lo + hi)
        for _ in range(ShiftedHullWhite.SHIFT_MAX_ITER):
            above = shadow + c > rate_floor
            error = np.maximum(rate_floor, shadow + c).mean() - target
            if abs(error) <= tol:
                break

            if error < 0.0:
                lo = c
            else:
                hi = c

            fraction_above = above.mean()
            step = c - error / fraction_above if fraction_above > 0.0 else hi
            c = step if lo < step < hi else 0.5 * (lo + hi)

        return float(c)

    @staticmethod
    def _validate_mean_reversion(a):
        """Clip the mean reversion speed to the range that keeps the deviation stable."""
        a = float(a)
        if not np.isfinite(a):
            raise ValueError('Mean reversion speed is not finite')

        clipped = min(max(a, ShiftedHullWhite.MIN_MEAN_REVERSION), ShiftedHullWhite.MAX_MEAN_REVERSION)
        if clipped != a:
            warnings.warn(
                'Mean reversion speed {} is outside the stable range and has been '
                'clipped to {}'.format(a, clipped),
                RuntimeWarning,
            )
        elif a >= 1.0:
            warnings.warn(
                'Mean reversion speed {} exceeds 1; the deviation process will '
                'oscillate rather than decay monotonically'.format(a),
                RuntimeWarning,
            )
        return clipped
