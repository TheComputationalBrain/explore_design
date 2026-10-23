"""
Ideal Observer model — sklearn-style estimator.
Assumes obs shape: n_options x n_trials (NaN = unobserved).
"""

import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# --- Core computations ---


def _transmat(vol, n_levels):
    T = np.eye(n_levels) * (1 - vol)
    T[T == 0] = vol / (n_levels - 1)
    return T


def _uniform_prior(n_levels, n_options=1):
    prior = np.ones(n_levels) / n_levels
    return np.tile(prior[:, None], (1, n_options)) if n_options > 1 else prior


def _likelihood(obs, latent_levels, sd):
    if np.isnan(obs):
        return np.ones(len(latent_levels))
    return sps.norm.pdf(obs, latent_levels, sd)


def _posterior(obs, sd, latent_levels, prior):
    lik = _likelihood(obs, latent_levels, sd)
    post = lik * prior
    return post / post.sum()


def _predict_prior(post, vol, n_levels):
    return _transmat(vol, n_levels) @ post


# --- Learning params parsing (mirrors delta rule style) ---


def _parse_learning_params(learning_params: dict) -> pd.DataFrame:
    rows = []
    for key, val in learning_params.items():
        parts = key.split("_", 1)
        row = {"base": parts[0], "param": val}
        if len(parts) > 1:
            for split in parts[1].split("_"):
                split_id, split_level = split.rsplit(".", 1)
                row[split_id] = split_level
        rows.append(row)
    return pd.DataFrame(rows).fillna(value=pd.NA)


def _get_splitids(spec: pd.DataFrame) -> list:
    return [c for c in spec.columns if c not in ("base", "param")]


def _get_single_vol(
    spec,
    splitids,
    obs,
    prior,
    latent_levels,
    trial_conditions=None,
    outcome_values=None,
    outcome_prior=None,
):
    mask = pd.Series([True] * len(spec), index=spec.index)

    if "cu" in splitids:
        mask &= spec["cu"] == ("c" if not np.isnan(obs) else "u")

    if "PE.sign" in splitids:
        expectation = latent_levels @ prior
        pe = obs - expectation
        sign = (
            "pos"
            if pe > 0
            else "neg"
            if pe < 0
            else np.random.choice(["pos", "neg"])
        )
        mask &= spec["PE.sign"] == sign

    if "PEO.sign" in splitids:
        if outcome_prior is None or outcome_values is None:
            raise ValueError(
                "PEO.sign split requires return_outcome_dist=True and outcome_range."
            )
        expectation = outcome_values @ outcome_prior
        pe = obs - expectation
        sign = (
            "pos"
            if pe > 0
            else "neg"
            if pe < 0
            else np.random.choice(["pos", "neg"])
        )
        mask &= spec["PEO.sign"] == sign

    if trial_conditions is not None:
        for col in splitids:
            if col not in ("cu", "PE.sign") and col in trial_conditions.index:
                mask &= spec[col] == str(trial_conditions[col])

    return float(spec.loc[mask, "param"].values[0])


def _latent2outcome(
    lat_dist, latent_levels, outcome_values, sd, normalize=True
):
    """Mixture of Gaussians: sum_k P(level_k) * N(outcome | level_k, sd)"""
    out = np.stack(
        [sps.norm.pdf(outcome_values, loc=lv, scale=sd) for lv in latent_levels]
    )  # (n_levels, n_outcomes)
    result = lat_dist @ out  # (n_outcomes,)
    return result / result.sum() if normalize else result


# --- Main sequence function ---


def ideal_observer(
    obs,
    latent_levels,
    sd,
    learning_params,
    conditions=None,
    priors_init=None,
    return_outcome_dist=False,
    outcome_range=None,
    normalize_outcome_dist=True,
):

    n_options, n_trials = obs.shape
    latent_levels = np.asarray(latent_levels)
    n_levels = len(latent_levels)
    sd = np.ones_like(obs) * sd if np.isscalar(sd) else np.asarray(sd)
    priors_init = (
        _uniform_prior(n_levels, n_options)
        if priors_init is None
        else np.asarray(priors_init)
    )
    conditions = pd.DataFrame(conditions) if conditions is not None else None

    spec = _parse_learning_params(learning_params)
    splitids = _get_splitids(spec)

    pred_post = np.full((n_levels, n_options, n_trials + 1), np.nan)
    prev_post = pred_post.copy()
    pred_post[:, :, 0] = priors_init

    if return_outcome_dist:
        outcome_values = np.arange(outcome_range[0], outcome_range[1] + 1)
        n_out = len(outcome_values)
        pred_out = np.full((n_out, n_options, n_trials + 1), np.nan)
        prev_out = pred_out.copy()
        # init outcome prior from latent prior
        for iopt in range(n_options):
            pred_out[:, iopt, 0] = _latent2outcome(
                priors_init[:, iopt] if n_options > 1 else priors_init,
                latent_levels,
                outcome_values,
                sd[iopt, 0],
                normalize_outcome_dist,
            )
    else:
        outcome_values = None

    for iopt in range(n_options):
        for itrial in range(n_trials):
            prior = pred_post[:, iopt, itrial]
            o = obs[iopt, itrial]
            s = sd[iopt, itrial]
            trial_cond = (
                conditions.loc[itrial] if conditions is not None else None
            )
            outcome_prior = (
                pred_out[:, iopt, itrial] if return_outcome_dist else None
            )

            vol = _get_single_vol(
                spec,
                splitids,
                o,
                prior,
                latent_levels,
                trial_cond,
                outcome_values,
                outcome_prior,
            )
            post = _posterior(o, s, latent_levels, prior)
            next_prior = _predict_prior(post, vol, n_levels)

            prev_post[:, iopt, itrial + 1] = post
            pred_post[:, iopt, itrial + 1] = next_prior

            if return_outcome_dist:
                prev_out[:, iopt, itrial + 1] = _latent2outcome(
                    post,
                    latent_levels,
                    outcome_values,
                    s,
                    normalize_outcome_dist,
                )
                pred_out[:, iopt, itrial + 1] = _latent2outcome(
                    next_prior,
                    latent_levels,
                    outcome_values,
                    s,
                    normalize_outcome_dist,
                )

    posteriors = {
        "predictive_posterior": pred_post,
        "prevol_posterior": prev_post,
    }
    if return_outcome_dist:
        posteriors["predictive_outcome_distribution"] = pred_out
        posteriors["prevol_outcome_distribution"] = prev_out
    return posteriors


# --- Sklearn estimator ---


class IdealObserverEstimator(BaseEstimator):
    """
    Parameters
    ----------
    latent_levels : array-like
        Possible reward levels (e.g. [30, 50, 70]).
    sd : float or array (n_options x n_trials)
        Observation noise.
    learning_params : dict
        Volatility, optionally split (e.g. {'vol': 0.1} or {'vol_cu.c': 0.2, 'vol_cu.u': 0.05}).
    option_cols : list of str
        Columns in X with per-option observations.
    condition_cols : list of str, optional
        Columns in X used as trial conditions for split volatilities.
    return_outcome_dist : bool, optional
        Whether to return the predictive outcome distribution.
    outcome_range : tuple, optional
        Range of possible outcome values.
    """

    def __init__(
        self,
        latent_levels,
        sd,
        learning_params,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
        return_outcome_dist=False,
        outcome_range=None,
        normalize_outcome_dist=True,
    ):
        self.latent_levels = latent_levels
        self.sd = sd
        self.learning_params = learning_params
        self.option_cols = option_cols
        self.condition_cols = condition_cols
        self.return_outcome_dist = return_outcome_dist
        self.outcome_range = outcome_range
        self.normalize_outcome_dist = normalize_outcome_dist

    def _parse_X(self, X):
        obs = X[list(self.option_cols)].to_numpy().T  # (n_options, n_trials)
        conditions = (
            X[list(self.condition_cols)].reset_index(drop=True)
            if self.condition_cols
            else None
        )
        return obs, conditions

    def fit(self, X: pd.DataFrame, y=None):
        obs, conditions = self._parse_X(X)
        self.posteriors_ = ideal_observer(
            obs=obs,
            latent_levels=self.latent_levels,
            sd=self.sd,
            learning_params=self.learning_params,
            conditions=conditions,
            return_outcome_dist=self.return_outcome_dist,
            outcome_range=self.outcome_range,
            normalize_outcome_dist=self.normalize_outcome_dist,
        )
        self.option_cols_ = list(self.option_cols)
        return self

    def predict(self, X: pd.DataFrame):
        """
        Returns the predictive posterior expectation per option per trial.
        Shape: (n_trials, n_options).
        """
        check_is_fitted(self, "posteriors_")
        levels = np.asarray(self.latent_levels)
        pred = self.posteriors_["predictive_posterior"][
            :, :, :-1
        ]  # (n_levels, n_options, n_trials)
        expectations = np.einsum(
            "l,lot->ot", levels, pred
        ).T  # (n_trials, n_options)
        cols = ["IO__" + c for c in self.option_cols_]
        return pd.DataFrame(expectations, columns=cols, index=X.index)

    def fit_predict(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).predict(X)


def set_unobserved_value(quantity, unobserved_value=0):
    quantity[np.isnan(quantity)] = unobserved_value
    return quantity


def get_missing(obs):
    return np.where(np.all(np.isnan(obs), axis=0))[0]


def get_outcome_surprise(
    outcome_distribution, obs, outcome_range, unobserved_value=0
):
    n_options, n_trials = obs.shape
    surprise = np.full((n_options, n_trials), np.nan)
    for iopt in range(n_options):
        for itrial in range(n_trials):
            if ~np.isnan(obs[iopt, itrial]):
                # Offset the index by the min of outcome_range
                idx = int(obs[iopt, itrial]) - outcome_range[0]
                surprise[iopt, itrial] = -np.log(
                    outcome_distribution[idx, iopt, itrial]
                )
    surprise = set_unobserved_value(surprise, unobserved_value)
    surprise[:, get_missing(obs)] = np.nan
    return surprise


class SurpriseEstimator(IdealObserverEstimator):
    """
    Extends IdealObserverEstimator to predict outcome surprise (-log P(obs)).
    Requires return_outcome_dist=True and outcome_range to be set.

    Parameters
    ----------
    unobserved_value : int or 'nan'
        Value to fill for unchosen options. Int (e.g. 0) or anything non-int → NaN.
    All other parameters inherited from IdealObserverEstimator.
    """

    def __init__(
        self,
        latent_levels,
        sd,
        learning_params,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
        outcome_range=None,
        normalize_outcome_dist=True,
        unobserved_value=0,
    ):
        super().__init__(
            latent_levels=latent_levels,
            sd=sd,
            learning_params=learning_params,
            option_cols=option_cols,
            condition_cols=condition_cols,
            return_outcome_dist=True,
            outcome_range=outcome_range,
            normalize_outcome_dist=normalize_outcome_dist,
        )
        self.unobserved_value = unobserved_value

    def predict(self, X: pd.DataFrame):
        """
        Returns surprise (-log P(obs)) per option per trial.
        Uses the predictive outcome distribution (prior to observing the trial).
        Shape: (n_trials, n_options).
        """
        check_is_fitted(self, "posteriors_")
        obs, _ = self._parse_X(X)
        # predictive_outcome_distribution[:,  :, 1:] aligns index 0 with trial 0
        # (the distribution formed before seeing trial t's outcome)
        out_dist = self.posteriors_["predictive_outcome_distribution"][
            :, :, :-1
        ]
        surprise = get_outcome_surprise(
            out_dist, obs, self.outcome_range, self.unobserved_value
        )
        cols = ["Surprise__" + c for c in self.option_cols_]
        return pd.DataFrame(surprise.T, columns=cols, index=X.index)


class UncertaintyEstimator(IdealObserverEstimator):
    """
    Extends IdealObserverEstimator to predict estimation uncertainty
    (1 - max posterior) on the chosen option per trial.

    Parameters
    ----------
    unobserved_value : int or 'nan'
        Value to fill for unchosen options (same convention as SurpriseEstimator).
    All other parameters inherited from IdealObserverEstimator.
    """

    def __init__(
        self,
        latent_levels,
        sd,
        learning_params,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
        unobserved_value=0,
    ):
        super().__init__(
            latent_levels=latent_levels,
            sd=sd,
            learning_params=learning_params,
            option_cols=option_cols,
            condition_cols=condition_cols,
        )
        self.unobserved_value = unobserved_value

    def predict(self, X: pd.DataFrame):
        check_is_fitted(self, "posteriors_")
        obs, _ = self._parse_X(X)
        n_options, n_trials = obs.shape

        # prevol_posterior: (n_levels, n_options, n_trials+1)
        # index 1: onwards aligns with trial 0 (after observing trial t)
        prevol = self.posteriors_["prevol_posterior"][:, :, 1:]

        uncertainty = np.full((n_options, n_trials), np.nan)
        for iopt in range(n_options):
            for itrial in range(n_trials):
                if ~np.isnan(obs[iopt, itrial]):
                    uncertainty[iopt, itrial] = 1 - np.max(
                        prevol[:, iopt, itrial]
                    )

        uncertainty = set_unobserved_value(uncertainty, self.unobserved_value)
        uncertainty[:, get_missing(obs)] = np.nan

        cols = ["Eu_ch__" + c for c in self.option_cols_]
        return pd.DataFrame(uncertainty.T, columns=cols, index=X.index)

    def fit_predict(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).predict(X)


class PredictionErrorEstimator(IdealObserverEstimator):
    """
    Extends IdealObserverEstimator to predict signed prediction error
    (observed outcome - predictive posterior expectation) per option per trial.

    Uses the predictive posterior formed *before* the trial's outcome is observed,
    i.e. the same expectation returned by IdealObserverEstimator.predict.

    Parameters
    ----------
    unobserved_value : int or 'nan'
        Value to fill for unchosen options (same convention as SurpriseEstimator
        and UncertaintyEstimator).
    All other parameters inherited from IdealObserverEstimator.
    """

    def __init__(
        self,
        latent_levels,
        sd,
        learning_params,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
        unobserved_value=0,
    ):
        super().__init__(
            latent_levels=latent_levels,
            sd=sd,
            learning_params=learning_params,
            option_cols=option_cols,
            condition_cols=condition_cols,
        )
        self.unobserved_value = unobserved_value

    def predict(self, X: pd.DataFrame):
        """
        Returns prediction error (obs - E[outcome]) per option per trial.
        Shape: (n_trials, n_options).
        """
        check_is_fitted(self, "posteriors_")
        obs, _ = self._parse_X(X)

        levels = np.asarray(self.latent_levels)
        pred = self.posteriors_["predictive_posterior"][
            :, :, :-1
        ]  # (n_levels, n_options, n_trials)
        expectations = np.einsum(
            "l,lot->ot", levels, pred
        )  # (n_options, n_trials)

        pe = obs - expectations
        pe = set_unobserved_value(pe, self.unobserved_value)
        pe[:, get_missing(obs)] = np.nan

        cols = ["Pe__" + c for c in self.option_cols_]
        return pd.DataFrame(pe.T, columns=cols, index=X.index)

    def fit_predict(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).predict(X)


class ConfidenceEstimator(IdealObserverEstimator):
    """
    Extends IdealObserverEstimator to predict the confidence in the
    prediction of the next outcome, formalized as the log precision of the
    predictive distribution over the latent state:

        confidence_t = -log( var(θ_{t+1} | y_{1:t}, υ) )

    The predictive distribution p(θ_{t+1} | y_{1:t}, υ) is the posterior
    after trial t propagated through the transition matrix
    p(θ_{t+1} | θ_t, υ) (i.e. the predictive posterior formed *before*
    observing trial t's outcome, index t of "predictive_posterior").

    Parameters
    ----------
    unobserved_value : int or 'nan'
        Value to fill for unchosen options (same convention as the other
        estimators in this module).
    All other parameters inherited from IdealObserverEstimator.
    """

    _chosen = True
    _prefix = "Conf_ch__"

    def __init__(
        self,
        latent_levels,
        sd,
        learning_params,
        option_cols=("obsA", "obsB"),
        condition_cols=None,
        unobserved_value=0,
    ):
        super().__init__(
            latent_levels=latent_levels,
            sd=sd,
            learning_params=learning_params,
            option_cols=option_cols,
            condition_cols=condition_cols,
        )
        self.unobserved_value = unobserved_value

    def predict(self, X: pd.DataFrame):
        """
        Returns confidence (-log variance of the predictive distribution
        over the latent state) per option per trial.
        Shape: (n_trials, n_options).
        """
        check_is_fitted(self, "posteriors_")
        obs, _ = self._parse_X(X)
        n_options, n_trials = obs.shape

        levels = np.asarray(self.latent_levels)
        # predictive_posterior[:, :, 0:n_trials] aligns index t with trial t
        # (the distribution over θ_{t+1} formed before seeing trial t's outcome)
        pred = self.posteriors_["predictive_posterior"][:, :, :-1]

        confidence = np.full((n_options, n_trials), np.nan)
        for iopt in range(n_options):
            for itrial in range(n_trials):
                if np.isnan(obs[iopt, itrial]) != self._chosen:
                    p = pred[:, iopt, itrial]
                    mean = levels @ p
                    var = float((levels**2) @ p) - mean**2
                    # Guard a degenerate (zero) variance, which would give -log(0) = inf
                    var = max(var, 1e-12)
                    confidence[iopt, itrial] = -np.log(var)

        confidence = set_unobserved_value(confidence, self.unobserved_value)
        confidence[:, get_missing(obs)] = np.nan

        cols = [self._prefix + c for c in self.option_cols_]
        return pd.DataFrame(confidence.T, columns=cols, index=X.index)


class UnchosenConfidenceEstimator(ConfidenceEstimator):
    """
    Same as ConfidenceEstimator, but the confidence is returned for the
    unchosen option(s) on each trial; the chosen option is filled with
    unobserved_value.
    """

    _chosen = False
    _prefix = "Conf_unch__"
