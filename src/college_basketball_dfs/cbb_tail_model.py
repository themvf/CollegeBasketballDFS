from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd


def _safe_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _tail_probability(threshold: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0 if threshold > mu else 1.0
    z = (float(threshold) - float(mu)) / float(sigma)
    p = 1.0 - _safe_normal_cdf(z)
    return float(max(0.0, min(1.0, p)))


def _to_numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    out = pd.to_numeric(df[col], errors="coerce")
    return out.fillna(default).astype(float)


def _fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x.shape) != 2 or x.shape[0] == 0:
        return np.zeros((x.shape[1] if len(x.shape) == 2 else 1,), dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return beta


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _team_acronym(name: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(name or "").lower())
    if not words:
        return ""
    skip = {"the", "of", "and", "at"}
    letters = [w[0] for w in words if w and w not in skip]
    return "".join(letters)


def _team_token_similarity(abbrev_token: str, full_name: str) -> float:
    token = _normalize_text(abbrev_token)
    full = _normalize_text(full_name)
    if not token or not full:
        return 0.0
    if token == full:
        return 1.0
    if token in full or full in token:
        return 0.96
    acronym = _normalize_text(_team_acronym(full_name))
    if acronym:
        if token == acronym:
            return 0.93
        if token in acronym or acronym in token:
            return 0.86

    ratio = float(SequenceMatcher(a=token, b=full).ratio())
    if len(token) >= 4:
        word_hits = 0
        words = re.findall(r"[a-z0-9]+", str(full_name or "").lower())
        for w in words:
            nw = _normalize_text(w)
            if token == nw or token in nw or nw in token:
                word_hits += 1
        if word_hits:
            ratio = max(ratio, 0.7 + (0.08 * min(2, word_hits)))
    return float(max(0.0, min(1.0, ratio)))


def fit_total_tail_model(games_df: pd.DataFrame) -> dict[str, Any]:
    base = games_df.copy()
    for col in ["total_error", "total_points", "vegas_home_margin"]:
        base[col] = pd.to_numeric(base.get(col), errors="coerce")
    if "has_total_line" in base.columns:
        base = base.loc[base["has_total_line"]]
    base = base.dropna(subset=["total_error", "total_points"]).reset_index(drop=True)
    if base.empty:
        return {
            "samples": 0,
            "total_mean": 150.0,
            "spread_abs_mean": 6.0,
            "mu_beta": [0.0, 0.0, 0.0],
            "sigma_beta": [math.log(10.0), 0.0, 0.0],
            "sigma_floor": 4.0,
            "sigma_cap": 30.0,
        }

    base["abs_spread"] = pd.to_numeric(base["vegas_home_margin"], errors="coerce").abs().fillna(0.0)
    total_mean = float(base["total_points"].mean())
    spread_abs_mean = float(base["abs_spread"].mean())
    x_total = (base["total_points"] - total_mean).astype(float)
    x_spread = (base["abs_spread"] - spread_abs_mean).astype(float)
    x = np.column_stack([np.ones(len(base)), x_total.to_numpy(), x_spread.to_numpy()])
    y = base["total_error"].astype(float).to_numpy()

    mu_beta = _fit_ols(x, y)
    mu_pred = x @ mu_beta
    residual = y - mu_pred

    # Model expected absolute residual to get game-specific sigma.
    abs_resid = np.abs(residual)
    sigma_floor = float(max(2.5, np.quantile(abs_resid, 0.1) * math.sqrt(math.pi / 2.0)))
    sigma_cap = float(max(8.0, np.quantile(abs_resid, 0.95) * math.sqrt(math.pi / 2.0)))
    target_sigma = np.log(abs_resid + 1e-3)
    sigma_beta = _fit_ols(x, target_sigma)

    return {
        "samples": int(len(base)),
        "total_mean": total_mean,
        "spread_abs_mean": spread_abs_mean,
        "mu_beta": [float(x) for x in mu_beta.tolist()],
        "sigma_beta": [float(x) for x in sigma_beta.tolist()],
        "sigma_floor": sigma_floor,
        "sigma_cap": sigma_cap,
    }


def score_odds_games_for_tail(odds_df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    out = odds_df.copy()
    if out.empty:
        return out

    total_mean = float(model.get("total_mean", 150.0))
    spread_abs_mean = float(model.get("spread_abs_mean", 6.0))
    mu_beta = np.array(model.get("mu_beta", [0.0, 0.0, 0.0]), dtype=float)
    sigma_beta = np.array(model.get("sigma_beta", [math.log(10.0), 0.0, 0.0]), dtype=float)
    sigma_floor = float(model.get("sigma_floor", 4.0))
    sigma_cap = float(model.get("sigma_cap", 30.0))

    total = _to_numeric_series(out, "total_points")
    spread_home = _to_numeric_series(out, "spread_home")
    spread_away = _to_numeric_series(out, "spread_away")
    vegas_margin = -spread_home
    vegas_margin = vegas_margin.where(spread_home.notna(), spread_away)
    abs_spread = pd.to_numeric(vegas_margin, errors="coerce").abs().fillna(0.0)

    x_total = (total - total_mean).astype(float)
    x_spread = (abs_spread - spread_abs_mean).astype(float)
    x = np.column_stack([np.ones(len(out)), x_total.to_numpy(), x_spread.to_numpy()])

    mu = x @ mu_beta
    expected_abs_resid = np.exp(x @ sigma_beta)
    sigma = expected_abs_resid * math.sqrt(math.pi / 2.0)
    sigma = np.clip(sigma, sigma_floor, sigma_cap)

    out["tail_residual_mu"] = pd.Series(mu, index=out.index).astype(float)
    out["tail_sigma"] = pd.Series(sigma, index=out.index).astype(float)
    out["p_plus_8"] = out.apply(
        lambda r: _tail_probability(8.0, float(r.get("tail_residual_mu") or 0.0), float(r.get("tail_sigma") or 1.0)),
        axis=1,
    )
    out["p_plus_12"] = out.apply(
        lambda r: _tail_probability(12.0, float(r.get("tail_residual_mu") or 0.0), float(r.get("tail_sigma") or 1.0)),
        axis=1,
    )
    out["volatility_score"] = out["tail_sigma"]
    return out


def map_slate_games_to_tail_features(
    slate_game_keys: list[str],
    odds_tail_df: pd.DataFrame,
) -> pd.DataFrame:
    if not slate_game_keys or odds_tail_df.empty:
        return pd.DataFrame(
            columns=[
                "game_key",
                "game_tail_event_id",
                "game_tail_match_score",
                "game_total_line",
                "game_spread_line",
                "game_tail_residual_mu",
                "game_tail_sigma",
                "game_p_plus_8",
                "game_p_plus_12",
                "game_volatility_score",
            ]
        )

    odds = odds_tail_df.copy()
    for c in ["home_team", "away_team"]:
        odds[c] = odds.get(c, "").astype(str)
    odds["event_id"] = odds.get("event_id", "").astype(str)
    odds["tail_residual_mu"] = pd.to_numeric(odds.get("tail_residual_mu"), errors="coerce")
    odds["tail_sigma"] = pd.to_numeric(odds.get("tail_sigma"), errors="coerce")
    odds["p_plus_8"] = pd.to_numeric(odds.get("p_plus_8"), errors="coerce")
    odds["p_plus_12"] = pd.to_numeric(odds.get("p_plus_12"), errors="coerce")
    odds["volatility_score"] = pd.to_numeric(odds.get("volatility_score"), errors="coerce")
    odds["total_points"] = pd.to_numeric(odds.get("total_points"), errors="coerce")
    odds["spread_home"] = pd.to_numeric(odds.get("spread_home"), errors="coerce")

    used_events: set[str] = set()
    rows: list[dict[str, Any]] = []
    for game_key in sorted({str(x or "").strip().upper() for x in slate_game_keys if str(x or "").strip()}):
        if "@" not in game_key:
            continue
        away_key, home_key = game_key.split("@", 1)
        away_key = away_key.strip()
        home_key = home_key.strip()
        if not away_key or not home_key:
            continue

        best_score = -1.0
        best_row: pd.Series | None = None
        for _, r in odds.iterrows():
            event_id = str(r.get("event_id") or "").strip()
            if event_id and event_id in used_events:
                continue

            same_away = _team_token_similarity(away_key, str(r.get("away_team") or ""))
            same_home = _team_token_similarity(home_key, str(r.get("home_team") or ""))
            same_score = (same_away + same_home) / 2.0

            flip_away = _team_token_similarity(away_key, str(r.get("home_team") or ""))
            flip_home = _team_token_similarity(home_key, str(r.get("away_team") or ""))
            flip_score = (flip_away + flip_home) / 2.0

            score = max(same_score, flip_score)
            min_side = min(same_away, same_home) if same_score >= flip_score else min(flip_away, flip_home)
            if score > best_score and min_side >= 0.30:
                best_score = score
                best_row = r

        if best_row is None:
            continue
        if best_score < 0.57:
            continue

        event_id = str(best_row.get("event_id") or "").strip()
        if event_id:
            used_events.add(event_id)

        rows.append(
            {
                "game_key": game_key,
                "game_tail_event_id": event_id,
                "game_tail_match_score": float(best_score),
                "game_total_line": best_row.get("total_points"),
                "game_spread_line": best_row.get("spread_home"),
                "game_tail_residual_mu": best_row.get("tail_residual_mu"),
                "game_tail_sigma": best_row.get("tail_sigma"),
                "game_p_plus_8": best_row.get("p_plus_8"),
                "game_p_plus_12": best_row.get("p_plus_12"),
                "game_volatility_score": best_row.get("volatility_score"),
            }
        )

    return pd.DataFrame(rows)
