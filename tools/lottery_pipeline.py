"""High-level lottery forecasting pipeline with multi-lottery scraping support."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from scipy.stats import chisquare, mode, norm

import keras_tuner as kt

from deap import algorithms, base, creator, tools
from skopt import gp_minimize
from skopt.space import Integer, Real

# Optional dependencies: requests, bs4, shap.  They are imported lazily so the
# core pipeline can still operate when those modules are unavailable (for
# example inside the numpy CI environment).
try:  # pragma: no cover - optional import
    import requests
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional import
    requests = None
    BeautifulSoup = None

try:  # pragma: no cover - optional import
    import shap
except Exception:  # pragma: no cover - optional import
    shap = None

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

NUMBER_PATTERN = re.compile(r"\d+")


@dataclass
class LotteryScraper:
    """Configuration and callable for scraping a specific lottery."""

    key: str
    name: str
    main_balls: int
    bonus_balls: int
    fetcher: Callable[["LotteryScraper", Optional[int], Optional[int]], pd.DataFrame]
    description: str = ""
    default_start_year: Optional[int] = None

    def fetch(self, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
        """Fetch the lottery results via the configured fetcher.

        Parameters
        ----------
        start_year:
            Optional start year filter for scrapers that expose year-based
            pagination.  When omitted, :attr:`default_start_year` is used.
        end_year:
            Optional end year filter.
        """

        if requests is None or BeautifulSoup is None:
            raise RuntimeError(
                "The scraping helpers require the 'requests' and 'beautifulsoup4'"
                " packages. Install them to enable web scraping."
            )
        effective_start = start_year if start_year is not None else self.default_start_year
        return self.fetcher(self, effective_start, end_year)


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _parse_numbers(value: str) -> List[int]:
    numbers = [int(match) for match in NUMBER_PATTERN.findall(str(value))]
    return numbers


def _normalise_records(records: Iterable[Tuple[datetime, Sequence[int], Sequence[int]]],
                        main_balls: int,
                        bonus_balls: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for draw_date, main_numbers, bonus_numbers in records:
        if len(main_numbers) < main_balls:
            continue
        row: Dict[str, object] = {"draw_date": draw_date}
        for idx, number in enumerate(main_numbers[:main_balls], start=1):
            row[f"ball_{idx}"] = number
        for idx, number in enumerate(bonus_numbers[:bonus_balls], start=1):
            row[f"bonus_{idx}"] = number
        rows.append(row)
    if not rows:
        raise RuntimeError("No draw information parsed from scraper.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["draw_date"]).sort_values("draw_date")
    df.reset_index(drop=True, inplace=True)
    return df


def _scrape_lotteryusa(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int],
                       slug: str) -> pd.DataFrame:
    """Scrape lotteryusa.com yearly archive pages for US games."""

    assert requests is not None
    records: List[Tuple[datetime, Sequence[int], Sequence[int]]] = []
    current_year = datetime.utcnow().year
    first_year = start_year or current_year
    if start_year is None:
        # LotteryUSA typically stores archives back to early years.  To avoid
        # excessive traffic we probe backwards until we encounter an HTTP 404.
        first_year = current_year
    year = first_year
    seen_years: set[int] = set()
    while year >= 1996:  # historical limit for Mega Millions / Powerball
        if end_year is not None and year > end_year:
            year -= 1
            continue
        url = f"https://www.lotteryusa.com/{slug}/year/{year}/"
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            if start_year is not None:
                break
            if year == current_year:
                year -= 1
                continue
            else:
                break
        response.raise_for_status()
        tables = pd.read_html(response.text)
        if not tables:
            if start_year is not None:
                break
            year -= 1
            continue
        table = tables[0]
        table.columns = [str(col).strip().lower() for col in table.columns]
        for _, row in table.iterrows():
            date_text = str(row.get("date") or row.iloc[0])
            try:
                draw_date = pd.to_datetime(date_text).to_pydatetime()
            except Exception:
                continue
            winning_cols = [col for col in table.columns if "winning" in col]
            bonus_keywords = ("powerball", "mega ball", "megaball", "megaball", "megaball", "megaplier")
            bonus_cols = [col for col in table.columns if any(keyword in col for keyword in bonus_keywords)]
            main_numbers: List[int] = []
            for col in winning_cols:
                main_numbers.extend(_parse_numbers(row[col]))
            bonus_numbers: List[int] = []
            for col in bonus_cols:
                bonus_numbers.extend(_parse_numbers(row[col]))
            if not bonus_cols and len(main_numbers) > scraper.main_balls:
                bonus_numbers = main_numbers[scraper.main_balls:scraper.main_balls + scraper.bonus_balls]
                main_numbers = main_numbers[:scraper.main_balls]
            records.append((draw_date, main_numbers, bonus_numbers))
        seen_years.add(year)
        if start_year is None and year > 1996:
            year -= 1
        else:
            break
    return _normalise_records(records, scraper.main_balls, scraper.bonus_balls)


def _scrape_lotterynet(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int],
                       path: str) -> pd.DataFrame:
    """Scrape lottery.net archive pages used by several European lotteries."""

    assert requests is not None
    records: List[Tuple[datetime, Sequence[int], Sequence[int]]] = []
    page = 1
    while page < 500:
        url = f"https://www.lottery.net/{path}?page={page}"
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            break
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        if table is None:
            break
        tbody = table.find("tbody") or table
        rows = tbody.find_all("tr")
        if not rows:
            break
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            date_text = cells[0].get_text(strip=True)
            try:
                draw_date = pd.to_datetime(date_text, dayfirst=True).to_pydatetime()
            except Exception:
                continue
            if start_year and draw_date.year < start_year:
                continue
            if end_year and draw_date.year > end_year:
                continue
            main_numbers: List[int] = []
            bonus_numbers: List[int] = []
            number_cells = cells[1:-1] if len(cells) > 2 else cells[1:]
            for cell in number_cells:
                classes = cell.get("class", [])
                values = _parse_numbers(cell.get_text(" ", strip=True))
                if "bonus" in classes or "star" in classes or "lucky" in classes:
                    bonus_numbers.extend(values)
                else:
                    main_numbers.extend(values)
            if not bonus_numbers and scraper.bonus_balls:
                bonus_numbers = main_numbers[scraper.main_balls:scraper.main_balls + scraper.bonus_balls]
                main_numbers = main_numbers[:scraper.main_balls]
            records.append((draw_date, main_numbers, bonus_numbers))
        page += 1
    return _normalise_records(records, scraper.main_balls, scraper.bonus_balls)


def scrape_euromillions(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    return _scrape_lotterynet(scraper, start_year, end_year, "euromillions/results")


def scrape_eurojackpot(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    return _scrape_lotterynet(scraper, start_year, end_year, "eurojackpot/results")


def scrape_superenalotto(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    return _scrape_lotterynet(scraper, start_year, end_year, "superenalotto/results")


def scrape_superstar_enalotto(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    return _scrape_lotterynet(scraper, start_year, end_year, "superenalotto/superstar-results")


def scrape_powerball(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    return _scrape_lotteryusa(scraper, start_year, end_year, "powerball")


def scrape_mega_millions(scraper: LotteryScraper, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:
    return _scrape_lotteryusa(scraper, start_year, end_year, "mega-millions")


LOTTERY_SCRAPERS: Dict[str, LotteryScraper] = {
    "euromillions": LotteryScraper(
        key="euromillions",
        name="EuroMillions",
        main_balls=5,
        bonus_balls=2,
        fetcher=scrape_euromillions,
        description="EuroMillions main draw with Lucky Stars",
        default_start_year=2004,
    ),
    "eurojackpot": LotteryScraper(
        key="eurojackpot",
        name="Eurojackpot",
        main_balls=5,
        bonus_balls=2,
        fetcher=scrape_eurojackpot,
        description="Eurojackpot draw with Euro numbers",
        default_start_year=2012,
    ),
    "superenalotto": LotteryScraper(
        key="superenalotto",
        name="SuperEnalotto",
        main_balls=6,
        bonus_balls=1,
        fetcher=scrape_superenalotto,
        description="Italian SuperEnalotto including the Jolly number",
        default_start_year=1997,
    ),
    "superstar": LotteryScraper(
        key="superstar",
        name="SuperStar SuperEnalotto",
        main_balls=6,
        bonus_balls=2,
        fetcher=scrape_superstar_enalotto,
        description="SuperEnalotto with SuperStar add-on",
        default_start_year=2007,
    ),
    "megamillions": LotteryScraper(
        key="megamillions",
        name="Mega Millions",
        main_balls=5,
        bonus_balls=1,
        fetcher=scrape_mega_millions,
        description="US Mega Millions including the Mega Ball",
        default_start_year=2002,
    ),
    "powerball": LotteryScraper(
        key="powerball",
        name="US Powerball",
        main_balls=5,
        bonus_balls=1,
        fetcher=scrape_powerball,
        description="US Powerball draw including the Powerball",
        default_start_year=1997,
    ),
}


# ---------------------------------------------------------------------------
# Data preparation and modelling helpers
# ---------------------------------------------------------------------------

def expand_draw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("draw_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    value_cols = [col for col in df.columns if col.startswith("ball_") or col.startswith("bonus_")]
    df["draw_index"] = np.arange(len(df))

    numbers = df[value_cols]
    df["mean"] = numbers.mean(axis=1)
    df["variance"] = numbers.var(axis=1)
    flattened = numbers.to_numpy().ravel()
    freq_map = pd.Series(flattened).value_counts()
    df["frequency"] = numbers.apply(lambda row: sum(freq_map.get(val, 0) for val in row if pd.notna(val)), axis=1)
    fft_values = np.abs(np.fft.fft(numbers.to_numpy(), axis=0))
    df["fft_max"] = np.max(fft_values, axis=1)

    return df


def compute_frequency_baseline(df: pd.DataFrame, value_cols: Sequence[str]) -> np.ndarray:
    values = df[value_cols].to_numpy(dtype=float).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise RuntimeError("Insufficient data to compute frequency baseline.")
    counts = pd.Series(values).value_counts()
    selected: List[int] = []
    for number in counts.index:
        if number in selected:
            continue
        selected.append(int(number))
        if len(selected) >= len(value_cols):
            break
    return np.array(selected[: len(value_cols)], dtype=int)


def _runs_test(values: np.ndarray) -> float:
    if values.size < 10:
        return float("nan")
    median = np.median(values)
    signs = values > median
    if np.all(signs) or np.all(~signs):
        return float("nan")
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n1 = signs.sum()
    n2 = signs.size - n1
    if n1 == 0 or n2 == 0:
        return float("nan")
    mean_runs = 1 + (2 * n1 * n2) / (n1 + n2)
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    if var_runs <= 0:
        return float("nan")
    z_score = (runs - mean_runs) / np.sqrt(var_runs)
    return float(2 * (1 - norm.cdf(abs(z_score))))


def randomness_diagnostics(df: pd.DataFrame, value_cols: Sequence[str]) -> Dict[str, float]:
    numbers = df[value_cols]
    flattened = numbers.to_numpy(dtype=float).ravel()
    flattened = flattened[~np.isnan(flattened)]
    diagnostics: Dict[str, float] = {}
    if flattened.size:
        value_counts = pd.Series(flattened).value_counts().sort_index()
        expected = np.full_like(value_counts.to_numpy(), fill_value=flattened.size / len(value_counts), dtype=float)
        chi_stat, chi_p = chisquare(value_counts.to_numpy(), f_exp=expected)
        diagnostics["chi_square_stat"] = float(chi_stat)
        diagnostics["chi_square_p_value"] = float(chi_p)
    mean_series = numbers.mean(axis=1).to_numpy(dtype=float)
    diagnostics["runs_test_p_value"] = _runs_test(mean_series)
    if mean_series.size > 2:
        mean_series = mean_series - np.mean(mean_series)
        autocorr_num = np.dot(mean_series[:-1], mean_series[1:])
        autocorr_den = np.dot(mean_series, mean_series)
        if autocorr_den != 0:
            lag1_corr = autocorr_num / autocorr_den
            z_score = lag1_corr * np.sqrt((mean_series.size - 2) / (1 - lag1_corr ** 2)) if abs(lag1_corr) < 1 else 0.0
            diagnostics["lag1_autocorrelation"] = float(lag1_corr)
            diagnostics["lag1_autocorrelation_p_value"] = float(2 * (1 - norm.cdf(abs(z_score))))
        else:
            diagnostics["lag1_autocorrelation"] = float("nan")
            diagnostics["lag1_autocorrelation_p_value"] = float("nan")
    return diagnostics


def augment_numeric_data(df: pd.DataFrame, value_cols: Sequence[str], noise: float = 0.01) -> pd.DataFrame:
    augmented_rows: List[pd.Series] = []
    for _, row in df.iterrows():
        noise_vec = np.random.normal(0, noise, size=len(value_cols))
        augmented = np.clip(row[value_cols].to_numpy(dtype=float) + noise_vec, 0, 1)
        augmented_rows.append(pd.Series(augmented, index=value_cols))
    df_aug = pd.DataFrame(augmented_rows)
    return pd.concat([df.reset_index(drop=True), df_aug.reset_index(drop=True)], ignore_index=True)


def sequence_dataset(df: pd.DataFrame, value_cols: Sequence[str], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    values = df[value_cols].to_numpy(dtype=float)
    sequences: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for idx in range(len(values) - seq_len):
        sequences.append(values[idx:idx + seq_len])
        targets.append(values[idx + seq_len])
    return np.stack(sequences), np.stack(targets)


def build_transformer_model(hp: kt.HyperParameters, seq_len: int, feature_count: int) -> keras.Model:
    inputs = keras.Input(shape=(seq_len, feature_count))
    positions = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.float32)
    positions = tf.reshape(positions, (1, seq_len, 1))
    x = inputs + positions

    heads = hp.Int("heads", min_value=2, max_value=8, step=2)
    key_dim = hp.Int("key_dim", min_value=16, max_value=64, step=16)
    x = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)

    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    ff_dim = hp.Int("ff_dim", min_value=32, max_value=256, step=32)
    ff = keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = keras.layers.Dense(feature_count)(ff)
    x = x + ff
    x = keras.layers.GlobalAveragePooling1D()(x)

    dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(feature_count, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("lr", [1e-3, 5e-4, 1e-4])),
        loss="mse",
        metrics=["mae"],
    )
    return model


def tune_transformer(X: np.ndarray, y: np.ndarray, max_trials: int = 10, epochs: int = 60) -> keras.Model:
    tscv = TimeSeriesSplit(n_splits=3)
    best_model: Optional[keras.Model] = None
    best_loss = float("inf")

    for split_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        tuner = kt.BayesianOptimization(
            lambda hp: build_transformer_model(hp, X.shape[1], X.shape[2]),
            objective="val_loss",
            max_trials=max_trials,
            directory="tuner_logs",
            project_name=f"split_{split_idx}",
            overwrite=True,
            seed=SEED,
        )
        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,
            batch_size=16,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        model = tuner.hypermodel.build(tuner.get_best_hyperparameters(1)[0])
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )
        val_loss = history.history["val_loss"][-1]
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
    if best_model is None:
        raise RuntimeError("Transformer tuning failed to produce a model.")
    return best_model


def predict_transformer_uncertainty(model: keras.Model, last_seq: np.ndarray, samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    predictions: List[np.ndarray] = []
    for _ in range(max(samples, 1)):
        preds = model(last_seq, training=True)
        predictions.append(preds.numpy())
    stacked = np.concatenate(predictions, axis=0)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True)
    return mean, std


def tune_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    X_flat = X.reshape(X.shape[0], -1)

    def objective(params: Sequence[int]) -> float:
        n_estimators, max_depth = params
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=SEED,
        )
        scores: List[float] = []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_flat):
            model.fit(X_flat[train_idx], y[train_idx])
            pred = model.predict(X_flat[val_idx])
            scores.append(np.mean((pred - y[val_idx]) ** 2))
        return float(np.mean(scores))

    result = gp_minimize(
        objective,
        [Integer(200, 600), Integer(4, 16)],
        n_calls=20,
        random_state=SEED,
    )
    model = RandomForestRegressor(
        n_estimators=int(result.x[0]),
        max_depth=int(result.x[1]),
        random_state=SEED,
    )
    model.fit(X_flat, y)
    return model


def tune_xgboost(X: np.ndarray, y: np.ndarray) -> MultiOutputRegressor:
    X_flat = X.reshape(X.shape[0], -1)

    def objective(params: Sequence[float]) -> float:
        n_estimators, learning_rate, max_depth = params
        base_model = XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
        )
        model = MultiOutputRegressor(base_model)
        scores: List[float] = []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_flat):
            model.fit(X_flat[train_idx], y[train_idx])
            pred = model.predict(X_flat[val_idx])
            scores.append(np.mean((pred - y[val_idx]) ** 2))
        return float(np.mean(scores))

    result = gp_minimize(
        objective,
        [Integer(200, 600), Real(0.01, 0.3), Integer(4, 12)],
        n_calls=20,
        random_state=SEED,
    )
    base_model = XGBRegressor(
        n_estimators=int(result.x[0]),
        learning_rate=float(result.x[1]),
        max_depth=int(result.x[2]),
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=1,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_flat, y)
    return model


def stack_models(transformer: keras.Model, rf: RandomForestRegressor, xgb: MultiOutputRegressor,
                 X: np.ndarray, y: np.ndarray) -> Ridge:
    X_flat = X.reshape(X.shape[0], -1)
    transformer_preds = transformer.predict(X, verbose=0)
    rf_preds = rf.predict(X_flat)
    xgb_preds = xgb.predict(X_flat)
    stacked = np.hstack([transformer_preds, rf_preds, xgb_preds])
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stacked, y)
    return meta_model


def run_genetic_algorithm(target_vector: np.ndarray) -> np.ndarray:
    n_features = target_vector.shape[0]
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual: Sequence[float]) -> Tuple[float]:
        distance = np.mean(np.abs(target_vector - np.array(individual)))
        return (1.0 / (1.0 + distance),)

    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=80)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    best = tools.selBest(population, k=1)[0]
    return np.array(best)


def explain_with_shap(model: keras.Model, data: np.ndarray) -> None:
    if shap is None:
        raise RuntimeError(
            "SHAP is not installed. Install the 'shap' package to enable model explanations."
        )
    background = shap.kmeans(data.reshape(data.shape[0], -1), 10)
    explainer = shap.KernelExplainer(lambda x: model.predict(x.reshape((-1, data.shape[1], data.shape[2])), verbose=0),
                                     background)
    shap_values = explainer.shap_values(data[:50].reshape(50, -1), nsamples=100)
    shap.summary_plot(shap_values, data[:50].reshape(50, -1))


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(lottery_key: str, sequence_length: int = 5,
                 start_year: Optional[int] = None,
                 end_year: Optional[int] = None,
                 use_shap: bool = False,
                 augment: bool = True,
                 diagnostics: bool = False,
                 mc_samples: int = 50) -> Dict[str, object]:
    if lottery_key not in LOTTERY_SCRAPERS:
        raise KeyError(f"Unknown lottery '{lottery_key}'. Available: {', '.join(sorted(LOTTERY_SCRAPERS))}")
    scraper = LOTTERY_SCRAPERS[lottery_key]
    raw_df = scraper.fetch(start_year=start_year, end_year=end_year)

    value_cols = [col for col in raw_df.columns if col.startswith("ball_") or col.startswith("bonus_")]
    baseline = compute_frequency_baseline(raw_df, value_cols)
    diagnostics_payload = randomness_diagnostics(raw_df, value_cols) if diagnostics else None

    df = expand_draw_dataframe(raw_df)

    scaler = MinMaxScaler()
    df[value_cols] = scaler.fit_transform(df[value_cols])

    if augment:
        df = augment_numeric_data(df, value_cols)

    X, y = sequence_dataset(df, value_cols, sequence_length)

    transformer = tune_transformer(X, y)
    rf = tune_random_forest(X, y)
    xgb = tune_xgboost(X, y)
    meta_model = stack_models(transformer, rf, xgb, X, y)

    last_seq = X[-1:].copy()
    transformer_pred = transformer.predict(last_seq, verbose=0)
    transformer_mc_mean, transformer_mc_std = predict_transformer_uncertainty(transformer, last_seq, samples=mc_samples)
    rf_pred = rf.predict(last_seq.reshape(1, -1))
    xgb_pred = xgb.predict(last_seq.reshape(1, -1))
    stacked = np.hstack([transformer_pred, rf_pred, xgb_pred])
    meta_pred = meta_model.predict(stacked)
    ga_pred = run_genetic_algorithm(meta_pred.reshape(-1))

    transformer_inv = scaler.inverse_transform(transformer_pred)[0]
    transformer_mc_mean_inv = scaler.inverse_transform(transformer_mc_mean)[0]
    transformer_mc_std_inv = transformer_mc_std[0] * (scaler.data_max_ - scaler.data_min_)
    rf_inv = scaler.inverse_transform(rf_pred.reshape(1, -1))[0]
    xgb_inv = scaler.inverse_transform(xgb_pred.reshape(1, -1))[0]
    meta_inv = scaler.inverse_transform(meta_pred)[0]
    ga_inv = scaler.inverse_transform(ga_pred.reshape(1, -1))[0]

    final_prediction = mode(
        [np.round(meta_inv), np.round(ga_inv)], axis=0, keepdims=False
    ).mode

    if use_shap:
        explain_with_shap(transformer, X)

    result: Dict[str, object] = {
        "transformer": np.round(transformer_inv).astype(int),
        "transformer_uncertainty_mean": transformer_mc_mean_inv.astype(float),
        "transformer_uncertainty_std": transformer_mc_std_inv.astype(float),
        "random_forest": np.round(rf_inv).astype(int),
        "xgboost": np.round(xgb_inv).astype(int),
        "stacked": np.round(meta_inv).astype(int),
        "genetic": np.round(ga_inv).astype(int),
        "frequency_baseline": baseline.astype(int),
        "final": final_prediction.astype(int),
    }
    if diagnostics_payload is not None:
        result["randomness_diagnostics"] = diagnostics_payload
    return result


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lottery forecasting pipeline with scraping support")
    parser.add_argument("lottery", choices=sorted(LOTTERY_SCRAPERS), help="Lottery identifier to scrape")
    parser.add_argument("--sequence-length", type=int, default=5, help="Sequence length for the transformer model")
    parser.add_argument("--start-year", type=int, default=None, help="Optional start year for scraping")
    parser.add_argument("--end-year", type=int, default=None, help="Optional end year for scraping")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--shap", action="store_true", help="Compute SHAP explanations (requires shap)")
    parser.add_argument("--diagnostics", action="store_true", help="Include randomness diagnostics in the output")
    parser.add_argument("--mc-samples", type=int, default=50, help="Number of Monte Carlo samples for transformer uncertainty")
    parser.add_argument("--output", type=str, default=None, help="Optional path to store predictions as JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    results = run_pipeline(
        lottery_key=args.lottery,
        sequence_length=args.sequence_length,
        start_year=args.start_year,
        end_year=args.end_year,
        use_shap=args.shap,
        augment=not args.no_augment,
        diagnostics=args.diagnostics,
        mc_samples=max(args.mc_samples, 1),
    )

    def _to_serialisable(value: object) -> object:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: _to_serialisable(v) for k, v in value.items()}
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    serialisable = {key: _to_serialisable(value) for key, value in results.items()}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=2)
    print(json.dumps(serialisable, indent=2))


if __name__ == "__main__":
    main()
