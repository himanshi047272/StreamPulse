"""Data loading, cleaning, feature engineering and a synthetic Netflix-schema generator."""
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_bool_dtype

TARGET_CANDIDATES = ["churned", "churn", "is_churned", "label"]
ID_CANDIDATES = ["customer_id", "user_id", "id", "customerid"]
MAX_CAT_LEVELS = 25


def standardize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def find_target(df):
    return next((c for c in TARGET_CANDIDATES if c in df.columns), None)


def find_id(df):
    return next((c for c in ID_CANDIDATES if c in df.columns), None)


def find_date(df):
    for c in df.columns:
        if re.search(r"(^|_)(date|datetime|timestamp)($|_)|_at$", c):
            if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.9:
                return c
    return None


def parse_target(s):
    if is_numeric_dtype(s) or is_bool_dtype(s):
        return s.astype(float)
    m = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0, "churned": 1, "retained": 0}
    return s.astype(str).str.strip().str.lower().map(m).astype(float)


def clean(df):
    """Standardise column names, drop duplicates / unlabeled rows. File order is preserved."""
    df = standardize_columns(df)
    info = {"rows_in": int(len(df))}
    df = df.drop_duplicates().reset_index(drop=True)
    info["duplicates_removed"] = info["rows_in"] - len(df)
    t = find_target(df)
    info["target"] = t
    info["unlabelled_removed"] = 0
    if t is not None:
        df[t] = parse_target(df[t])
        before = len(df)
        df = df.dropna(subset=[t]).reset_index(drop=True)
        df[t] = df[t].astype(int)
        info["unlabelled_removed"] = before - len(df)
    info["rows_out"] = int(len(df))
    return df, info


def chronological(df, id_col):
    """Order rows by the number inside customer_id (e.g. CUST00123): an acquisition-order proxy."""
    if id_col is None or id_col not in df.columns:
        return df
    nums = df[id_col].astype(str).str.extract(r"(\d+)\s*$")[0]
    if nums.notna().all():
        return df.assign(_k=nums.astype(int)).sort_values("_k", kind="stable").drop(columns="_k").reset_index(drop=True)
    return df


class FeatureBuilder:
    """Fitted on the TRAIN split only, then applied identically to validation / test (no leakage)."""

    def fit(self, df, target):
        self.target = target
        self.id_col = find_id(df)
        self.date_col = find_date(df)
        skip = {target, self.id_col, self.date_col}
        self.num_cols, self.cat_cols = [], []
        for c in df.columns:
            if c in skip:
                continue
            if is_datetime64_any_dtype(df[c]):
                continue
            if is_numeric_dtype(df[c]) or is_bool_dtype(df[c]):
                self.num_cols.append(c)
            elif 2 <= df[c].nunique() <= MAX_CAT_LEVELS:
                self.cat_cols.append(c)
        num = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
        self.medians = num.median().to_dict()
        self.modes = {c: df[c].mode().iloc[0] for c in self.cat_cols}
        self.levels = {c: sorted(df[c].dropna().astype(str).unique())[1:] for c in self.cat_cols}  # drop-first
        self.q75_login = float(num["last_login_days"].quantile(0.75)) if "last_login_days" in num else None
        self.q25_watch = float(num["watch_hours"].quantile(0.25)) if "watch_hours" in num else None
        self.all_features = list(self._build(df).columns)
        self.selected = list(self.all_features)
        self.corr = {}
        return self

    def _build(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.num_cols:
            col = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
            out[c] = col.fillna(self.medians[c]).astype(float)
        if {"watch_hours", "number_of_profiles"} <= set(out):
            out["engagement_per_profile"] = out["watch_hours"] / out["number_of_profiles"].clip(lower=1)
        if {"monthly_fee", "watch_hours"} <= set(out):
            out["fee_per_watch_hour"] = out["monthly_fee"] / (out["watch_hours"] + 1)
        if self.q75_login is not None:
            out["inactive_flag"] = (out["last_login_days"] >= self.q75_login).astype(float)
        if self.q25_watch is not None:
            out["low_engagement_flag"] = (out["watch_hours"] <= self.q25_watch).astype(float)
        for c in self.cat_cols:
            base = df[c] if c in df else pd.Series(self.modes[c], index=df.index)
            vals = base.fillna(self.modes[c]).astype(str)
            for lv in self.levels[c]:
                out[f"{c}={lv}"] = (vals == lv).astype(float)
        return out

    def select(self, X, y, thr=0.01, min_keep=8):
        """Correlation-based feature selection on the TRAIN split (drops near-zero |corr| with churn)."""
        y = np.asarray(y, float)
        corr = {c: (abs(np.corrcoef(X[c].values, y)[0, 1]) if X[c].std() > 0 else 0.0) for c in X.columns}
        corr = {c: (0.0 if np.isnan(v) else float(v)) for c, v in corr.items()}
        keep = [c for c, v in corr.items() if v >= thr]
        if len(keep) < min_keep:
            keep = sorted(corr, key=corr.get, reverse=True)[:min_keep]
        self.selected = [c for c in X.columns if c in keep]
        self.corr = corr
        return self.selected, [c for c in X.columns if c not in self.selected]

    def transform(self, df, selected=True):
        X = self._build(df)
        return X[self.selected] if selected else X


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset using the public "Netflix Customer Churn" column schema.
# Only for demo/testing when you don't have the real CSV to hand.
# ──────────────────────────────────────────────────────────────────────────────
def make_sample(n=5000, seed=7, start_id=1):
    rng = np.random.default_rng(seed)
    ids = [f"CUST{i:05d}" for i in range(start_id, start_id + n)]
    cohort = np.arange(n) / max(n / 60, 1)  # ~60 acquisition cohorts
    sub = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.34, 0.33, 0.33])
    fee = pd.Series(sub).map({"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}).values
    watch = np.clip(rng.gamma(2.0, 9.0, n) * (1 + 0.25 * (sub == "Premium")), 0.5, 200).round(1)
    last_login = rng.integers(0, 60, n)
    profiles = rng.integers(1, 6, n)
    pay = rng.choice(["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"], n, p=[.3, .25, .25, .12, .08])
    logit = (-0.9 + 0.045 * last_login - 0.035 * watch + 0.55 * (sub == "Basic") - 0.12 * profiles
             + 0.45 * (pay == "Gift Card") + 0.012 * cohort + 0.45 * np.sin(2 * np.pi * cohort / 12)
             + rng.normal(0, 0.6, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "customer_id": ids,
        "age": rng.integers(18, 75, n),
        "gender": rng.choice(["Male", "Female", "Other"], n, p=[.48, .48, .04]),
        "subscription_type": sub,
        "watch_hours": watch,
        "last_login_days": last_login,
        "region": rng.choice(["Africa", "Asia", "Europe", "North America", "Oceania", "South America"], n),
        "device": rng.choice(["Mobile", "TV", "Laptop", "Tablet", "Desktop"], n),
        "monthly_fee": fee,
        "churned": churn,
        "payment_method": pay,
        "number_of_profiles": profiles,
        "avg_watch_time_per_day": (watch / rng.uniform(20, 40, n)).round(2),
        "favorite_genre": rng.choice(["Action", "Comedy", "Drama", "Documentary", "Horror", "Romance", "Sci-Fi"], n),
    })


if __name__ == "__main__":
    import os
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")
    os.makedirs(d, exist_ok=True)
    make_sample(5000, 7, 1).to_csv(os.path.join(d, "netflix_train_sample.csv"), index=False)
    make_sample(1500, 11, 5001).to_csv(os.path.join(d, "netflix_test_sample.csv"), index=False)
    print("Sample CSVs written to", d)
