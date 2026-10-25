"""Model zoo (6 non-tree models), class balancing, training and validation."""
import time
from collections import OrderedDict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import RANDOM_STATE
from .metrics import classification_metrics, best_threshold, roc_points

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:  # falls back to a small built-in SMOTE
    SMOTE_AVAILABLE = False


def model_zoo():
    """name -> dict(estimator, family, desc, max_rows). No Random Forest / tree ensembles."""
    return OrderedDict([
        ("Logistic Regression", dict(
            est=LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
            family="Linear", desc="Linear baseline: log-odds of churn as a weighted sum of features.", max_rows=None)),
        ("Linear Discriminant", dict(
            est=LinearDiscriminantAnalysis(),
            family="Linear", desc="Finds the linear boundary that best separates churners from stayers (Gaussian classes).", max_rows=None)),
        ("Naive Bayes", dict(
            est=GaussianNB(),
            family="Probabilistic", desc="Bayes rule with independent Gaussian features. Fast and surprisingly strong.", max_rows=None)),
        ("K-Nearest Neighbors", dict(
            est=KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
            family="Instance-based", desc="Votes among the 25 most similar customers.", max_rows=None)),
        ("SVM (RBF)", dict(
            est=SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            family="Kernel", desc="Non-linear margin classifier using an RBF kernel (fit on up to 6,000 rows for speed).", max_rows=6000)),
        ("Neural Network (MLP)", dict(
            est=MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300, early_stopping=True,
                              random_state=RANDOM_STATE),
            family="Neural network", desc="Two hidden layers (64, 32) with ReLU and early stopping.", max_rows=None)),
    ])


def balance(X, y, tracker=None):
    """SMOTE oversampling of the minority class (applied to TRAIN only). Returns X, y, applied(bool)."""
    y = np.asarray(y).astype(int)
    pos, neg = int(y.sum()), int((y == 0).sum())
    if min(pos, neg) / max(len(y), 1) >= 0.40:
        return X, y, False  # already balanced enough
    minority = 1 if pos < neg else 0
    if SMOTE_AVAILABLE:
        Xr, yr = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X, y)
        return Xr, yr, True
    rng = np.random.default_rng(RANDOM_STATE)
    Xm = X[y == minority]
    k = min(6, len(Xm))
    nn = NearestNeighbors(n_neighbors=k).fit(Xm)
    neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]
    n_new = abs(neg - pos)
    base = rng.integers(0, len(Xm), n_new)
    pick = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
    lam = rng.random((n_new, 1))
    synth = Xm[base] + lam * (Xm[pick] - Xm[base])
    Xr = np.vstack([X, synth])
    yr = np.concatenate([y, np.full(n_new, minority)])
    p = rng.permutation(len(Xr))
    return Xr[p], yr[p], True


def train_and_validate(Xtr, ytr, Xva, yva, tracker):
    """Fit each model on the train split, score on the validation split, tune the decision threshold."""
    zoo = model_zoo()
    fitted, rows = {}, {}
    rng = np.random.default_rng(RANDOM_STATE)
    for name, spec in zoo.items():
        tracker.model(name, "training", 10)
        t0 = time.time()
        X_fit, y_fit = Xtr, ytr
        if spec["max_rows"] and len(Xtr) > spec["max_rows"]:
            idx = rng.choice(len(Xtr), spec["max_rows"], replace=False)
            X_fit, y_fit = Xtr[idx], ytr[idx]
        est = spec["est"]
        est.fit(X_fit, y_fit)
        secs = time.time() - t0
        tracker.model(name, "validating", 70)
        proba = est.predict_proba(Xva)[:, 1]
        thr = best_threshold(yva, proba)
        val = classification_metrics(yva, proba, thr)
        fitted[name] = est
        rows[name] = dict(name=name, family=spec["family"], desc=spec["desc"], train_seconds=round(secs, 2),
                          threshold=thr, val=val, val_roc=roc_points(yva, proba), val_proba=proba)
        tracker.model(name, "done", 100)
        tracker.log(f"{name}: validation F1={val['f1']:.4f}  AUC={val['roc_auc']:.4f}  thr={thr}  ({secs:.1f}s)")
    return fitted, rows
