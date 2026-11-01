"""Customer segments: RFM -> log -> scale -> KMeans -> named segments.

RFM = Recency (days since last order), Frequency (number of orders), Monetary (total spend).

Why log-transform? Spend and order counts are very skewed: a few wholesale customers spend
£400,000 while the median customer spends £855. KMeans uses straight-line distance, so without the
log the big spenders would dominate and everyone else would land in one cluster.

Why scale? After the log the three columns still have different ranges; StandardScaler gives each
mean 0 and standard deviation 1 so none of them counts more than the others.

Two questions, two fits (the second is where leakage hides):

1. DESCRIBE today's customers. Fit on RFM as of the end of the data. Nothing lies in the future of
   "today", so nothing can leak. This feeds the dashboard and the `customers` table.
2. CHECK that the segments mean something for the FUTURE. For each cutoff date, the WHOLE recipe
   (scaler, number of segments k, KMeans, names) is refitted using only RFM as of that date, then we
   measure who really ordered in the next 90 days. Reusing the end-of-data fit here would let the
   future shape the segment boundaries and flatter the result.

Run:  python -m rci.segments   (needs sales.parquet; writes segments.parquet + segments.json)
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from rci import config, db, features

RFM_COLS = ["recency_days", "frequency", "monetary"]

ACTIONS = {
    "Champions": "Reward and retain: loyalty perks, early access, ask for reviews. Do not discount; they already buy.",
    "Promising": "Nurture: a second-purchase offer and product suggestions to turn them into regulars.",
    "At risk": "Win back: a personalised reminder with an incentive. Test it with an A/B experiment first.",
    "Hibernating": "Low-cost reactivation only (one email). Do not spend much on them.",
}


@dataclass
class SegmentModel:
    scaler: StandardScaler
    kmeans: KMeans
    names: dict[int, str]  # KMeans cluster id -> segment name


def log_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df[RFM_COLS].to_numpy(dtype=float))


def select_k(Z: np.ndarray, k_range=range(2, 9), allowed=(3, 6), seed: int = config.SEED):
    """Silhouette score for each k; the best k within `allowed` is chosen.

    Silhouette (-1..1) says how well each point fits its own cluster versus the next nearest one.
    k=2 usually scores highest on RFM data but only separates 'active' from 'lapsed', which is not
    actionable for marketing, so we restrict the choice to 3-6 segments.
    """
    scores = {}
    for k in k_range:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        scores[k] = float(silhouette_score(Z, labels))
    candidates = {k: s for k, s in scores.items() if allowed[0] <= k <= allowed[1]}
    return max(candidates, key=candidates.get), scores


def name_clusters(centroids_z: np.ndarray) -> dict[int, str]:
    """Name each cluster from where its centre sits in the scaled R/F/M space.

    recent  = recency below average (z < 0, lower recency is better)
    engaged = frequency and spend above average (mean of their z-scores > 0)
    Four combinations -> four names. If two clusters land on the same name (k > 4) the more engaged
    one keeps it and the others get " 2", " 3" so names stay unique.
    """
    r, f, m = centroids_z[:, 0], centroids_z[:, 1], centroids_z[:, 2]
    engagement = (f + m) / 2
    base = {}
    for i in range(len(centroids_z)):
        recent, engaged = r[i] < 0, engagement[i] > 0
        base[i] = ("Champions" if engaged else "Promising") if recent else ("At risk" if engaged else "Hibernating")
    names: dict[int, str] = {}
    for name in set(base.values()):
        members = sorted((i for i in base if base[i] == name), key=lambda i: -engagement[i])
        for rank, i in enumerate(members):
            names[i] = name if rank == 0 else f"{name} {rank + 1}"
    return names


def fit(rfm: pd.DataFrame, seed: int = config.SEED) -> tuple[SegmentModel, int, dict[int, float]]:
    scaler = StandardScaler()
    Z = scaler.fit_transform(log_matrix(rfm))
    k, scores = select_k(Z, seed=seed)
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    return SegmentModel(scaler, km, name_clusters(km.cluster_centers_)), k, scores


def assign(model: SegmentModel, rfm: pd.DataFrame) -> pd.Series:
    """Segment name for each row, using an already-fitted model (e.g. RFM at an earlier date)."""
    Z = model.scaler.transform(log_matrix(rfm))
    ids = model.kmeans.predict(Z)
    return pd.Series([model.names[i] for i in ids], index=rfm.index, name="segment")


def summarize(rfm: pd.DataFrame, segment: pd.Series) -> list[dict]:
    df = rfm.assign(segment=segment.to_numpy())
    total_rev = df["monetary"].sum()
    rows = []
    for name, g in df.groupby("segment"):
        rows.append({
            "segment": name,
            "customers": int(len(g)),
            "customer_share": round(len(g) / len(df), 4),
            "revenue_share": round(float(g["monetary"].sum() / total_rev), 4),
            "avg_recency_days": round(float(g["recency_days"].mean()), 1),
            "avg_orders": round(float(g["frequency"].mean()), 1),
            "avg_spend": round(float(g["monetary"].mean()), 0),
            "median_spend": round(float(g["monetary"].median()), 0),
            "action": ACTIONS.get(name.rsplit(" ", 1)[0] if name[-1].isdigit() else name, ""),
        })
    return sorted(rows, key=lambda r: -r["avg_spend"])


def fit_at_cutoff(con, cutoff: dt.date | str, seed: int = config.SEED):
    """Fit the whole segment recipe on RFM as of `cutoff`, using only orders BEFORE that date.

    Returns (model, k, rfm, silhouette_by_k). Because every input comes from rfm.sql with
    `$as_of = cutoff`, nothing on or after the cutoff can influence the scaler, k or the centroids.
    """
    cutoff = pd.Timestamp(cutoff).date()
    rfm = db.query(con, "rfm", as_of=cutoff)
    model, k, scores = fit(rfm, seed)
    return model, k, rfm, scores


def validate_at_cutoffs(con, cutoffs: list[tuple[str, str]], horizon: int = config.HORIZON_DAYS) -> list[dict]:
    """Leak-free check that segments predict behaviour, one entry per (split, cutoff).

    For each cutoff: refit the segments on data before it (`fit_at_cutoff`), assign every customer,
    then measure the share who ordered again in the following `horizon` days. A cutoff whose label
    window is not fully observed is refused outright, not silently scored.
    """
    last_day = db.data_end(con)
    results = []
    for split, c in cutoffs:
        cutoff = pd.Timestamp(c).date()
        if not features.label_is_observable(last_day, cutoff, horizon):
            raise ValueError(
                f"cutoff {c}: its {horizon}-day label window ends {cutoff + dt.timedelta(days=horizon)}, "
                f"but the data ends {last_day}."
            )
        model, k, rfm, scores = fit_at_cutoff(con, cutoff)
        labels = db.query(con, "features", cutoff=cutoff, horizon=horizon)[["customer_id", "label"]]
        df = rfm.merge(labels, on="customer_id", how="inner", validate="one_to_one")
        if len(df) != len(rfm):
            raise ValueError(f"cutoff {c}: RFM and label tables disagree on which customers exist")
        df["segment"] = assign(model, df).to_numpy()
        by_segment = (
            df.groupby("segment")["label"].agg(customers="size", repurchase_rate="mean").reset_index()
            .sort_values("repurchase_rate", ascending=False)
        )
        results.append({
            "split": split,
            "cutoff": str(cutoff),
            "k": int(k),
            "silhouette": round(scores[k], 4),
            "customers": int(len(df)),
            "overall_rate": round(float(df["label"].mean()), 4),
            "segments": [
                {"segment": r["segment"], "customers": int(r["customers"]), "repurchase_rate": round(float(r["repurchase_rate"]), 4)}
                for _, r in by_segment.iterrows()
            ],
        })
    return results


def main() -> None:
    con = db.connect()
    as_of = db.data_end(con) + dt.timedelta(days=1)   # "today" = the day after the last order
    rfm = db.query(con, "rfm", as_of=as_of)
    model, k, scores = fit(rfm)
    segment = assign(model, rfm)

    Z = model.scaler.transform(log_matrix(rfm))
    pca = PCA(2, random_state=config.SEED).fit(Z)
    xy = pca.transform(Z)
    out = rfm[["customer_id", *RFM_COLS]].assign(segment=segment.to_numpy(), pc1=xy[:, 0].round(3), pc2=xy[:, 1].round(3))
    out.to_parquet(config.PROCESSED_DIR / "segments.parquet", index=False, compression="zstd")

    summary = summarize(rfm, segment)

    # Leak-free validation: at EVERY cutoff, refit on data before it, then measure the next 90 days.
    all_cutoffs = [(split, c) for split, cs in config.SPLITS.items() for c in cs]
    by_cutoff = validate_at_cutoffs(con, all_cutoffs)
    primary_cutoff = config.SPLITS["test"][0]          # the latest period; used for the A/B baseline
    primary = next(v for v in by_cutoff if v["cutoff"] == primary_cutoff)
    by_name = {r["segment"]: r for r in summary}
    for s in primary["segments"]:
        if s["segment"] in by_name:                    # a name missing here simply has no leak-free estimate
            by_name[s["segment"]]["repurchase_rate_next_90d"] = s["repurchase_rate"]
            by_name[s["segment"]]["validation_customers"] = s["customers"]

    report = {
        "as_of": str(as_of),
        "chosen_k": int(k),
        "silhouette_by_k": {str(kk): round(s, 4) for kk, s in scores.items()},
        "pca_explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "validation_cutoff": primary_cutoff,
        "segments": summary,
        "validation": {
            "method": (
                "At each cutoff the whole recipe (scaler, k, KMeans, names) is refitted using only RFM as of that date, "
                "then the share of customers who ordered in the next 90 days is measured per segment."
            ),
            "horizon_days": config.HORIZON_DAYS,
            "primary_cutoff": primary_cutoff,
            "by_cutoff": by_cutoff,
        },
        "notes": (
            "Silhouette is highest at k=2 (active vs lapsed), but k is chosen from 3-6 because two "
            "segments are not actionable. The segment table describes customers as of the end of the data. "
            "The 'bought again' rates come from segments refitted at the validation cutoff using only earlier data, "
            "so the future cannot shape the segment boundaries."
        ),
    }
    (config.PROCESSED_DIR / "segments.json").write_text(json.dumps(report, indent=2))

    print(f"k={k}  silhouette by k: " + ", ".join(f"{kk}:{s:.3f}" for kk, s in scores.items()))
    print(pd.DataFrame(summary).drop(columns=["action"]).to_string(index=False))
    print("\nLeak-free validation (segments refitted at each cutoff from earlier data only):")
    for v in by_cutoff:
        rates = ", ".join(f"{s['segment']} {s['repurchase_rate']:.0%} (n={s['customers']})" for s in v["segments"])
        print(f"  {v['split']:<5} {v['cutoff']}  k={v['k']}  overall {v['overall_rate']:.0%} | {rates}")


if __name__ == "__main__":
    main()
