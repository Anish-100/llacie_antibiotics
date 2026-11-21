"""Evaluation of antibiotics labels (the antibiotic_labels table): one source's drug courses
(a strategy, or a second human) compared against a human annotator's.

Drug names are compared by their canonical name from drugs.db (`drug_name`), which is the
source of truth for drug identity. A name that is not in drugs.db falls back to its normalized
raw name, so it can still match the same unlisted drug on the other side."""
import warnings
import numpy as np
import pandas as pd
import confidenceinterval as ci

from textwrap import dedent

from .evaluate import ConfusionMatrix, f1_score_binary

COURSE_FIELDS = ["start_date", "end_date", "duration_days", "is_current", "relevance"]


def drug_key(df):
    """The name used to match drugs: canonical drug_name, else the normalized raw name."""
    # astype: an empty or all-NULL column comes back as floats, which have no .str
    raw = (df["drug_name_raw"].astype("string").fillna("").str.strip().str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True))
    return df["drug_name"].where(df["drug_name"].notna(), raw)


def match_courses(df_true, df_pred):
    """Pairs up drug courses within each episode by drug. Repeat courses of one drug are paired
    in order of start date (unknown dates last), then end date.

    Both frames hold antibiotic_labels rows without the "none found" rows. Returns one row
    per course with `FK_episode_id`, `drug_key`, `found` ("both", "true_only" or "pred_only")
    and each COURSE_FIELDS column suffixed `_true` and `_pred`."""
    def ranked(df):
        df = df.copy()
        df["drug_key"] = drug_key(df)
        df = df.sort_values(["FK_episode_id", "drug_key", "start_date", "end_date", "id"],
            na_position="last")
        df["course"] = df.groupby(["FK_episode_id", "drug_key"]).cumcount()
        return df[["FK_episode_id", "drug_key", "course"] +
            [c for c in COURSE_FIELDS if c in df.columns]]

    merged = pd.merge(ranked(df_true), ranked(df_pred), how="outer",
        on=["FK_episode_id", "drug_key", "course"], suffixes=("_true", "_pred"),
        indicator="found")
    merged["found"] = merged["found"].map(
        {"both": "both", "left_only": "true_only", "right_only": "pred_only"})
    return merged.sort_values(["FK_episode_id", "drug_key", "course"]).reset_index(drop=True)


class EpisodeConfusionMatrix(ConfusionMatrix):
    """Episode-level confusion matrix (e.g., any antibiotic found), without the value-based
    metrics that only apply to presenting symptoms."""
    METRIC_METHODS = {k: v for k, v in ConfusionMatrix.METRIC_METHODS.items()
        if k not in ("Spearman-R", "Wilcox-Signed-Rank")}

    def __init__(self, vec_truth, vec_pred, **kwargs):
        super().__init__(np.asarray(vec_truth, bool), np.asarray(vec_pred, bool), None, None,
            **kwargs)

    def _format_metric(self, metric_name, metric_value):
        # The base class picks digits from log10(1 - value), which fails for a perfect 1.0
        if isinstance(metric_value, tuple):
            return f"{metric_name:23.23} {_fmt(metric_value)}"
        return f"{metric_name:23.23} {metric_value:10.3f}"


def _fmt(value):
    if isinstance(value, tuple):
        # Some CI methods (e.g. Takahashi's for F1) can step outside [0, 1]; clip for display
        low, high = (float(np.clip(b, 0, 1)) for b in value[1])
        return f"{value[0]:6.3f} ({low:6.3f}, {high:6.3f})"
    return f"{value:6.3f}"


def _f1(y_true, y_pred, tp, fn, fp):
    """F1 with a confidence interval; the CI method can't handle a table with no errors
    (e.g., every course found), so fall back to the plain value there."""
    if fn == 0 or fp == 0:
        return 2 * tp / (2 * tp + fn + fp)
    return f1_score_binary(y_true, y_pred)


def detection_report(courses, label):
    """Precision/recall/F1 for drug courses found, treating unmatched human courses as
    missed and unmatched predicted courses as extra. There are no true negatives."""
    found = courses["found"]
    n_both, n_missed, n_extra = (found == "both").sum(), (found == "true_only").sum(), \
        (found == "pred_only").sum()
    lines = [f"{label}: {n_both + n_missed} labeled, {n_both + n_extra} predicted, "
        f"{n_both} found, {n_missed} missed, {n_extra} extra"]
    if n_both + n_missed == 0 or n_both + n_extra == 0:
        return "\n".join(lines + ["  (not enough courses for precision/recall)"])
    y_true = found.isin(["both", "true_only"]).to_numpy()
    y_pred = found.isin(["both", "pred_only"]).to_numpy()
    lines += [
        f"  {'Precision aka PPV':23} {_fmt(ci.ppv_score(y_true, y_pred))}",
        f"  {'Recall aka sensitivity':23} {_fmt(ci.tpr_score(y_true, y_pred))}",
        f"  {'F1 score':23} {_fmt(_f1(y_true, y_pred, n_both, n_missed, n_extra))}",
    ]
    return "\n".join(lines)


def _as_days(series):
    return pd.to_datetime(series).dt.normalize()


def field_report(matched):
    """For courses found by both sides, how often each field agrees."""
    header = (f"{'Field':15} {'both given':>10} {'agree':>14} {'MAE':>8} "
        f"{'only human':>10} {'only pred':>9} {'neither':>7}")
    lines = [header]
    for field in ["start_date", "end_date", "duration_days", "is_current"]:
        t, p = matched[f"{field}_true"], matched[f"{field}_pred"]
        both = t.notna() & p.notna()
        n_both = int(both.sum())
        mae = ""
        if field in ("start_date", "end_date"):
            diff = (_as_days(t[both]) - _as_days(p[both])).dt.days.abs()
        elif field == "duration_days":
            diff = (t[both].astype(float) - p[both].astype(float)).abs()
        else:
            diff = (t[both].astype(bool) != p[both].astype(bool)).astype(int)
        agree = int((diff == 0).sum())
        if field != "is_current" and n_both:
            mae = f"{diff.mean():8.1f}"
        agree_str = f"{agree} ({agree / n_both:4.0%})" if n_both else "-"
        lines.append(f"{field:15} {n_both:>10} {agree_str:>14} {mae:>8} "
            f"{int((t.notna() & p.isna()).sum()):>10} {int((t.isna() & p.notna()).sum()):>9} "
            f"{int((t.isna() & p.isna()).sum()):>7}")
    return "\n".join(lines)


def relevance_report(courses, n_resamples=None):
    """Relevance agreement on courses found by both sides, then the end-to-end question:
    which relevant drugs were found."""
    matched = courses[courses["found"] == "both"]
    lines = []
    if len(matched):
        t = (matched["relevance_true"] == "relevant").to_numpy()
        p = (matched["relevance_pred"] == "relevant").to_numpy()
        agree = int((t == p).sum())
        lines.append(f"Courses found by both: relevance agrees on {agree} of {len(matched)} "
            f"({agree / len(matched):.0%}); human relevant {int(t.sum())}, "
            f"predicted relevant {int(p.sum())}")
    relevant_only = courses[
        ((courses["found"] != "pred_only") & (courses["relevance_true"] == "relevant")) |
        ((courses["found"] != "true_only") & (courses["relevance_pred"] == "relevant"))].copy()
    # Re-label which side counts each course as a relevant drug
    is_t = (relevant_only["found"] != "pred_only") & (relevant_only["relevance_true"] == "relevant")
    is_p = (relevant_only["found"] != "true_only") & (relevant_only["relevance_pred"] == "relevant")
    relevant_only["found"] = np.select([is_t & is_p, is_t], ["both", "true_only"], "pred_only")
    lines.append(detection_report(relevant_only, "Relevant drug courses"))
    return "\n".join(lines)


def evaluate_antibiotic_labels(df_true, df_pred, ep_ids, n_resamples=None):
    """Builds the text report comparing `df_pred` against `df_true` over episodes `ep_ids`.

    Both frames hold antibiotic_labels rows with a `relevance` column, including the
    "none found" rows (drug_name_raw NULL). Episodes in `ep_ids` that `df_pred` has no rows for
    count as nothing found."""
    df_true = df_true[df_true["FK_episode_id"].isin(ep_ids)]
    df_pred = df_pred[df_pred["FK_episode_id"].isin(ep_ids)]
    drugs_true = df_true[df_true["drug_name_raw"].notna()]
    drugs_pred = df_pred[df_pred["drug_name_raw"].notna()]

    any_true = np.isin(ep_ids, drugs_true["FK_episode_id"].unique())
    any_pred = np.isin(ep_ids, drugs_pred["FK_episode_id"].unique())
    episode_cm = EpisodeConfusionMatrix(any_true, any_pred, n_resamples=n_resamples)

    courses = match_courses(drugs_true, drugs_pred)
    unlisted_true = int(drugs_true["drug_name"].isna().sum())
    unlisted_pred = int(drugs_pred["drug_name"].isna().sum())

    # Bootstrap resamples of small or one-sided samples make sklearn warn about single-class
    #   confusion matrices and undefined kappa; the affected metrics just show nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        episode_report = str(episode_cm)

    sections = [
        "== Episodes with any antibiotic ==\n" + episode_report,
        "== Drug courses (matched by drug within each episode) ==\n" +
            detection_report(courses, "Drug courses") +
            f"\n  Not in drugs.db (matched by raw name): {unlisted_true} labeled, "
            f"{unlisted_pred} predicted",
        "== Fields of courses found by both ==\n" +
            field_report(courses[courses["found"] == "both"]),
        "== Relevance ==\n" + relevance_report(courses, n_resamples),
    ]
    return "\n\n".join(sections), courses
