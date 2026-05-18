from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
MODELS_DIR = ROOT_DIR / "models"
VISUALS_DIR = ROOT_DIR / "visuals"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = ROOT_DIR / "metrics"

FEATURE_COLUMNS = [
    "wpm",
    "accuracy",
    "error_rate",
    "backspace_count",
    "pause_time_ms",
    "session_duration_min",
    "words_typed",
]
TARGET_COLUMN = "typing_style"
CLASS_ORDER = ["Fast Typist", "Balanced Typist", "Careful Typist", "Inconsistent Typist"]


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    weighted_f1: float
    accuracy: float
    predictions: np.ndarray


def ensure_directories() -> None:
    for directory in [DATA_DIR, NOTEBOOKS_DIR, MODELS_DIR, VISUALS_DIR, REPORTS_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _bounded_normal(rng: np.random.Generator, mean: float, std: float, low: float, high: float, size: int) -> np.ndarray:
    values = rng.normal(mean, std, size)
    return np.clip(values, low, high)


def _style_profiles(style: str, size: int, rng: np.random.Generator) -> pd.DataFrame:
    if style == "Fast Typist":
        wpm = _bounded_normal(rng, 100, 10, 78, 120, size)
        accuracy = _bounded_normal(rng, 94.5, 2.2, 86, 100, size)
        error_rate = np.clip(100 - accuracy + rng.normal(1.4, 1.2, size), 0, 12)
        backspace_count = np.clip(rng.normal(8, 4, size), 0, 24)
        pause_time_ms = _bounded_normal(rng, 220, 90, 50, 520, size)
        session_duration_min = _bounded_normal(rng, 16, 7, 3, 40, size)
    elif style == "Balanced Typist":
        wpm = _bounded_normal(rng, 72, 9, 50, 100, size)
        accuracy = _bounded_normal(rng, 91.5, 3.1, 82, 99.5, size)
        error_rate = np.clip(100 - accuracy + rng.normal(2.2, 1.4, size), 0, 16)
        backspace_count = np.clip(rng.normal(16, 5, size), 1, 30)
        pause_time_ms = _bounded_normal(rng, 420, 130, 120, 850, size)
        session_duration_min = _bounded_normal(rng, 20, 8, 4, 50, size)
    elif style == "Careful Typist":
        wpm = _bounded_normal(rng, 54, 11, 20, 78, size)
        accuracy = _bounded_normal(rng, 97.2, 1.5, 92, 100, size)
        error_rate = np.clip(100 - accuracy + rng.normal(0.7, 0.8, size), 0, 8)
        backspace_count = np.clip(rng.normal(11, 4, size), 0, 22)
        pause_time_ms = _bounded_normal(rng, 920, 240, 380, 1500, size)
        session_duration_min = _bounded_normal(rng, 25, 10, 5, 60, size)
    else:
        wpm = _bounded_normal(rng, 69, 22, 20, 120, size)
        accuracy = _bounded_normal(rng, 84, 7.5, 70, 97, size)
        error_rate = np.clip(100 - accuracy + rng.normal(6, 3, size), 3, 30)
        backspace_count = np.clip(rng.normal(33, 14, size), 0, 80)
        pause_time_ms = _bounded_normal(rng, 710, 270, 150, 1500, size)
        session_duration_min = _bounded_normal(rng, 18, 11, 2, 60, size)

    words_typed = np.maximum(np.round(wpm * session_duration_min / 60 * rng.uniform(0.92, 1.08, size)), 1).astype(int)
    return pd.DataFrame(
        {
            "wpm": np.round(wpm, 1),
            "accuracy": np.round(accuracy, 1),
            "error_rate": np.round(error_rate, 1),
            "backspace_count": np.round(backspace_count).astype(int),
            "pause_time_ms": np.round(pause_time_ms).astype(int),
            "session_duration_min": np.round(session_duration_min, 1),
            "words_typed": words_typed,
            TARGET_COLUMN: style,
        }
    )


def generate_synthetic_dataset(n_sessions: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    style_weights = np.array([0.26, 0.33, 0.18, 0.23])
    styles = rng.choice(CLASS_ORDER, size=n_sessions, p=style_weights)

    frames = []
    for style in CLASS_ORDER:
        style_size = int(np.sum(styles == style))
        if style_size:
            frames.append(_style_profiles(style, style_size, rng))
    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    data.insert(0, "session_id", np.arange(1, len(data) + 1))
    return data


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned["session_duration_min"] = cleaned["session_duration_min"].clip(2, 60)
    cleaned["wpm"] = cleaned["wpm"].clip(20, 120)
    cleaned["accuracy"] = cleaned["accuracy"].clip(70, 100)
    cleaned["error_rate"] = cleaned["error_rate"].clip(0, 30)
    cleaned["backspace_count"] = cleaned["backspace_count"].clip(0, 80)
    cleaned["pause_time_ms"] = cleaned["pause_time_ms"].clip(50, 1500)
    cleaned["words_typed"] = cleaned["words_typed"].clip(lower=1)
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_style_distribution(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index, counts.values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_title("Typing Style Distribution")
    ax.set_ylabel("Sessions")
    ax.set_xlabel("Typing Style")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, padding=3)
    plt.xticks(rotation=12)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "style_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=35, ha="right")
    ax.set_yticklabels(FEATURE_COLUMNS)
    ax.set_title("Feature Correlation Heatmap")
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_models(X_train: pd.DataFrame, y_train: np.ndarray, random_state: int = 42) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=260,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                    ),
                )
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=random_state,
                        learning_rate=0.08,
                        max_depth=3,
                        n_estimators=180,
                    ),
                )
            ]
        ),
    }
    for pipeline in models.values():
        pipeline.fit(X_train, y_train)
    return models


def evaluate_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[Dict[str, ModelResult], str]:
    results: Dict[str, ModelResult] = {}
    best_name = ""
    best_score = -1.0
    for name, pipeline in models.items():
        predictions = pipeline.predict(X_test)
        score = f1_score(y_test, predictions, average="weighted")
        accuracy = accuracy_score(y_test, predictions)
        results[name] = ModelResult(name=name, pipeline=pipeline, weighted_f1=score, accuracy=accuracy, predictions=predictions)
        if score > best_score:
            best_score = score
            best_name = name
    return results, best_name


def feature_importance_values(model: Pipeline, feature_names: Iterable[str]) -> pd.Series:
    estimator = model.named_steps["model"]
    feature_names = list(feature_names)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).mean(axis=0)
    else:
        values = np.ones(len(feature_names))
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(model: Pipeline) -> None:
    importance = feature_importance_values(model, FEATURE_COLUMNS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance.index[::-1], importance.values[::-1], color="#4c72b0")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_test: np.ndarray, predictions: np.ndarray, encoder: LabelEncoder) -> None:
    matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    labels = encoder.inverse_transform(np.arange(len(encoder.classes_)))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(VISUALS_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_reports(metrics: dict, best_report: str) -> None:
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    model_comparison = pd.DataFrame.from_dict(metrics["model_scores"], orient="index").reset_index()
    model_comparison = model_comparison.rename(columns={"index": "model"})
    model_comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    feature_summary = pd.DataFrame.from_dict(metrics["feature_summary"], orient="index")
    feature_summary.to_csv(REPORTS_DIR / "feature_summary.csv")
    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": metrics["best_model"],
        "random_state": 42,
        "artifact_group": "TypingDNA Analytics",
    }
    with open(REPORTS_DIR / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2)
    with open(METRICS_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(best_report)


def verify_artifacts(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected artifacts: {', '.join(missing)}")


def build_project_report(df: pd.DataFrame, best_model_name: str, metrics: dict) -> str:
    class_counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    summary = df[FEATURE_COLUMNS].describe().round(2)
    model_rows = "\n".join(
        f"| {name} | {score['weighted_f1']:.4f} | {score['accuracy']:.4f} |"
        for name, score in metrics["model_scores"].items()
    )
    return f"""# TypingDNA Analytics Project Report

## Overview

This project generated {len(df):,} synthetic typing sessions and trained three supervised classification models to identify typing behavior styles.

## Target Distribution

{class_counts.to_string()}

## Target Distribution (%)

{pd.Series(metrics['class_distribution_pct']).to_string()}

## Best Model

- Model: {best_model_name}
- Weighted F1: {metrics['best_weighted_f1']:.4f}
- Accuracy: {metrics['best_accuracy']:.4f}

## Model Selection Rationale

The winning model balances precision and recall across all four classes while capturing the threshold-based structure baked into the synthetic labels. In this dataset, tree-based models have an advantage because the decision boundaries depend on combinations of speed, accuracy, corrections, and pause timing.

## Model Comparison

| Model | Weighted F1 | Accuracy |
| --- | ---: | ---: |
{model_rows}

## Dataset Summary

{summary.to_string()}

## Key Findings

- Faster sessions tend to pair high WPM with lower pause time and fewer corrections.
- Careful typists show the highest accuracy and the longest pause times.
- Inconsistent typists are separated by wider spreads in error rate, backspace usage, and session rhythm.
- Tree-based models are expected to outperform the linear baseline because the classes are rule-driven and non-linear.

## Limitations

- The labels are synthetic, so the project is ideal for demonstrating machine learning workflow rather than proving real-user biometric behavior.
- The target classes are intentionally structured, which makes the prediction task easier than a noisy production setting.

## Next Steps

- Extend the generator with more session-level telemetry such as key-hold timing and dwell time.
- Add cross-validation and calibration curves for a more complete evaluation view.
- Package the pipeline into a command-line interface for repeated runs with different random seeds.

## Recommended Use

Use the saved best model for batch scoring or portfolio demonstrations. The artifacts in `models/`, `visuals/`, `metrics/`, and `reports/` are all reproducible from the pipeline.
"""


def build_executive_summary(metrics: dict) -> str:
    return f"""# TypingDNA Analytics Executive Summary

- Best model: {metrics['best_model']}
- Weighted F1-score: {metrics['best_weighted_f1']:.4f}
- Accuracy: {metrics['best_accuracy']:.4f}
- Key takeaway: typing behavior can be separated reliably using speed, correction, and pause-pattern features.
- Portfolio value: the project demonstrates synthetic data generation, EDA, model benchmarking, and artifact packaging end to end.
"""


def run_pipeline(random_state: int = 42) -> dict:
    ensure_directories()
    raw = generate_synthetic_dataset(n_sessions=5000, random_state=random_state)
    cleaned = clean_dataset(raw)

    save_dataframe(raw, DATA_DIR / "typing_behavior.csv")
    save_dataframe(cleaned, DATA_DIR / "cleaned_typing_behavior.csv")

    plot_style_distribution(cleaned)
    plot_correlation_heatmap(cleaned)

    encoder = LabelEncoder()
    encoder.fit(CLASS_ORDER)
    y = encoder.transform(cleaned[TARGET_COLUMN])
    X = cleaned[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    models = train_models(X_train, y_train, random_state=random_state)
    results, best_name = evaluate_models(models, X_test, y_test)
    best_result = results[best_name]

    joblib.dump(best_result.pipeline, MODELS_DIR / "typing_style_classifier.pkl")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.pkl")

    plot_feature_importance(best_result.pipeline)
    plot_confusion_matrix(y_test, best_result.predictions, encoder)

    best_predictions_labels = encoder.inverse_transform(best_result.predictions)
    best_report = classification_report(y_test, best_result.predictions, target_names=encoder.inverse_transform(np.arange(len(encoder.classes_))))

    metrics = {
        "best_model": best_name,
        "best_weighted_f1": best_result.weighted_f1,
        "best_accuracy": best_result.accuracy,
        "class_distribution": cleaned[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER).to_dict(),
        "class_distribution_pct": (cleaned[TARGET_COLUMN].value_counts(normalize=True).reindex(CLASS_ORDER) * 100).round(2).to_dict(),
        "feature_summary": cleaned[FEATURE_COLUMNS].describe().round(2).to_dict(),
        "model_scores": {
            name: {
                "weighted_f1": result.weighted_f1,
                "accuracy": result.accuracy,
            }
            for name, result in results.items()
        },
        "feature_columns": FEATURE_COLUMNS,
        "n_sessions": int(len(cleaned)),
    }

    write_reports(metrics, best_report)

    project_report = build_project_report(cleaned, best_name, metrics)
    with open(REPORTS_DIR / "project_report.md", "w", encoding="utf-8") as file:
        file.write(project_report)
    with open(REPORTS_DIR / "executive_summary.md", "w", encoding="utf-8") as file:
        file.write(build_executive_summary(metrics))

    verify_artifacts(
        [
            DATA_DIR / "typing_behavior.csv",
            DATA_DIR / "cleaned_typing_behavior.csv",
            MODELS_DIR / "typing_style_classifier.pkl",
            MODELS_DIR / "label_encoder.pkl",
            VISUALS_DIR / "style_distribution.png",
            VISUALS_DIR / "correlation_heatmap.png",
            VISUALS_DIR / "feature_importance.png",
            VISUALS_DIR / "confusion_matrix.png",
            REPORTS_DIR / "model_metrics.json",
            REPORTS_DIR / "model_comparison.csv",
            REPORTS_DIR / "feature_summary.csv",
            REPORTS_DIR / "run_metadata.json",
            REPORTS_DIR / "project_report.md",
            REPORTS_DIR / "executive_summary.md",
            METRICS_DIR / "classification_report.txt",
        ]
    )

    return {
        "raw_path": str(DATA_DIR / "typing_behavior.csv"),
        "cleaned_path": str(DATA_DIR / "cleaned_typing_behavior.csv"),
        "best_model": best_name,
        "weighted_f1": best_result.weighted_f1,
        "accuracy": best_result.accuracy,
        "model_paths": [str(MODELS_DIR / "typing_style_classifier.pkl"), str(MODELS_DIR / "label_encoder.pkl")],
        "visuals": [
            str(VISUALS_DIR / "style_distribution.png"),
            str(VISUALS_DIR / "correlation_heatmap.png"),
            str(VISUALS_DIR / "feature_importance.png"),
            str(VISUALS_DIR / "confusion_matrix.png"),
        ],
        "report_path": str(REPORTS_DIR / "project_report.md"),
    }


if __name__ == "__main__":
    outcome = run_pipeline()
    print(json.dumps(outcome, indent=2))
