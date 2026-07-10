"""Utility functions used by the Musica demonstration notebook."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.metrics import classification_report, confusion_matrix

COLORS = {
    "blue": "#4C78A8",
    "teal": "#72B7B2",
    "green": "#54A24B",
    "orange": "#F58518",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#6B7280",
}
SPLIT_COLORS = {
    "train": COLORS["green"],
    "validation": COLORS["orange"],
    "test": COLORS["red"],
}


def setup_notebook_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titleweight"] = "bold"


def show_table(
        title: str,
        rows: list[tuple[Any, Any]],
        col_labels: tuple[str, str] = ("Champ", "Valeur"),
        figsize: tuple[float, float] | None = None,
) -> None:
    table_rows = [(str(left), str(right)) for left, right in rows]
    height = max(1.8, 0.42 * len(table_rows) + 1.1)
    width = 12 if figsize is None else figsize[0]
    height = height if figsize is None else figsize[1]

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.28, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(COLORS["blue"])
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "white")
    ax.set_title(title, pad=14)
    plt.show()


def annotate_bars(ax: Any, fmt: str = "{:.0f}", padding: int = 3) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding)


def source_from_path(path: Path, dataset_dir: Path) -> str:
    relative_parts = path.relative_to(dataset_dir).parts
    return relative_parts[0] if relative_parts else "inconnu"


def counts_for_labels(paths: list[Path], dataset: Any) -> np.ndarray:
    counts = Counter(dataset.label_from_path(path) for path in paths)
    return np.array([counts.get(label, 0) for label in dataset.labels])


def dataset_audit(prepared: Any) -> dict[str, Any]:
    dataset = prepared.dataset
    split_paths = {
        "train": prepared.split.train_paths,
        "validation": prepared.split.val_paths,
        "test": prepared.split.test_paths,
    }
    return {
        "dataset": dataset,
        "split_paths": split_paths,
        "split_sizes": {name: len(paths) for name, paths in split_paths.items()},
        "source_counts": Counter(
            source_from_path(path, dataset.dataset_dir) for path in dataset.audio_paths
        ),
        "quality_counts": Counter(
            dataset.label_from_path(path).split("_")[1] for path in dataset.audio_paths
        ),
        "split_count_matrix": np.vstack(
            [counts_for_labels(paths, dataset) for paths in split_paths.values()]
        ),
    }


def show_dataset_summary(prepared: Any, audit: dict[str, Any]) -> None:
    dataset = audit["dataset"]
    split_sizes = audit["split_sizes"]
    show_table(
        "Résumé du dataset",
        [
            ("Fichiers WAV", format_int(len(dataset.audio_paths))),
            ("Classes", len(dataset.labels)),
            ("Sources audio", ", ".join(sorted(audit["source_counts"]))),
            ("Train", split_sizes["train"]),
            ("Validation", split_sizes["validation"]),
            ("Test", split_sizes["test"]),
            ("Shape train", prepared.x_train.shape),
            ("Shape validation", prepared.x_val.shape),
            ("Shape test", prepared.x_test.shape),
        ],
    )


def plot_dataset_overview(prepared: Any, audit: dict[str, Any]) -> None:
    dataset = audit["dataset"]
    split_sizes = audit["split_sizes"]
    source_counts = audit["source_counts"]
    quality_counts = audit["quality_counts"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))

    summary_labels = ["Audio", "Classes", "Train", "Validation", "Test"]
    summary_values = [
        len(dataset.audio_paths),
        len(dataset.labels),
        split_sizes["train"],
        split_sizes["validation"],
        split_sizes["test"],
    ]
    axes[0, 0].bar(
        summary_labels,
        summary_values,
        color=[
            COLORS["blue"],
            COLORS["teal"],
            COLORS["green"],
            COLORS["orange"],
            COLORS["red"],
        ],
    )
    axes[0, 0].set_title("Résumé du dataset")
    axes[0, 0].set_ylabel("Nombre")
    annotate_bars(axes[0, 0])

    feature_shapes = {
        "train": prepared.x_train.shape[0],
        "validation": prepared.x_val.shape[0],
        "test": prepared.x_test.shape[0],
    }
    axes[0, 1].bar(
        feature_shapes.keys(),
        feature_shapes.values(),
        color=[SPLIT_COLORS[name] for name in feature_shapes],
    )
    axes[0, 1].set_title("Features extraites")
    axes[0, 1].set_ylabel("Exemples")
    axes[0, 1].text(
        0.5,
        -0.22,
        f"Shape feature: {prepared.x_train.shape[1:]}",
        ha="center",
        transform=axes[0, 1].transAxes,
    )
    annotate_bars(axes[0, 1])

    source_names, source_values = zip(*sorted(source_counts.items()))
    axes[1, 0].bar(source_names, source_values, color=COLORS["blue"])
    axes[1, 0].set_title("Composition du dataset par source")
    axes[1, 0].set_ylabel("Fichiers WAV")
    axes[1, 0].tick_params(axis="x", rotation=20)
    annotate_bars(axes[1, 0])

    quality_names, quality_values = zip(*sorted(quality_counts.items()))
    axes[1, 1].bar(quality_names, quality_values, color=COLORS["teal"])
    axes[1, 1].set_title("Répartition par qualité d'accord")
    axes[1, 1].set_ylabel("Fichiers WAV")
    annotate_bars(axes[1, 1])

    plt.tight_layout()
    plt.show()


def plot_split_distribution(dataset: Any, audit: dict[str, Any]) -> None:
    split_sizes = audit["split_sizes"]
    split_paths = audit["split_paths"]
    split_count_matrix = audit["split_count_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].pie(
        split_sizes.values(),
        labels=split_sizes.keys(),
        autopct="%.1f%%",
        startangle=90,
        colors=[SPLIT_COLORS[name] for name in split_sizes],
    )
    axes[0].set_title(f"Répartition des splits ({sum(split_sizes.values())} fichiers)")

    x = np.arange(len(dataset.labels))
    width = 0.26
    for offset, (name, paths) in zip(
            [-width, 0, width],
            split_paths.items(),
            strict=True,
    ):
        axes[1].bar(
            x + offset,
            counts_for_labels(paths, dataset),
            width,
            label=name,
            color=SPLIT_COLORS[name],
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dataset.labels, rotation=90)
    axes[1].set_ylabel("Nombre de fichiers")
    axes[1].set_title("Distribution des classes par split")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 3.8))
    sns.heatmap(
        split_count_matrix,
        cmap="YlGnBu",
        annot=True,
        fmt="d",
        xticklabels=dataset.labels,
        yticklabels=list(split_paths.keys()),
        cbar_kws={"label": "Fichiers"},
    )
    plt.title("Contrôle stratifié par classe")
    plt.xlabel("Classe")
    plt.ylabel("Split")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_audio_and_feature(
        audio_path: Path,
        feature: np.ndarray,
        label: str,
        config: Any,
) -> None:
    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    duration = len(audio) / config.sample_rate
    time_axis = np.linspace(0, duration, num=len(audio), endpoint=False)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )
    axes[0].plot(time_axis, audio, color=COLORS["blue"], linewidth=1)
    axes[0].set_title(f"Signal audio - {audio_path.name} ({label})")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Amplitude")

    image = axes[1].imshow(feature.T, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Feature Chroma-CQT transmise au CNN")
    axes[1].set_xlabel("Trames temporelles")
    axes[1].set_ylabel("Classe de hauteur")
    axes[1].set_yticks(range(config.n_chroma))
    axes[1].set_yticklabels(config_roots())
    fig.colorbar(image, ax=axes[1], label="Intensité")

    plt.tight_layout()
    plt.show()


def config_roots() -> list[str]:
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def plot_training_curves(history_log_path: Path) -> dict[str, Any]:
    log_data = np.genfromtxt(history_log_path, delimiter=",", names=True)
    curves = {name: np.atleast_1d(log_data[name]) for name in log_data.dtype.names}
    epochs = curves["epoch"] if "epoch" in curves else np.arange(len(curves["loss"]))
    best_index = int(np.argmin(curves["val_loss"]))
    best_epoch = int(epochs[best_index])
    best_val_loss = float(curves["val_loss"][best_index])
    best_val_accuracy = float(curves["val_accuracy"][best_index])

    lr_changes = []
    if "learning_rate" in curves:
        learning_rates = curves["learning_rate"]
        lr_changes = [
            int(epochs[index])
            for index in range(1, len(learning_rates))
            if learning_rates[index] != learning_rates[index - 1]
        ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(epochs, curves["accuracy"], label="train", color=COLORS["green"])
    axes[0].plot(
        epochs,
        curves["val_accuracy"],
        label="validation",
        color=COLORS["orange"],
    )
    axes[0].axvline(best_epoch, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    axes[1].plot(epochs, curves["loss"], label="train", color=COLORS["green"])
    axes[1].plot(
        epochs,
        curves["val_loss"],
        label="validation",
        color=COLORS["orange"],
    )
    axes[1].scatter(
        [best_epoch],
        [best_val_loss],
        color=COLORS["red"],
        zorder=3,
        label=f"meilleure val_loss: {best_val_loss:.3f}",
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Époque")
    axes[1].legend()

    if "learning_rate" in curves:
        axes[2].step(epochs, curves["learning_rate"], where="post", color=COLORS["purple"])
        axes[2].set_yscale("log")
        axes[2].set_title("Learning rate")
        axes[2].set_xlabel("Époque")
        axes[2].set_ylabel("lr")
        for epoch in lr_changes:
            axes[2].axvline(epoch, color=COLORS["gray"], linestyle="--", linewidth=0.8)
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Learning rate non disponible", ha="center", va="center")

    plt.tight_layout()
    plt.show()

    return {
        "curves": curves,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
    }


def plot_test_metrics(evaluation: Any) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(
        ["Accuracy", "F1 macro"],
        [evaluation.test_accuracy, evaluation.f1_macro],
        color=[COLORS["blue"], COLORS["teal"]],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Scores de généralisation")
    annotate_bars(axes[0], fmt="%.3f")

    axes[1].bar(["Loss"], [evaluation.test_loss], color=COLORS["red"])
    axes[1].set_title("Erreur test")
    annotate_bars(axes[1], fmt="%.3f")

    plt.tight_layout()
    plt.show()


def evaluate_predictions(model: Any, prepared: Any, labels: list[str]) -> dict[str, Any]:
    y_pred_proba = model.predict(prepared.x_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    report = classification_report(
        prepared.y_test,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(prepared.y_test, y_pred, labels=np.arange(len(labels)))
    row_totals = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )
    return {
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "report": report,
        "cm": cm,
        "cm_normalized": cm_normalized,
    }


def plot_confusion_matrix(labels: list[str], cm_normalized: np.ndarray) -> None:
    plt.figure(figsize=(13, 11))
    sns.heatmap(
        cm_normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Part des exemples de la vraie classe"},
    )
    plt.xlabel("Prédiction")
    plt.ylabel("Vrai label")
    plt.title("Matrice de confusion normalisée")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def classification_report_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    total_support = report.get("weighted avg", {}).get("support", np.nan)
    rows = {}
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows[label] = metrics
        else:
            rows[label] = {
                "precision": metrics,
                "recall": metrics,
                "f1-score": metrics,
                "support": total_support,
            }

    dataframe = pd.DataFrame.from_dict(rows, orient="index")
    ordered_columns = ["precision", "recall", "f1-score", "support"]
    dataframe = dataframe[[column for column in ordered_columns if column in dataframe.columns]]
    for column in ("precision", "recall", "f1-score"):
        if column in dataframe:
            dataframe[column] = dataframe[column].round(3)
    if "support" in dataframe:
        dataframe["support"] = dataframe["support"].round(0).astype(int)
    return dataframe


def show_classification_report_dataframe(report: dict[str, Any]) -> pd.DataFrame:
    dataframe = classification_report_dataframe(report)
    display(dataframe)
    return dataframe


def plot_error_analysis(labels: list[str], report: dict[str, Any], cm: np.ndarray, cm_normalized: np.ndarray) -> None:
    class_scores = []
    for label in labels:
        metrics = report[label]
        class_scores.append(
            {
                "label": label,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
            }
        )
    weakest_classes = sorted(class_scores, key=lambda item: (item["f1"], item["recall"]))[:10]

    confusions = []
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if true_index != pred_index and cm[true_index, pred_index] > 0:
                confusions.append(
                    {
                        "true": true_label,
                        "pred": pred_label,
                        "count": int(cm[true_index, pred_index]),
                        "rate": float(cm_normalized[true_index, pred_index]),
                    }
                )
    top_confusions = sorted(
        confusions,
        key=lambda item: (item["count"], item["rate"]),
        reverse=True,
    )[:10]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    axes[0].barh(
        [item["label"] for item in weakest_classes][::-1],
        [item["f1"] for item in weakest_classes][::-1],
        color=COLORS["orange"],
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("F1")
    axes[0].set_title("Classes les plus difficiles")
    for index, item in enumerate(weakest_classes[::-1]):
        axes[0].text(min(item["f1"] + 0.01, 0.98), index, f'{item["f1"]:.3f}', va="center")

    if top_confusions:
        confusion_labels = [f'{item["true"]} → {item["pred"]}' for item in top_confusions][::-1]
        values = [item["count"] for item in top_confusions][::-1]
        axes[1].barh(confusion_labels, values, color=COLORS["red"])
        axes[1].set_xlabel("Erreurs")
        axes[1].set_title("Top confusions")
        for index, item in enumerate(top_confusions[::-1]):
            axes[1].text(item["count"] + 0.05, index, f'{item["rate"]:.1%}', va="center")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Aucune confusion sur le test", ha="center", va="center", fontsize=12)

    plt.tight_layout()
    plt.show()
    return top_confusions


def plot_example_predictions(predictor: Any, model: Any, dataset: Any, example_audio_files: list[Path]) -> list[
    tuple[str, str, str, float]]:
    fig, axes = plt.subplots(
        len(example_audio_files),
        1,
        figsize=(11, 3.2 * len(example_audio_files)),
    )
    axes = np.atleast_1d(axes)

    prediction_summaries = []
    for ax, audio_path in zip(axes, example_audio_files, strict=True):
        predictions = predictor.predict(model, audio_path)
        labels = [label for label, _ in predictions]
        probabilities = [probability for _, probability in predictions]
        expected_label = dataset.label_from_path(audio_path)
        predicted_label = labels[0]
        prediction_summaries.append(
            (audio_path.name, expected_label, predicted_label, probabilities[0])
        )
        colors = [
            COLORS["green"] if label == predicted_label else COLORS["blue"]
            for label in labels
        ]

        ax.barh(labels[::-1], probabilities[::-1], color=colors[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilité")
        ax.set_title(
            f"{audio_path.name} | attendu: {expected_label} | prédit: {predicted_label}"
        )
        for index, probability in enumerate(probabilities[::-1]):
            ax.text(min(probability + 0.02, 0.98), index, f"{probability:.3f}", va="center")

    plt.tight_layout()
    plt.show()
    return prediction_summaries


def display_prediction_summary(prediction_summaries: list[tuple[str, str, str, float]]) -> None:
    summary_text = "\n".join(
        f"- `{name}` : attendu `{expected}`, prédit `{predicted}` avec {format_probability(probability)}"
        for name, expected, predicted, probability in prediction_summaries
    )
    display(Markdown("Résumé des prédictions :\n" + summary_text))


def display_model_summary(model: Any) -> None:
    model.summary()


def format_probability(value: float) -> str:
    return f"{100 * value:.1f}%"


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")
