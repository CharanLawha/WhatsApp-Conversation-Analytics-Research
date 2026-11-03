"""Utilities for reproducible, annotation-driven conversation experiments."""

from __future__ import annotations

import re
from collections import Counter
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import emoji


SENTIMENT_LABELS = ("positive", "negative", "neutral")
REQUIRED_ANNOTATION_COLUMNS = ("message", "sentiment", "conversation_id")
AUTHOR_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_annotated_dataset(data: bytes, filename: str) -> pd.DataFrame:
    """Load a user-annotated CSV or JSON file without generating labels."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if suffix == "csv":
            frame = pd.read_csv(BytesIO(data))
        elif suffix == "json":
            frame = pd.read_json(BytesIO(data))
        else:
            raise ValueError("Upload an annotated .csv or .json file.")
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read the annotation file: {exc}") from exc

    validate_annotated_dataset(frame)
    frame = frame.copy()
    frame["message"] = frame["message"].fillna("").astype(str).str.strip()
    frame["sentiment"] = frame["sentiment"].fillna("").astype(str).str.strip().str.lower()
    frame["conversation_id"] = (
        frame["conversation_id"].fillna("").astype(str).str.strip()
    )
    return frame.reset_index(drop=True)


def validate_annotated_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate labels and fields required for conversation-aware evaluation."""
    missing = [column for column in REQUIRED_ANNOTATION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The annotation dataset is empty.")

    data = frame.loc[:, REQUIRED_ANNOTATION_COLUMNS].copy()
    data["message"] = data["message"].fillna("").astype(str).str.strip()
    data["sentiment"] = data["sentiment"].fillna("").astype(str).str.strip().str.lower()
    data["conversation_id"] = data["conversation_id"].fillna("").astype(str).str.strip()

    invalid_labels = sorted(set(data["sentiment"]) - set(SENTIMENT_LABELS))
    if invalid_labels:
        raise ValueError(
            "Sentiment labels must be positive, negative, or neutral. "
            f"Invalid labels: {', '.join(invalid_labels)}"
        )
    if (data["message"] == "").any():
        raise ValueError("Messages cannot be blank.")
    if (data["conversation_id"] == "").any():
        raise ValueError("Every row needs a conversation_id for leakage-safe splitting.")
    if data["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required for evaluation.")

    emotion_distribution = {}
    if "emotion" in frame.columns:
        emotions = frame["emotion"].dropna().astype(str).str.strip()
        emotions = emotions[emotions != ""]
        emotion_distribution = {
            str(label): int(count)
            for label, count in emotions.value_counts().sort_index().items()
        }
    return {
        "rows": int(len(data)),
        "conversations": int(data["conversation_id"].nunique()),
        "class_distribution": {
            str(label): int(count)
            for label, count in data["sentiment"].value_counts().sort_index().items()
        },
        "has_emotion_annotations": "emotion" in frame.columns,
        "emotion_class_distribution": emotion_distribution,
    }


def conversation_train_validation_test_split(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole conversations into train, validation, and test partitions."""
    frame = frame.copy()
    frame["message"] = frame["message"].fillna("").astype(str).str.strip()
    frame["sentiment"] = frame["sentiment"].fillna("").astype(str).str.strip().str.lower()
    frame["conversation_id"] = (
        frame["conversation_id"].fillna("").astype(str).str.strip()
    )
    validate_annotated_dataset(frame)
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split proportions must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("Test and validation proportions must sum to less than 1.")

    groups = frame["conversation_id"].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3:
        raise ValueError(
            "At least three distinct conversations are required for train/validation/test splitting."
        )

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_validation_idx, test_idx = next(
        first_split.split(frame, groups=groups)
    )
    train_validation = frame.iloc[train_validation_idx]
    remaining_groups = groups.iloc[train_validation_idx]
    relative_validation_size = validation_size / (1 - test_size)

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_size,
        random_state=random_state,
    )
    train_idx, validation_idx = next(
        second_split.split(train_validation, groups=remaining_groups)
    )
    train = train_validation.iloc[train_idx].copy()
    validation = train_validation.iloc[validation_idx].copy()
    test = frame.iloc[test_idx].copy()

    group_sets = [
        set(part["conversation_id"].astype(str))
        for part in (train, validation, test)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise RuntimeError("Conversation leakage detected across data partitions.")
    return train, validation, test


def build_tfidf_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Build baselines whose vectorizers are fit only on each training partition."""
    def create_vectorizer():
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        )

    return {
        "TF-IDF + Linear SVM": Pipeline(
            [
                ("tfidf", create_vectorizer()),
                ("classifier", LinearSVC(random_state=random_state)),
            ]
        ),
        "TF-IDF + Naive Bayes": Pipeline(
            [("tfidf", create_vectorizer()), ("classifier", MultinomialNB())]
        ),
        "TF-IDF + Random Forest": Pipeline(
            [
                ("tfidf", create_vectorizer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def score_predictions(
    y_true: list[str] | pd.Series,
    y_pred: list[str] | np.ndarray,
    labels: tuple[str, ...] = SENTIMENT_LABELS,
) -> dict[str, Any]:
    """Return standard classification metrics and a labeled confusion matrix."""
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": {
            "labels": list(labels),
            "values": matrix.astype(int).tolist(),
        },
    }


def evaluate_tfidf_sentiment_baselines(
    frame: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate VADER and fold-safe TF-IDF baselines on held-out conversations."""
    train, validation, test = conversation_train_validation_test_split(
        frame, random_state=random_state
    )
    results: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, Any]] = {}
    vader = SentimentIntensityAnalyzer()
    vader_predictions = [
        _vader_label(message, vader) for message in test["message"].astype(str)
    ]
    vader_metrics = score_predictions(
        test["sentiment"].astype(str).str.lower(), vader_predictions
    )
    results.append({
        "model": "VADER",
        "validation_accuracy": None,
        **{
            key: value for key, value in vader_metrics.items()
            if key != "confusion_matrix"
        },
    })
    matrices["VADER"] = vader_metrics["confusion_matrix"]

    for model_name, model in build_tfidf_models(random_state).items():
        model.fit(
            train["message"].astype(str),
            train["sentiment"].astype(str).str.lower(),
        )
        validation_predictions = model.predict(validation["message"].astype(str))
        validation_metrics = score_predictions(
            validation["sentiment"].astype(str).str.lower(),
            validation_predictions,
        )
        predictions = model.predict(test["message"].astype(str))
        metrics = score_predictions(
            test["sentiment"].astype(str).str.lower(),
            predictions,
        )
        results.append({
            "model": model_name,
            "validation_accuracy": validation_metrics["accuracy"],
            **{
                key: value for key, value in metrics.items()
                if key != "confusion_matrix"
            },
        })
        matrices[model_name] = metrics["confusion_matrix"]

    return pd.DataFrame(results), matrices


def evaluate_transformer_sentiment(
    frame: pd.DataFrame,
    model_name: str,
    context_messages: int = 0,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate an explicitly selected sentiment checkpoint on held-out conversations."""
    if context_messages and context_messages not in range(2, 6):
        raise ValueError("Context must include between 2 and 5 preceding messages.")
    _, _, test = conversation_train_validation_test_split(
        frame, random_state=random_state
    )
    import torch
    from model_runtime import load_transformer_pipeline

    classifier = load_transformer_pipeline(
        "sentiment-analysis",
        model_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    inputs = []
    ordered_test = test.copy()
    order_column = next(
        (column for column in ("timestamp", "date_time", "datetime") if column in test),
        None,
    )
    if order_column:
        ordered_test[order_column] = pd.to_datetime(
            ordered_test[order_column], errors="coerce"
        )
        ordered_test = ordered_test.sort_values(
            ["conversation_id", order_column], kind="stable"
        )

    for _, conversation in ordered_test.groupby("conversation_id", sort=False):
        history: list[tuple[str, str]] = []
        for _, row in conversation.iterrows():
            message = str(row["message"])
            context = history[-context_messages:] if context_messages else []
            speaker = str(row.get("user", "speaker"))
            inputs.append(
                "\n".join(
                    [f"[{previous_speaker}] {previous_message}"
                     for previous_speaker, previous_message in context]
                    + [f"[{speaker}] {message}"]
                )
                if context_messages
                else message
            )
            history.append((speaker, message))

    predictions = classifier(inputs, truncation=True, batch_size=16)
    labels = [
        _normalize_sentiment_label(prediction["label"])
        for prediction in predictions
    ]
    metrics = score_predictions(
        ordered_test["sentiment"].astype(str).str.lower().tolist(),
        labels,
    )
    return {
        "model": model_name,
        "validation_accuracy": None,
        **{
            key: value for key, value in metrics.items()
            if key != "confusion_matrix"
        },
    }, metrics["confusion_matrix"]


def _normalize_sentiment_label(label: str) -> str:
    normalized = str(label).strip().lower()
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "label_2": "positive",
        "negative": "negative",
        "neg": "negative",
        "label_0": "negative",
        "neutral": "neutral",
        "label_1": "neutral",
    }
    if normalized not in mapping:
        raise ValueError(f"Unrecognized sentiment model label: {label}")
    return mapping[normalized]


def _vader_label(message: str, analyzer: SentimentIntensityAnalyzer) -> str:
    scores = analyzer.polarity_scores(str(message))
    if scores["compound"] >= 0.05:
        return "positive"
    if scores["compound"] <= -0.05:
        return "negative"
    return "neutral"


def evaluate_authorship_baselines(
    frame: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate author classifiers using whole-conversation holdout partitions."""
    frame = frame.copy()
    for column in ("message", "user", "conversation_id"):
        if column in frame:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    if "user" in frame:
        frame = frame[frame["user"] != "group_notification"].copy()
    required = ("message", "user", "conversation_id")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if (frame[list(required)] == "").any().any():
        raise ValueError("Messages, authors, and conversation IDs cannot be blank.")
    if frame["user"].nunique() < 2:
        raise ValueError("At least two distinct authors are required.")
    if frame["conversation_id"].nunique() < 3:
        raise ValueError("At least three conversations are required for held-out evaluation.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_validation_idx, test_idx = next(
        splitter.split(frame, groups=frame["conversation_id"].astype(str))
    )
    train_validation = frame.iloc[train_validation_idx]
    groups = train_validation["conversation_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=random_state)
    train_idx, validation_idx = next(splitter.split(train_validation, groups=groups))
    train = train_validation.iloc[train_idx]
    validation = train_validation.iloc[validation_idx]
    test = frame.iloc[test_idx]

    labels = tuple(sorted(frame["user"].astype(str).unique()))
    if set(train["user"].astype(str)) != set(labels):
        raise ValueError("Every author must appear in the training conversations.")
    results = []
    matrices = {}
    for model_name, model in build_author_models(random_state).items():
        model.fit(train["message"].astype(str), train["user"].astype(str))
        validation_predictions = model.predict(validation["message"].astype(str))
        validation_metrics = score_predictions(
            validation["user"].astype(str), validation_predictions, labels=labels
        )
        for partition_name, partition in (("validation", validation), ("test", test)):
            predictions = (
                validation_predictions
                if partition_name == "validation"
                else model.predict(partition["message"].astype(str))
            )
            metrics = score_predictions(
                partition["user"].astype(str),
                predictions,
                labels=labels,
            )
            if partition_name == "test":
                report = classification_report(
                    partition["user"].astype(str),
                    predictions,
                    labels=list(labels),
                    output_dict=True,
                    zero_division=0,
                )
                matrices[model_name] = {
                    **metrics["confusion_matrix"],
                    "per_author": {
                        label: {
                            "precision": float(report[label]["precision"]),
                            "recall": float(report[label]["recall"]),
                            "f1": float(report[label]["f1-score"]),
                            "support": int(report[label]["support"]),
                        }
                        for label in labels
                    },
                }
                results.append({
                    "model": model_name,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "test_accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "macro_f1": metrics["macro_f1"],
                })
    return pd.DataFrame(results), matrices


def build_author_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Build author baselines with vectorization inside each fitted pipeline."""
    return {
        "TF-IDF + Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            )),
            ("classifier", LinearSVC(random_state=random_state)),
        ]),
        "TF-IDF + Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            )),
            ("classifier", MultinomialNB()),
        ]),
        "TF-IDF + Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }


@st.cache_resource
def load_sentence_encoder(model_name: str = AUTHOR_EMBEDDING_MODEL):
    """Load a multilingual Sentence-BERT encoder only when requested."""
    from model_runtime import load_transformer_encoder

    return load_transformer_encoder(model_name)


def _encode_sentences(texts: list[str], model_name: str) -> np.ndarray:
    import torch

    tokenizer, encoder, device = load_sentence_encoder(model_name)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            inputs = tokenizer(
                texts[start:start + 32],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            outputs = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy())
    return np.vstack(batches)


def evaluate_transformer_authorship(
    frame: pd.DataFrame,
    model_name: str = AUTHOR_EMBEDDING_MODEL,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate multilingual Sentence-BERT representations and style features."""
    frame = frame.copy()
    for column in ("message", "user", "conversation_id"):
        if column in frame:
            frame[column] = frame[column].fillna("").astype(str).str.strip()
    if "user" in frame:
        frame = frame[frame["user"] != "group_notification"].copy()
    required = ("message", "user", "conversation_id")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if (frame[list(required)] == "").any().any():
        raise ValueError("Messages, authors, and conversation IDs cannot be blank.")
    if frame["user"].nunique() < 2:
        raise ValueError("At least two distinct authors are required.")
    if frame["conversation_id"].nunique() < 3:
        raise ValueError("At least three conversations are required for held-out evaluation.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_validation_idx, test_idx = next(
        splitter.split(frame, groups=frame["conversation_id"].astype(str))
    )
    train_validation = frame.iloc[train_validation_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=random_state)
    train_idx, validation_idx = next(
        splitter.split(
            train_validation,
            groups=train_validation["conversation_id"].astype(str),
        )
    )
    train = train_validation.iloc[train_idx]
    validation = train_validation.iloc[validation_idx]
    test = frame.iloc[test_idx]
    labels = tuple(sorted(frame["user"].astype(str).unique()))
    if set(train["user"].astype(str)) != set(labels):
        raise ValueError("Every author must appear in the training conversations.")

    train_embeddings = _encode_sentences(train["message"].astype(str).tolist(), model_name)
    validation_embeddings = _encode_sentences(
        validation["message"].astype(str).tolist(), model_name
    )
    test_embeddings = _encode_sentences(test["message"].astype(str).tolist(), model_name)
    train_styles = np.asarray([
        list(extract_style_features(message).values())
        for message in train["message"].astype(str)
    ])
    validation_styles = np.asarray([
        list(extract_style_features(message).values())
        for message in validation["message"].astype(str)
    ])
    test_styles = np.asarray([
        list(extract_style_features(message).values())
        for message in test["message"].astype(str)
    ])
    style_scaler = StandardScaler().fit(train_styles)
    style_sets = (
        style_scaler.transform(train_styles),
        style_scaler.transform(validation_styles),
        style_scaler.transform(test_styles),
    )
    features = {
        "Sentence-BERT embeddings": (
            train_embeddings,
            validation_embeddings,
            test_embeddings,
        ),
        "Sentence-BERT + writing-style features": tuple(
            np.hstack((embeddings, styles))
            for embeddings, styles in zip(
                (train_embeddings, validation_embeddings, test_embeddings),
                style_sets,
            )
        ),
    }

    results = []
    matrices = {}
    for model_label, (x_train, x_validation, x_test) in features.items():
        classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        classifier.fit(x_train, train["user"].astype(str))
        validation_prediction = classifier.predict(x_validation)
        test_prediction = classifier.predict(x_test)
        validation_metrics = score_predictions(
            validation["user"].astype(str),
            validation_prediction,
            labels=labels,
        )
        test_metrics = score_predictions(
            test["user"].astype(str),
            test_prediction,
            labels=labels,
        )
        report = classification_report(
            test["user"].astype(str),
            test_prediction,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        )
        matrices[model_label] = {
            **test_metrics["confusion_matrix"],
            "per_author": {
                label: {
                    "precision": float(report[label]["precision"]),
                    "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]),
                    "support": int(report[label]["support"]),
                }
                for label in labels
            },
        }
        results.append({
            "model": model_label,
            "validation_accuracy": validation_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "macro_f1": test_metrics["macro_f1"],
        })
    return pd.DataFrame(results), matrices


def export_results(results: pd.DataFrame, file_format: str) -> bytes:
    """Serialize experiment results as CSV or JSON."""
    normalized_format = file_format.lower()
    if normalized_format == "csv":
        return results.to_csv(index=False).encode("utf-8")
    if normalized_format == "json":
        return results.to_json(orient="records", indent=2).encode("utf-8")
    raise ValueError("Export format must be csv or json.")


def evaluate_summary_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Score manually supplied reference/candidate summaries without inventing labels."""
    required = ("model", "reference_summary", "candidate_summary")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The summary evaluation dataset is empty.")
    data = frame.copy()
    for column in required:
        data[column] = data[column].fillna("").astype(str).str.strip()
    if (data[list(required)] == "").any().any():
        raise ValueError("Model names and both summary texts must be non-empty.")

    data["rouge_1"] = [
        _rouge_n(reference, candidate, 1)
        for reference, candidate in zip(data["reference_summary"], data["candidate_summary"])
    ]
    data["rouge_2"] = [
        _rouge_n(reference, candidate, 2)
        for reference, candidate in zip(data["reference_summary"], data["candidate_summary"])
    ]
    data["rouge_l"] = [
        _rouge_l(reference, candidate)
        for reference, candidate in zip(data["reference_summary"], data["candidate_summary"])
    ]

    possible_human_columns = [
        column for column in (
            "human_relevance",
            "human_factuality",
            "human_coverage",
        )
        if column in data.columns
    ]
    human_columns = []
    for column in possible_human_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        observed = data[column].dropna()
        if not observed.between(1, 5).all():
            raise ValueError(f"{column} scores must be on a 1 to 5 scale.")
        if not observed.empty:
            human_columns.append(column)

    aggregate_columns = ["rouge_1", "rouge_2", "rouge_l", *human_columns]
    return data.groupby("model", as_index=False)[aggregate_columns].mean()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)


def _rouge_n(reference: str, candidate: str, n: int) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    reference_ngrams = Counter(
        tuple(reference_tokens[index:index + n])
        for index in range(max(0, len(reference_tokens) - n + 1))
    )
    candidate_ngrams = Counter(
        tuple(candidate_tokens[index:index + n])
        for index in range(max(0, len(candidate_tokens) - n + 1))
    )
    overlap = sum((reference_ngrams & candidate_ngrams).values())
    return _f1(
        overlap / sum(reference_ngrams.values()) if reference_ngrams else 0.0,
        overlap / sum(candidate_ngrams.values()) if candidate_ngrams else 0.0,
    )


def _rouge_l(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    if len(candidate_tokens) > len(reference_tokens):
        reference_tokens, candidate_tokens = candidate_tokens, reference_tokens

    previous = [0] * (len(candidate_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, candidate_token in enumerate(candidate_tokens, 1):
            if reference_token == candidate_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    overlap = previous[-1]
    return _f1(
        overlap / len(reference_tokens),
        overlap / len(candidate_tokens),
    )


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def extract_style_features(message: str) -> dict[str, float]:
    """Extract interpretable, non-identifying writing-style features."""
    text = str(message)
    words = re.findall(r"\b[\w']+\b", text.lower(), flags=re.UNICODE)
    return {
        "character_count": float(len(text)),
        "word_count": float(len(words)),
        "average_word_length": (
            float(np.mean([len(word) for word in words])) if words else 0.0
        ),
        "type_token_ratio": float(len(set(words)) / len(words)) if words else 0.0,
        "question_marks": float(text.count("?")),
        "exclamation_marks": float(text.count("!")),
        "ellipsis_count": float(text.count("...")),
        "uppercase_ratio": (
            sum(character.isupper() for character in text) / len(text) if text else 0.0
        ),
        "digit_count": float(sum(character.isdigit() for character in text)),
        "emoji_count": float(len(emoji.emoji_list(text))),
        "non_ascii_character_count": float(
            sum(ord(character) > 127 for character in text)
        ),
    }


def json_safe(value: Any) -> Any:
    """Convert common NumPy and pandas values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value
