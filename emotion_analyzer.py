"""Optional local emotion recognition for English messages."""

import pandas as pd
import streamlit as st
import torch

from model_runtime import load_transformer_pipeline


EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"


@st.cache_resource
def load_emotion_pipeline(model_name=EMOTION_MODEL):
    return load_transformer_pipeline(
        "text-classification",
        model_name,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None,
    )


def analyze_emotions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return each text message's highest-scoring emotion and its model score."""
    required = ("message", "user")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Conversation data is missing columns: {', '.join(missing)}")

    messages = frame[
        (frame["message"].astype(str).str.strip() != "")
        & (frame["message"] != "<Media omitted>")
    ].copy()
    if messages.empty:
        return pd.DataFrame(
            columns=["message_id", "date_time", "user", "message", "emotion", "confidence"]
        )

    classifier = load_emotion_pipeline()
    predictions = classifier(messages["message"].astype(str).tolist(), truncation=True)
    if predictions and isinstance(predictions[0], dict):
        predictions = (
            [[prediction] for prediction in predictions]
            if len(predictions) == len(messages)
            else [predictions]
        )
    top_predictions = [
        max(message_predictions, key=lambda prediction: prediction["score"])
        for message_predictions in predictions
    ]
    messages["emotion"] = [
        str(prediction["label"]).lower() for prediction in top_predictions
    ]
    messages["confidence"] = [
        float(prediction["score"]) for prediction in top_predictions
    ]
    return messages
