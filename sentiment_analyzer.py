"""
High-Performance Sentiment Analysis Module
Uses RoBERTa trained on Twitter/social media data for better sarcasm 
and modern slang detection compared to VADER.
"""

import pandas as pd
import streamlit as st
import torch
import model_runtime


XLM_ROBERTA_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


@st.cache_resource
def load_sentiment_pipeline(model_name):
    """Load an explicitly selected local Hugging Face model on demand."""
    return model_runtime.load_transformer_pipeline(
        "sentiment-analysis",
        model_name,
        device=0 if torch.cuda.is_available() else -1
    )


def normalize_sentiment_label(label):
    """Normalize common model label formats to the app's three classes."""
    normalized = str(label).strip().lower()
    label_map = {
        "positive": "POSITIVE",
        "pos": "POSITIVE",
        "label_2": "POSITIVE",
        "negative": "NEGATIVE",
        "neg": "NEGATIVE",
        "label_0": "NEGATIVE",
        "neutral": "NEUTRAL",
        "label_1": "NEUTRAL",
    }
    return label_map.get(normalized, str(label).upper())


class RoBERTaSentimentAnalyzer:
    """
    High-performance sentiment analyzer using RoBERTa model 
    trained on Twitter/social media data.
    """
    
    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
        self.load_error = None
    
    def load_model(self):
        """Load the RoBERTa sentiment model (twitter-roberta-base-sentiment)."""
        if self.is_loaded:
            return True
        
        try:
            # Using Twitter-trained RoBERTa model which handles slang and sarcasm better
            self.pipeline = model_runtime.load_transformer_pipeline(
                "sentiment-analysis",
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            self.is_loaded = True
            return True
        except Exception as e:
            self.load_error = str(e)
            return False
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of a single text.
        
        Args:
            text: str - Text to analyze
        
        Returns:
            dict: {sentiment: str, score: float, label: str}
        """
        if not self.is_loaded:
            if not self.load_model():
                return {"sentiment": "UNKNOWN", "score": 0.0, "label": "Model not loaded"}
        
        if not text or len(text.strip()) == 0:
            return {"sentiment": "NEUTRAL", "score": 0.0, "label": "Empty text"}
        
        try:
            result = self.pipeline(text[:512])[0]  # Limit to 512 chars for efficiency
            
            # Map model outputs to more intuitive labels
            label = result['label']
            score = result['score']
            
            # Normalize labels from model
            sentiment_map = {
                'Positive': 'POSITIVE',
                'Negative': 'NEGATIVE',
                'Neutral': 'NEUTRAL',
                'POS': 'POSITIVE',
                'NEG': 'NEGATIVE',
                'LABEL_0': 'NEGATIVE',
                'LABEL_1': 'NEUTRAL',
                'LABEL_2': 'POSITIVE'
            }
            
            sentiment = sentiment_map.get(label, label.upper())
            
            return {
                "sentiment": sentiment,
                "score": score,
                "label": label
            }
        except Exception as e:
            return {"sentiment": "ERROR", "score": 0.0, "label": str(e)}
    
    def analyze_batch(self, messages_list):
        """
        Analyze sentiment for multiple messages.
        
        Args:
            messages_list: list of str - Messages to analyze
        
        Returns:
            list of dicts: Sentiment analysis for each message
        """
        if not self.is_loaded:
            if not self.load_model():
                return [{"sentiment": "UNKNOWN", "score": 0.0} for _ in messages_list]
        
        results = []
        for message in messages_list:
            result = self.analyze_sentiment(message)
            results.append(result)
        
        return results
    
    def analyze_user_sentiment(self, df, selected_user='Overall'):
        """
        Analyze sentiment distribution for a user.
        
        Args:
            df: DataFrame with 'user' and 'message' columns
            selected_user: str - User to analyze, or 'Overall'
        
        Returns:
            dict: Sentiment distribution and statistics
        """
        if not self.is_loaded and not self.load_model():
            return {"error": f"Could not load the sentiment model: {self.load_error}"}

        if selected_user != 'Overall':
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df
        
        # Filter out media messages
        messages = user_df[user_df['message'] != '<Media omitted>']['message'].tolist()
        
        if not messages:
            return {"error": "No messages to analyze"}
        
        # Analyze sentiments
        sentiments = []
        for message in messages:
            result = self.analyze_sentiment(message)
            sentiments.append(result['sentiment'])
        
        # Calculate distribution
        sentiment_counts = {}
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment_counts[sentiment] = sentiments.count(sentiment)
        
        total = len(sentiments)
        sentiment_percentages = {
            k: f"{(v/total)*100:.1f}%" for k, v in sentiment_counts.items()
        }
        
        return {
            "counts": sentiment_counts,
            "percentages": sentiment_percentages,
            "total_messages_analyzed": total,
            "positive_score": sentiment_counts['POSITIVE'] / total if total > 0 else 0
        }
    
    def get_sentiment_timeline(self, df, selected_user='Overall', period='D'):
        """
        Get sentiment trends over time.
        
        Args:
            df: DataFrame with 'user', 'message', 'datetime' columns
            selected_user: str - User to analyze
            period: str - 'D' for day, 'W' for week, 'M' for month
        
        Returns:
            DataFrame: Sentiment counts by time period
        """
        if selected_user != 'Overall':
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df
        
        # Filter valid messages
        user_df = user_df[user_df['message'] != '<Media omitted>'].copy()
        
        if not user_df.empty and 'datetime' in user_df.columns:
            user_df['date'] = pd.to_datetime(user_df['datetime']).dt.to_period(period)
            
            # Analyze each message
            sentiments = []
            for message in user_df['message'].tolist():
                result = self.analyze_sentiment(message)
                sentiments.append(result['sentiment'])
            
            user_df['sentiment'] = sentiments
            
            # Group by period
            timeline = user_df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            return timeline
        
        return None


class ContextAwareSentimentAnalyzer:
    """Run multilingual message-level sentiment with preceding conversation context."""

    def __init__(self, model_name=XLM_ROBERTA_SENTIMENT_MODEL):
        self.model_name = model_name
        self.pipeline = None

    def analyze(self, df, context_messages=3):
        if context_messages not in range(2, 6):
            raise ValueError("Context must include between 2 and 5 previous messages.")
        if "message" not in df or "user" not in df:
            raise ValueError("Conversation data must contain 'message' and 'user' columns.")

        model = self.pipeline or load_sentiment_pipeline(self.model_name)
        self.pipeline = model
        frame = df.reset_index(drop=True)
        participants = {
            participant: f"speaker_{index}"
            for index, participant in enumerate(frame["user"].astype(str).unique(), 1)
        }
        inputs = []
        valid_rows = []
        for index, row in frame.iterrows():
            message = str(row["message"])
            if not message.strip() or message == "<Media omitted>":
                continue
            start = max(0, index - context_messages)
            context = frame.iloc[start:index + 1]
            inputs.append("\n".join(
                f"[{participants.get(str(item['user']), 'speaker_unknown')}] "
                f"{str(item['message'])}"
                for _, item in context.iterrows()
                if str(item["message"]).strip()
                and str(item["message"]) != "<Media omitted>"
            ))
            valid_rows.append(index)

        if not inputs:
            return pd.DataFrame(
                columns=["message_id", "date_time", "user", "message", "sentiment", "confidence"]
            )

        predictions = model(inputs, truncation=True, batch_size=16)
        results = frame.iloc[valid_rows].copy()
        results["sentiment"] = [
            normalize_sentiment_label(prediction["label"])
            for prediction in predictions
        ]
        results["confidence"] = [
            float(prediction["score"]) for prediction in predictions
        ]
        return results


# Initialize and cache sentiment analyzer
@st.cache_resource
def get_sentiment_analyzer():
    """Cached sentiment analyzer instance."""
    return RoBERTaSentimentAnalyzer()
