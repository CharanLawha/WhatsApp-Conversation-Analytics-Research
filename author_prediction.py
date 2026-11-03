"""
Author Prediction Module - "Who Said It?"
Uses TF-IDF + Machine Learning to predict which user wrote a message.
Trains dynamically on uploaded chat data for fast predictions.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import streamlit as st


class AuthorPredictor:
    """
    Predicts the author of a message using TF-IDF and ensemble learning.
    Trains three models and selects the best performer.
    """
    
    def __init__(self):
        self.vectorizer = None
        self.best_model = None
        self.model_name = None
        self.users = None
        self.is_trained = False
    
    def train(self, df, selected_user=None):
        """
        Train the author prediction model on the chat data.
        
        Args:
            df: DataFrame with 'user' and 'message' columns
            selected_user: If provided, train on specific user's messages only
        
        Returns:
            dict: Training metrics including model name and accuracy
        """
        # Filter data
        valid_messages = df[
            (df['message'] != '<Media omitted>')
            & (df['user'] != 'group_notification')
            & (df['message'].astype(str).str.strip() != '')
        ]
        messages = valid_messages['message'].astype(str).tolist()
        authors = valid_messages['user'].astype(str).tolist()
        
        if len(messages) < 10:
            return {"error": "Not enough messages to train (minimum 10 required)"}
        
        self.users = sorted(set(authors))
        
        # If only 1 user, model can't predict
        if len(self.users) < 2:
            return {"error": "Need at least 2 different users for prediction"}
        
        y = authors
        class_counts = pd.Series(y).value_counts()
        folds = min(5, int(class_counts.min()))
        if folds < 2:
            return {
                "error": (
                    "Each participant needs at least two messages for stratified "
                    "cross-validation."
                )
            }
        
        # Train three models
        models = {
            "Naive Bayes": MultinomialNB(),
            "Linear SVM": LinearSVC(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        for model_name, model in models.items():
            estimator = Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=500,
                    lowercase=True,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1
                )),
                ("classifier", model),
            ])
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            try:
                scores = cross_val_score(
                    estimator,
                    messages,
                    y,
                    cv=cv,
                    scoring="accuracy",
                    error_score="raise",
                )
            except ValueError as error:
                return {
                    "error": (
                        f"Could not evaluate {model_name}; check that messages contain "
                        f"usable text features. {error}"
                    )
                }
            results[model_name] = scores.mean()

        # Select best model
        self.model_name = max(results, key=results.get)
        self.best_model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=500,
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )),
            ("classifier", models[self.model_name]),
        ])
        self.best_model.fit(messages, y)
        self.vectorizer = self.best_model.named_steps["tfidf"]
        
        self.is_trained = True
        
        return {
            "success": True,
            "best_model": self.model_name,
            "accuracy": f"{results[self.model_name]:.2%}",
            "all_accuracies": {k: f"{v:.2%}" for k, v in results.items()}
        }
    
    def predict(self, message):
        """
        Predict the author of a message.
        
        Args:
            message: str - The message to predict author for
        
        Returns:
            tuple: (predicted_author, confidence_score)
        """
        if not self.is_trained or self.best_model is None:
            return None, 0.0
        
        prediction = self.best_model.predict([message])[0]
        
        # Get confidence (probability for some models)
        confidence = 0.5  # Default
        classifier = self.best_model.named_steps["classifier"]
        if hasattr(classifier, 'predict_proba'):
            probabilities = self.best_model.predict_proba([message])[0]
            confidence = max(probabilities)
        
        return prediction, confidence
    
    def predict_batch(self, messages_list):
        """
        Predict authors for multiple messages.
        
        Args:
            messages_list: list of str - Messages to predict authors for
        
        Returns:
            list of tuples: [(author, confidence), ...]
        """
        if not self.is_trained:
            return [(None, 0.0) for _ in messages_list]
        
        results = []
        for message in messages_list:
            author, confidence = self.predict(message)
            results.append((author, confidence))
        
        return results
    
    def get_top_predictions(self, df, top_n=10):
        """
        Get predictions for random messages and display top confident ones.
        
        Args:
            df: DataFrame with messages
            top_n: int - Number of top predictions to return
        
        Returns:
            DataFrame: Predictions with confidence scores
        """
        if not self.is_trained:
            return None
        
        # Sample messages (excluding media)
        valid_messages = df[
            (df['message'] != '<Media omitted>')
            & (df['user'] != 'group_notification')
        ].copy()
        
        if len(valid_messages) == 0:
            return None
        
        sample_df = valid_messages.sample(n=min(50, len(valid_messages)), random_state=42)
        
        predictions = []
        for _, row in sample_df.iterrows():
            author, confidence = self.predict(row['message'])
            actual = row['user']
            is_correct = author == actual
            predictions.append({
                'Message': row['message'][:50] + '...' if len(row['message']) > 50 else row['message'],
                'Actual Author': actual,
                'Predicted Author': author,
                'Confidence': f"{confidence:.2%}",
                'Correct': '✓' if is_correct else '✗'
            })
        
        result_df = pd.DataFrame(predictions)
        
        # Sort by confidence descending
        result_df['Conf_Score'] = result_df['Confidence'].str.rstrip('%').astype(float)
        result_df = result_df.sort_values('Conf_Score', ascending=False).head(top_n)
        
        return result_df[['Message', 'Actual Author', 'Predicted Author', 'Confidence', 'Correct']]
    
    def get_stats(self):
        """Get training statistics."""
        return {
            'is_trained': self.is_trained,
            'model': self.model_name,
            'num_users': len(self.users) if self.users else 0,
            'users': self.users if self.users else []
        }
