import unittest
from unittest.mock import patch

import pandas as pd

import privacy
import research_tools
import author_prediction
import preprocessor
import chat_summarizer
import sentiment_analyzer


def sentiment_fixture():
    rows = []
    examples = {
        "positive": ("that was wonderful", "I feel great", "love this"),
        "negative": ("that was awful", "I feel terrible", "hate this"),
        "neutral": ("meeting starts at noon", "the file is ready", "see you later"),
    }
    for conversation in range(12):
        for label, messages in examples.items():
            rows.append({
                "conversation_id": f"conversation-{conversation}",
                "message": messages[conversation % len(messages)],
                "sentiment": label,
                "emotion": "",
            })
    return pd.DataFrame(rows)


class ResearchToolsTests(unittest.TestCase):
    def test_validation_reports_only_annotated_labels(self):
        data = sentiment_fixture()
        result = research_tools.validate_annotated_dataset(data)
        self.assertEqual(result["rows"], len(data))
        self.assertEqual(result["conversations"], 12)
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            research_tools.validate_annotated_dataset(
                data.assign(sentiment="fabricated")
            )

    def test_conversation_split_has_no_group_overlap(self):
        data = sentiment_fixture()
        train, validation, test = research_tools.conversation_train_validation_test_split(data)
        partitions = [
            set(part["conversation_id"]) for part in (train, validation, test)
        ]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])

    def test_baseline_metrics_are_numeric_and_exportable(self):
        data = sentiment_fixture()
        results, confusion_matrices = research_tools.evaluate_tfidf_sentiment_baselines(data)
        self.assertIn("VADER", confusion_matrices)
        self.assertTrue(results["accuracy"].between(0, 1).all())
        self.assertIn(b"macro_f1", research_tools.export_results(results, "csv"))
        self.assertIn(b"macro_f1", research_tools.export_results(results, "json"))

    def test_privacy_helpers_mask_contact_details_and_anonymize(self):
        text = privacy.mask_sensitive_text("Email me at alice@example.com or +1 (555) 123-4567")
        self.assertNotIn("alice@example.com", text)
        self.assertNotIn("555", text)
        sanitized = privacy.sanitize_conversation(pd.DataFrame({
            "user": ["Alice", "Bob"],
            "message": ["Call 5551234567 about Project Aurora", "ok"],
            "Message": ["Alice: Call 5551234567", "Bob: ok"],
        }), extra_terms=["Project Aurora"])
        self.assertEqual(sanitized["user"].tolist(), ["Participant 1", "Participant 2"])
        self.assertNotIn("5551234567", sanitized.loc[0, "message"])
        self.assertNotIn("Project Aurora", sanitized.loc[0, "message"])
        self.assertNotIn("Alice", sanitized.loc[0, "Message"])

    def test_authorship_baselines_use_conversation_holdouts(self):
        rows = []
        for conversation in range(12):
            rows.extend([
                {
                    "conversation_id": f"conversation-{conversation}",
                    "user": "Alex",
                    "message": f"Alex writing style example {conversation}",
                },
                {
                    "conversation_id": f"conversation-{conversation}",
                    "user": "Sam",
                    "message": f"Sam writing style example {conversation}",
                },
            ])
        scores, matrices = research_tools.evaluate_authorship_baselines(pd.DataFrame(rows))
        self.assertEqual(len(scores), 3)
        self.assertIn("TF-IDF + Linear SVM", matrices)
        self.assertIn("Alex", matrices["TF-IDF + Linear SVM"]["per_author"])

    def test_writing_style_features_include_emoji_and_lexical_signals(self):
        features = research_tools.extract_style_features("Great!!! 😊")
        self.assertEqual(features["emoji_count"], 1.0)
        self.assertEqual(features["exclamation_marks"], 3.0)
        self.assertGreaterEqual(features["type_token_ratio"], 0.5)

    def test_summary_metrics_require_reference_texts_and_report_human_ratings(self):
        scores = research_tools.evaluate_summary_dataset(pd.DataFrame([{
            "model": "BART",
            "reference_summary": "Meeting moved to Friday",
            "candidate_summary": "Meeting moved to Friday",
            "human_relevance": 4,
            "human_factuality": 5,
            "human_coverage": 4,
        }]))
        self.assertEqual(scores.loc[0, "rouge_1"], 1.0)
        self.assertEqual(scores.loc[0, "rouge_2"], 1.0)
        self.assertEqual(scores.loc[0, "rouge_l"], 1.0)
        self.assertEqual(scores.loc[0, "human_factuality"], 5.0)

    def test_event_candidates_keep_verbatim_message_provenance(self):
        frame = pd.DataFrame([{
            "message_id": "msg_000007",
            "date_time": "2025-01-02 12:00",
            "user": "Alex",
            "message": "We decided to meet Friday.",
        }])
        candidates = chat_summarizer.ChatSummarizer().get_event_candidates(frame)
        self.assertEqual(candidates[0]["message_id"], "msg_000007")
        self.assertEqual(candidates[0]["message"], "We decided to meet Friday.")

    def test_topic_groups_preserve_source_message_ids(self):
        frame = pd.DataFrame([
            {
                "message_id": f"msg_{index:06d}",
                "user": "Alex" if index % 2 else "Sam",
                "message": message,
            }
            for index, message in enumerate([
                "project meeting schedule plan",
                "project meeting discussion",
                "family dinner cooking recipe",
                "family dinner shopping list",
                "project deadline meeting plan",
                "family recipe dinner food",
            ])
        ])
        result = chat_summarizer.ChatSummarizer().topic_message_groups(
            frame, n_topics=2
        )
        self.assertIn("topics", result)
        source_ids = {
            message_id
            for topic in result["topics"]
            for message_id in topic["messages"]["message_id"]
        }
        self.assertEqual(source_ids, set(frame["message_id"]))

    def test_textrank_returns_verbatim_messages_with_references(self):
        frame = pd.DataFrame([
            {
                "message_id": f"msg_{index}",
                "user": "Alex",
                "message": message,
            }
            for index, message in enumerate([
                "Project meeting moved to Friday",
                "The project schedule is ready",
                "Dinner recipe includes rice",
            ])
        ])
        result = chat_summarizer.ChatSummarizer().textrank_summary(
            frame, num_messages=2
        )
        self.assertEqual(result["model"], "TextRank")
        self.assertEqual(len(result["supporting_messages"]), 2)
        self.assertTrue(all(item["message_id"] for item in result["supporting_messages"]))

    def test_transformer_summary_chunks_long_text_and_keeps_source_refs(self):
        class FakeTokenizer:
            model_max_length = 128

            def encode(self, text, add_special_tokens=False):
                return str(text).split()

            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(tokens)

        class FakeSummarizer:
            tokenizer = FakeTokenizer()

            def __call__(self, text, **kwargs):
                return [{"summary_text": "Generated summary"}]

        frame = pd.DataFrame([{
            "message_id": "msg_long",
            "date_time": "2025-01-02 12:00",
            "user": "Alex",
            "message": " ".join(f"word{index}" for index in range(140)),
        }])
        with patch(
            "chat_summarizer.load_summary_pipeline",
            return_value=FakeSummarizer(),
        ):
            result = chat_summarizer.ChatSummarizer().summarize_transformer(
                frame,
                model_name="facebook/bart-large-cnn",
            )
        self.assertTrue(result["hierarchical"])
        self.assertEqual(result["messages_summarized"], 1)
        self.assertEqual(result["supporting_messages"][0]["message_id"], "msg_long")

    def test_context_sentiment_includes_speakers_and_previous_messages(self):
        class FakePipeline:
            def __init__(self):
                self.inputs = None

            def __call__(self, inputs, **kwargs):
                self.inputs = inputs
                return [{"label": "positive", "score": 0.9} for _ in inputs]

        fake_pipeline = FakePipeline()
        frame = pd.DataFrame([
            {"message_id": f"msg_{index}", "user": user, "message": message}
            for index, (user, message) in enumerate([
                ("Alex", "not great"),
                ("Sam", "I understand"),
                ("Alex", "Thanks 😊"),
            ])
        ])
        with patch(
            "sentiment_analyzer.load_sentiment_pipeline",
            return_value=fake_pipeline,
        ):
            result = sentiment_analyzer.ContextAwareSentimentAnalyzer().analyze(
                frame, context_messages=2
            )
        self.assertIn("[speaker_1] not great", fake_pipeline.inputs[-1])
        self.assertIn("[speaker_2] I understand", fake_pipeline.inputs[-1])
        self.assertIn("Thanks 😊", fake_pipeline.inputs[-1])
        self.assertEqual(result["sentiment"].tolist(), ["POSITIVE"] * 3)

    def test_existing_author_predictor_trains_with_fold_local_tfidf(self):
        messages = [
            {"user": "Alex", "message": f"alex common phrase {index}"}
            for index in range(6)
        ] + [
            {"user": "Sam", "message": f"sam common phrase {index}"}
            for index in range(6)
        ]
        predictor = author_prediction.AuthorPredictor()
        result = predictor.train(pd.DataFrame(messages))
        self.assertTrue(result["success"])
        self.assertIn(predictor.predict("alex common phrase")[0], {"Alex", "Sam"})

    def test_parser_adds_stable_message_ids(self):
        frame = preprocessor.preprocessor(
            "01/01/2025, 10:00 am - Alice: Hello\n"
            "01/01/2025, 10:01 am - Bob: Hi"
        )
        self.assertEqual(frame["message_id"].tolist(), ["msg_000001", "msg_000002"])


if __name__ == "__main__":
    unittest.main()
