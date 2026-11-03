"""
Chat Summarization Module
Generates concise summaries of conversations using extractive methods.
Provides both extractive (key sentences) and simple abstractive summaries.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


SUMMARY_MODELS = {
    "BART": ("facebook/bart-large-cnn", "summarization"),
    "FLAN-T5": ("google/flan-t5-base", "text2text-generation"),
    "Qwen 2.5 0.5B (local)": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "text-generation",
    ),
}


@st.cache_resource
def load_summary_pipeline(model_name, task):
    """Download and cache a selected summarization model only when requested."""
    from model_runtime import load_transformer_pipeline

    return load_transformer_pipeline(task, model_name)


class ChatSummarizer:
    """
    Summarizes WhatsApp chat conversations using extractive methods.
    Provides both extractive (key sentences) and simple abstractive summaries.
    """

    def __init__(self):
        self.is_loaded = True  # No complex model to load

    def load_model(self):
        """No model to load for extractive summarization."""
        return True

    def prepare_text_for_summarization(self, messages_list, max_length=1024):
        """
        Prepare messages for summarization by joining them.

        Args:
            messages_list: list of str - Messages to summarize
            max_length: int - Maximum tokens to use

        Returns:
            str: Prepared text
        """
        # Join messages with context
        text = " ".join(messages_list)

        # Remove special characters but keep context
        text = text.replace('<Media omitted>', '[media]')
        text = text.replace('\n', ' ')

        # Truncate if too long
        words = text.split()
        if len(words) > max_length:
            text = " ".join(words[:max_length])

        return text

    def summarize(self, text, max_summary_length=250, min_summary_length=150):
        """
        Generate a simple extractive summary of text.

        Args:
            text: str - Text to summarize
            max_summary_length: int - Maximum length of summary
            min_summary_length: int - Minimum length of summary

        Returns:
            str: Summary
        """
        if len(text.split()) < 20:  # Too short to summarize
            return text

        try:
            # Simple extractive summarization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 3:
                return text

            # Score sentences based on word frequency
            words = text.lower().split()
            word_freq = Counter(words)

            # Remove stop words (simple approach)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'}

            # Score sentences
            sentence_scores = {}
            for sentence in sentences:
                sentence_words = sentence.lower().split()
                score = sum(word_freq[word] for word in sentence_words if word not in stop_words)
                sentence_scores[sentence] = score / len(sentence_words) if sentence_words else 0

            # Get top sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_sentences = [sentence for sentence, score in top_sentences]

            # Join and ensure length constraints
            summary = '. '.join(summary_sentences) + '.'

            # Truncate if too long
            words = summary.split()
            if len(words) > max_summary_length:
                summary = ' '.join(words[:max_summary_length]) + '...'

            return summary

        except Exception as e:
            return f"Error: {str(e)}"

    def summarize_user_messages(self, df, selected_user='Overall', num_messages=None):
        """
        Summarize messages from a specific user or overall chat.

        Args:
            df: DataFrame with 'user' and 'message' columns
            selected_user: str - User to summarize, or 'Overall'
            num_messages: int - Number of recent messages to consider (None = all)

        Returns:
            dict: Summary and metadata
        """
        # Filter by user
        if selected_user != 'Overall':
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df

        # Get valid messages
        messages = user_df[user_df['message'] != '<Media omitted>']['message'].tolist()

        if not messages:
            return {"error": "No messages to summarize"}

        # Limit to recent messages if specified
        if num_messages:
            messages = messages[-num_messages:]

        # Prepare text
        text = self.prepare_text_for_summarization(messages)

        # Generate summary
        summary = self.summarize(text)

        return {
            "summary": summary,
            "user": selected_user,
            "messages_summarized": len(messages),
            "original_length": len(text.split()),
            "compression_ratio": f"{(1 - len(summary.split())/len(text.split()))*100:.1f}%" if len(text.split()) > 0 else "0%"
        }

    def summarize_conversation_topic(self, df, selected_user='Overall'):
        """
        Generate a high-level summary of conversation topics.

        Args:
            df: DataFrame with 'user', 'message' columns
            selected_user: str - User to analyze

        Returns:
            dict: Topic summary
        """
        if selected_user != 'Overall':
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df

        messages = user_df[user_df['message'] != '<Media omitted>']['message'].tolist()

        if not messages:
            return {"error": "No messages"}

        text = self.prepare_text_for_summarization(messages)
        summary = self.summarize(text, max_summary_length=150, min_summary_length=50)

        return {
            "topic_summary": summary,
            "user": selected_user,
            "total_messages": len(messages)
        }

    def topic_model(self, df, selected_user='Overall', n_topics=3, n_top_words=5):
        """
        Perform topic modeling on chat messages.

        Args:
            df: DataFrame with 'user' and 'message' columns
            selected_user: str - User to analyze or 'Overall'
            n_topics: int - Number of topics to extract
            n_top_words: int - Number of top keywords per topic

        Returns:
            dict: Topics and distribution data
        """
        if selected_user != 'Overall':
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df

        messages = user_df[user_df['message'] != '<Media omitted>']['message'].tolist()
        if not messages:
            return {"error": "No messages available for topic modeling"}

        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,
            stop_words='english'
        )
        dtm = vectorizer.fit_transform(messages)
        if dtm.shape[1] == 0:
            return {"error": "Not enough vocabulary for topic modeling"}

        n_components = min(n_topics, dtm.shape[0])
        if n_components <= 0:
            return {"error": "Not enough documents for topic modeling"}

        lda = LatentDirichletAllocation(
            n_components=n_components,
            random_state=42,
            learning_method='batch'
        )
        lda.fit(dtm)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append({
                'topic_id': topic_idx + 1,
                'keywords': top_features,
                'keywords_text': ', '.join(top_features)
            })

        doc_topic_dist = lda.transform(dtm)
        dominant_topics = doc_topic_dist.argmax(axis=1)
        topic_counts = np.bincount(dominant_topics, minlength=len(topics))
        distribution = [int(count) for count in topic_counts]

        return {
            'topics': topics,
            'distribution': distribution,
            'num_topics': len(topics),
            'total_documents': len(messages),
            'selected_user': selected_user
        }

    def topic_message_groups(self, df, selected_user='Overall', n_topics=4, n_top_words=6):
        """Group original messages by an LDA topic for source-linked summaries."""
        if n_topics < 1:
            raise ValueError("The number of topics must be positive.")
        messages = df.copy()
        if selected_user != 'Overall':
            messages = messages[messages['user'] == selected_user]
        messages = messages[
            (messages['message'] != '<Media omitted>')
            & (messages['user'] != 'group_notification')
            & (messages['message'].astype(str).str.strip() != '')
        ].copy()
        if messages.empty:
            return {"error": "No messages available for topic summaries."}

        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=1,
            stop_words='english',
        )
        try:
            document_term_matrix = vectorizer.fit_transform(
                messages['message'].astype(str)
            )
        except ValueError as error:
            return {"error": f"Could not identify topics: {error}"}
        component_count = min(n_topics, document_term_matrix.shape[0])
        lda = LatentDirichletAllocation(
            n_components=component_count,
            random_state=42,
            learning_method='batch',
        )
        lda.fit(document_term_matrix)
        feature_names = vectorizer.get_feature_names_out()
        dominant_topics = lda.transform(document_term_matrix).argmax(axis=1)
        topics = []
        for topic_index, component in enumerate(lda.components_):
            keywords = [
                feature_names[index]
                for index in component.argsort()[-n_top_words:][::-1]
            ]
            topic_messages = messages.iloc[
                np.flatnonzero(dominant_topics == topic_index)
            ].copy()
            if topic_messages.empty:
                continue
            topics.append({
                "topic_id": topic_index + 1,
                "keywords": keywords,
                "messages": topic_messages,
            })
        return {"topics": topics, "total_messages": len(messages)}

    def get_event_candidates(self, df, selected_user='Overall', limit=20):
        """Return verbatim messages with decision/event cue words for human review."""
        cue_pattern = re.compile(
            r"\b(decid\w*|agreed|agree|plan\w*|deadline|meeting|appointment|"
            r"confirm\w*|cancel\w*|complete\w*|submit\w*|due|paid|sent|"
            r"done|pakka|milte|final)\b|"
            r"(فیصلہ|منصوبہ|میٹنگ|بھیج|ہو گیا|ہوگیا)",
            flags=re.IGNORECASE,
        )
        messages = df.copy()
        if selected_user != 'Overall':
            messages = messages[messages['user'] == selected_user]
        candidates = []
        for _, row in messages.iterrows():
            message = str(row.get("message", ""))
            matches = sorted({match.group(0) for match in cue_pattern.finditer(message)})
            if matches:
                message_id = row.get("message_id")
                timestamp = row.get("date_time")
                candidates.append({
                    "message_id": message_id if pd.notna(message_id) else None,
                    "timestamp": str(timestamp) if pd.notna(timestamp) else None,
                    "participant": str(row.get("user", "")),
                    "cue_words": ", ".join(matches),
                    "message": message,
                })
        return candidates[:limit]

    def textrank_summary(self, df, selected_user='Overall', num_messages=5):
        """Create an extractive TextRank summary with verbatim source references."""
        if num_messages < 1:
            raise ValueError("TextRank must return at least one source message.")
        messages = df.copy()
        if selected_user != 'Overall':
            messages = messages[messages['user'] == selected_user]
        messages = messages[
            (messages['message'] != '<Media omitted>')
            & (messages['user'] != 'group_notification')
            & (messages['message'].astype(str).str.strip() != '')
        ].copy()
        if messages.empty:
            return {"error": "No messages are available for TextRank summarization."}
        if len(messages) > 2000:
            return {
                "error": (
                    "TextRank is limited to 2,000 messages per run. Narrow the date "
                    "range or participant filter and try again."
                )
            }

        try:
            matrix = TfidfVectorizer(
                lowercase=True,
                max_features=10000,
                ngram_range=(1, 2),
            ).fit_transform(messages['message'].astype(str))
        except ValueError as error:
            return {"error": f"Could not rank message text: {error}"}

        similarities = cosine_similarity(matrix, dense_output=True)
        similarities[np.arange(len(messages)), np.arange(len(messages))] = 0.0
        row_sums = similarities.sum(axis=1)
        transition = np.divide(
            similarities,
            row_sums[:, None],
            out=np.zeros_like(similarities),
            where=row_sums[:, None] != 0,
        )
        scores = np.full(len(messages), 1 / len(messages), dtype=float)
        for _ in range(100):
            updated_scores = 0.15 / len(messages) + 0.85 * transition.T.dot(scores)
            if np.linalg.norm(updated_scores - scores, ord=1) < 1e-8:
                scores = updated_scores
                break
            scores = updated_scores

        selected_indices = np.argsort(scores)[-min(num_messages, len(messages)):]
        selected_messages = messages.iloc[np.sort(selected_indices)]
        evidence = []
        for _, row in selected_messages.iterrows():
            message_id = row.get("message_id")
            timestamp = row.get("date_time")
            evidence.append({
                "message_id": message_id if pd.notna(message_id) else None,
                "timestamp": str(timestamp) if pd.notna(timestamp) else None,
                "participant": str(row["user"]),
                "message": str(row["message"]),
            })
        return {
            "summary": "\n".join(
                f"{item['participant']}: {item['message']}" for item in evidence
            ),
            "model": "TextRank",
            "messages_summarized": len(messages),
            "supporting_messages": evidence,
        }

    def get_key_sentences(self, df, selected_user='Overall', num_sentences=5):
        """
        Extract key sentences (extractive summarization).

        Args:
            df: DataFrame with messages
            selected_user: str - User to analyze
            num_sentences: int - Number of key sentences

        Returns:
            list: Key messages from the conversation
        """
        if selected_user != 'Overall':
            user_df = df[df['user'] == selected_user]
        else:
            user_df = df

        # Get messages with sentiment/importance scores
        messages_df = user_df[user_df['message'] != '<Media omitted>'].copy()

        if len(messages_df) == 0:
            return []

        # Simple importance scoring based on message length and frequency
        messages_df['length'] = messages_df['message'].str.len()
        messages_df['importance'] = (
            messages_df['length'].rank() / len(messages_df) * 0.5 +
            (pd.Series(range(len(messages_df))) / len(messages_df)) * 0.5
        )

        # Get top N messages
        source_columns = [
            column for column in ('message_id', 'date_time', 'user', 'message')
            if column in messages_df.columns
        ]
        key_messages = messages_df.nlargest(num_sentences, 'importance')[source_columns]

        results = []
        for _, row in key_messages.iterrows():
            message_id = row.get("message_id")
            results.append({
                "message_id": message_id if pd.notna(message_id) else None,
                "timestamp": str(row.get("date_time")) if pd.notna(row.get("date_time")) else None,
                "user": row["user"],
                "message": row["message"],
            })
        return results

    def summarize_time_period(self, df, start_date=None, end_date=None):
        """
        Summarize chat from a specific time period.

        Args:
            df: DataFrame with 'datetime' and 'message' columns
            start_date: datetime - Start of period
            end_date: datetime - End of period

        Returns:
            dict: Summary for time period
        """
        period_df = df.copy()

        if 'datetime' in period_df.columns:
            period_df['datetime'] = pd.to_datetime(period_df['datetime'])

            if start_date:
                period_df = period_df[period_df['datetime'] >= start_date]
            if end_date:
                period_df = period_df[period_df['datetime'] <= end_date]

        messages = period_df[period_df['message'] != '<Media omitted>']['message'].tolist()

        if not messages:
            return {"error": "No messages in this period"}

        text = self.prepare_text_for_summarization(messages)
        summary = self.summarize(text)

        return {
            "summary": summary,
            "messages_in_period": len(messages),
            "period": f"{start_date} to {end_date}" if start_date and end_date else "All time"
        }

    def summarize_transformer(
        self,
        df,
        selected_user='Overall',
        model_name="facebook/bart-large-cnn",
        start_date=None,
        end_date=None,
        chunk_word_limit=600,
    ):
        """Create a hierarchical local-model summary with source-message references."""
        if model_name not in {model_id for model_id, _ in SUMMARY_MODELS.values()}:
            raise ValueError("Choose one of the supported local BART or FLAN-T5 models.")
        if chunk_word_limit < 100:
            raise ValueError("Chunk size must be at least 100 words.")

        messages = df.copy()
        if selected_user != 'Overall':
            messages = messages[messages['user'] == selected_user]
        if 'date_time' in messages.columns:
            messages['date_time'] = pd.to_datetime(messages['date_time'], errors='coerce')
            if start_date is not None:
                messages = messages[messages['date_time'].dt.date >= pd.Timestamp(start_date).date()]
            if end_date is not None:
                messages = messages[messages['date_time'].dt.date <= pd.Timestamp(end_date).date()]

        messages = messages[
            (messages['message'].astype(str).str.strip() != '')
            & (messages['message'] != '<Media omitted>')
            & (messages['user'] != 'group_notification')
        ]
        if messages.empty:
            return {"error": "No text messages are available in the selected scope."}

        model_task = next(
            task for model_id, task in SUMMARY_MODELS.values() if model_id == model_name
        )
        summarizer = load_summary_pipeline(model_name, model_task)
        tokenizer = summarizer.tokenizer
        model_token_limit = getattr(tokenizer, "model_max_length", 1024)
        if model_token_limit > 100000:
            model_token_limit = 1024
        max_input_tokens = max(64, min(900, model_token_limit - 32))
        units = []
        for _, row in messages.iterrows():
            message = str(row['message'])
            speaker_prefix = f"{row['user']}: "
            full_text = speaker_prefix + message
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(token_ids) <= max_input_tokens:
                units.append((
                    row,
                    full_text,
                    len(token_ids),
                    len(message.split()),
                ))
                continue

            content_ids = tokenizer.encode(message, add_special_tokens=False)
            prefix_tokens = len(
                tokenizer.encode(speaker_prefix, add_special_tokens=False)
            )
            segment_size = max(1, max_input_tokens - prefix_tokens)
            for start in range(0, len(content_ids), segment_size):
                segment = tokenizer.decode(
                    content_ids[start:start + segment_size],
                    skip_special_tokens=True,
                )
                segment_text = speaker_prefix + segment
                segment_tokens = len(
                    tokenizer.encode(segment_text, add_special_tokens=False)
                )
                units.append((
                    row,
                    segment_text,
                    segment_tokens,
                    len(segment.split()),
                ))

        chunks = []
        current_chunk = []
        current_tokens = 0
        current_words = 0
        for unit in units:
            if current_chunk and (
                current_tokens + unit[2] > max_input_tokens
                or current_words + unit[3] > chunk_word_limit
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
                current_words = 0
            current_chunk.append(unit)
            current_tokens += unit[2]
            current_words += unit[3]
        if current_chunk:
            chunks.append(current_chunk)

        def format_model_input(source_text):
            if model_task == "text2text-generation":
                return f"Summarize this conversation accurately: {source_text}"
            if model_task == "text-generation":
                return (
                    "<|im_start|>system\nSummarize the supplied conversation accurately. "
                    "Do not add unsupported details.<|im_end|>\n"
                    f"<|im_start|>user\n{source_text}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            return source_text

        def summarize_text(source_text):
            model_input = format_model_input(source_text)
            if model_task == "text-generation":
                output = summarizer(
                    model_input,
                    max_new_tokens=160,
                    truncation=False,
                    do_sample=False,
                    return_full_text=False,
                )[0]
            else:
                output = summarizer(
                    model_input,
                    max_length=160,
                    truncation=False,
                    do_sample=False,
                )[0]
            return output.get(
                "summary_text", output.get("generated_text", "")
            ).strip()

        chunk_summaries = []
        evidence = []
        evidence_ids = set()
        for chunk in chunks:
            source_lines = [text for _, text, _, _ in chunk]
            source_text = "\n".join(source_lines)
            chunk_summaries.append(summarize_text(source_text))
            for row, _, _, _ in chunk:
                message_id = row.get("message_id")
                evidence_key = (
                    str(message_id) if pd.notna(message_id) else str(row.name)
                )
                if evidence_key in evidence_ids:
                    continue
                evidence_ids.add(evidence_key)
                evidence.append({
                    "message_id": message_id if pd.notna(message_id) else None,
                    "timestamp": (
                        str(row.get("date_time"))
                        if pd.notna(row.get("date_time"))
                        else None
                    ),
                    "participant": str(row["user"]),
                    "message": str(row["message"]),
                })

        summary_levels = chunk_summaries
        while len(summary_levels) > 1:
            grouped_summaries = []
            group = []
            group_tokens = 0
            group_words = 0
            for item in summary_levels:
                token_count = len(
                    tokenizer.encode(
                        format_model_input(item),
                        add_special_tokens=True,
                    )
                )
                word_count = len(item.split())
                if group and (
                    group_tokens + token_count > max_input_tokens
                    or group_words + word_count > chunk_word_limit
                ):
                    grouped_summaries.append(summarize_text("\n".join(group)))
                    group = []
                    group_tokens = 0
                    group_words = 0
                if token_count > max_input_tokens:
                    raise RuntimeError(
                        "A generated summary exceeds the model's safe input limit."
                    )
                group.append(item)
                group_tokens += token_count
                group_words += word_count
            if group:
                grouped_summaries.append(summarize_text("\n".join(group)))
            if len(grouped_summaries) >= len(summary_levels):
                raise RuntimeError("Hierarchical summarization did not reduce the input.")
            summary_levels = grouped_summaries
        summary = summary_levels[0]

        return {
            "summary": summary,
            "model": model_name,
            "hierarchical": len(chunk_summaries) > 1,
            "chunk_summaries": chunk_summaries,
            "messages_summarized": len(evidence),
            "supporting_messages": evidence,
            "warning": (
                "Machine-generated summary; review the linked source messages before "
                "treating any statement as a verified claim."
            ),
        }


# Initialize summarizer
def get_chat_summarizer():
    """Return a fresh chat summarizer instance."""
    summarizer = ChatSummarizer()
    summarizer.load_model()
    return summarizer