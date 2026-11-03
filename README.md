# WhatsApp Chat Analyzer

A comprehensive tool to analyze WhatsApp chat exports, providing insights through statistics, visualizations, and advanced NLP techniques.

## Description

This Streamlit application analyzes WhatsApp text exports locally and provides conversation statistics, participant and activity analysis, optional transformer inference, and annotation-driven experiments.

## Features

### Basic Features
- **Message Statistics**: Total messages, words, media files, and links shared
- **User Activity Analysis**: Identify most active users with percentages
- **Word Cloud Generation**: Visualize common words used in chats
- **Emoji Analysis**: Most used emojis and their frequencies
- **Timeline Analysis**: Message activity over time (daily, monthly, yearly)
- **Activity Heatmaps**: Hourly and weekly activity patterns

### Advanced Features
- **Sentiment Analysis**: Retains CardiffNLP RoBERTa and adds opt-in XLM-RoBERTa context-conditioned analysis with confidence scores and trends
- **Authorship Attribution**: Retains TF-IDF, Naive Bayes, SVM, and Random Forest baselines; research mode can evaluate multilingual Sentence-BERT embeddings and writing-style features
- **Conversation Summarization**: Retains extractive and LDA features, adds TextRank, and provides opt-in hierarchical BART, FLAN-T5, and Qwen 2.5 0.5B local summaries with source message IDs, timestamps, and participants
- **Emotion Recognition**: Optional English emotion classification with per-message scores; it is separate from sentiment and is not multilingual
- **Research Experiments**: Import manually annotated CSV/JSON datasets, validate labels, split by conversation, report accuracy/precision/recall/macro-F1 and confusion matrices, and export CSV/JSON results
- **Privacy Management**: Anonymize participants, mask phone numbers and emails, export sanitized data, and clear in-session conversation data and results
- **Interactive Visualizations**: Charts and graphs using Plotly and Matplotlib

Transformer features are opt-in and download model weights only when started. Conversation text is processed locally by these features; no external LLM API is called. Model comparison scores are shown only after evaluation on user-supplied annotations. BERTScore and supervised contrastive authorship results remain **Not evaluated yet**; provide human reference summaries/ratings for summary metrics.

The context sentiment feature conditions a multilingual sentiment checkpoint on the current message and its preceding conversation window. It is an inference-time context wrapper, not a separately fine-tuned context-aware architecture. Event candidates are verbatim messages matched by a small cue-word list and require human review.

## NLP and Research Workflow

1. Upload a WhatsApp `.txt` export and choose a participant and analysis date range.
2. Use **Research Dashboard** to download a blank annotation template or import a manually annotated `.csv`/`.json` file.
3. Use positive/negative/neutral sentiment labels and include a `conversation_id` on every row. Do not assign multiple IDs to parts of the same conversation; whole conversations are held out from training, validation, and test.
4. Start a baseline experiment or explicitly select a transformer model. Transformer weights are downloaded on demand; the first download requires network access and disk space.
5. Review the test-set metrics and confusion matrix. Export results as CSV or JSON. Empty/unevaluated model rows are not presented as measured results.
6. Use **Privacy Management** to create a sanitized export or delete the current upload and in-session experiment outputs.

Research dataset columns:

- Required: `message`, `sentiment`, `conversation_id`
- Optional: `user`, `emotion`, `message_id`, and `timestamp` (or `date_time`)
- Sentiment labels: `positive`, `negative`, `neutral`

The evaluation currently includes VADER and fold-local TF-IDF sentiment baselines; TF-IDF Naive Bayes, Linear SVM, and Random Forest authorship baselines; optional CardiffNLP/XLM-RoBERTa sentiment evaluation; optional multilingual Sentence-BERT authorship evaluation with writing-style features; and ROUGE-1/2/L plus optional human ratings when supplied. Summarization candidates for LDA, TextRank, BART, T5, and Qwen can be compared through the reference-summary CSV workflow. BERTScore and supervised contrastive learning are not reported as completed experiments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CharanLawha/whatsapp-chat-analyzer.git
   cd whatsapp-chat-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Export your WhatsApp chat (without media) as a .txt file
2. Upload the file through the Streamlit interface
3. Select a user or view overall statistics
4. Explore various analysis sections:
   - Statistics overview
   - Monthly/daily timelines
   - Activity heatmaps
   - Word clouds
   - Sentiment analysis
   - Author prediction
   - Chat summarization
   - Research Dashboard and Privacy Management

## Requirements

- Python 3.8+
- Libraries:
  - streamlit
  - pandas
  - numpy
  - plotly
  - matplotlib
  - seaborn
  - wordcloud
  - transformers
  - torch
  - scikit-learn
  - urlextract
  - vaderSentiment
  - Pillow
  - emoji

## Sample Data

A sample chat file (`sample_chat.txt`) is provided for testing the application.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and personal use only. Ensure you have permission to analyze any chat data you upload.
