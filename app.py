import hashlib
from io import BytesIO
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import preprocessor, helper
import author_prediction, sentiment_analyzer, chat_summarizer, emotion_analyzer
import privacy, research_tools

# —————————— GLOBAL PAGE CONFIG & CSS ——————————
st.set_page_config(
    page_title="WhatsApp Chat Analyzer | Charan Lawha",
    page_icon="✨",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://example.com/help',
        'Report a bug': 'https://example.com/bug',
        'About': 'This app analyzes your WhatsApp chat data.'
    }
)
# Inject a bit of custom CSS for titles, cards, and spacing
st.markdown(
    """
    <style>
    /* Title styling */
    .title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        letter-spacing: 0.02em;
    }
    /* Card containers for metrics */
    .metric-card {
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
        padding: 1rem;
        border-radius: 0.85rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.12);
        text-align: center;
        border: 1px solid #dbeafe;
    }
    /* Sidebar section headers */
    .sidebar-header {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #312E81;
    }
    /* Styled gradient separator */
    .separator {
        height: 2px;
        background: linear-gradient(to right, #4F46E5, #06B6D4);
        border: none;
        margin: 1rem 0;
        border-radius: 2px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# —————————— SIDEBAR ——————————

def request_uploaded_data_deletion():
    st.session_state["clear_uploaded_data"] = True


def load_sidebar():
    if st.session_state.pop("clear_uploaded_data", False):
        for key in (
            "chat_upload",
            "show_analysis",
            "annotated_dataset",
            "annotation_dataset_fingerprint",
            "annotation_results",
            "authorship_results",
            "annotation_upload",
            "summary_evaluation_upload",
            "summary_experiment_results",
            "summary_dataset_fingerprint",
            "research_transformer_sentiment",
            "context_sentiment_results",
            "context_sentiment_metadata",
            "emotion_results",
            "emotion_metadata",
            "roberta_sentiment_results",
            "roberta_sentiment_metadata",
            "transformer_summary_results",
            "transformer_summary_metadata",
            "transformer_authorship_results",
            "analysis_date_range",
            "summary_date_range",
            "analysis_dataframe_fingerprint",
            "author_predictor",
            "predictor_trained",
            "uploaded_data_fingerprint",
        ):
            st.session_state.pop(key, None)

    st.sidebar.markdown(
        """
        <style>
        /* Modern upload card */
        .upload-card {
            background: linear-gradient(135deg, #eef2ff, #f8fbff);
            border-radius: 14px;
            padding: 22px 20px;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.16);
            margin-bottom: 25px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1f2937;
            border: 1px solid #c7d2fe;
        }

        .upload-card h2 {
            font-size: 1.4rem;
            font-weight: 700;
            margin: 0 0 12px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            color: #4338CA;
        }

        .upload-card p {
            font-size: 0.95rem;
            margin-bottom: 15px;
            color: #475569;
        }
        /* Dropdown styling */
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            border: 2px solid #4F46E5 !important;
            background-color: #f8faff !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1f2937 !important;
            padding: 6px 9px !important;
            margin-bottom: 8px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
            height: auto !important;
        }

        div[data-baseweb="select"]:hover > div {
            border-color: #312E81 !important;
            box-shadow: 0 0 8px rgba(79, 70, 229, 0.2);
        }

        /* Show Analysis Button */
        .stButton > button {
            background: linear-gradient(135deg, #4F46E5, #06B6D4);
            border: none;
            border-radius: 14px;
            color: white;
            font-weight: 700;
            padding: 12px 0;
            width: 100%;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
            transition: background 0.3s ease;
            margin-bottom: 30px;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #4338CA, #0891B2);
            box-shadow: 0 0 12px rgba(79, 70, 229, 0.45);
        }

        /* File Uploader container */
        .stFileUploader {
            background-color: #f8faff;
            border: 1.5px solid #4F46E5;
            border-radius: 7px;
            padding: 6px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 0.9rem;
            margin-bottom: 10px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .stFileUploader:hover {
            border-color: #312E81;
            box-shadow: 0 0 8px rgba(79, 70, 229, 0.2);
        }

        /* Browse files button inside uploader */
        .stFileUploader [data-testid="stFileUploadDropzone"] {
            border: none !important;
            padding: 6px 12px !important;
            border-radius: 10px !important;
            background: linear-gradient(135deg, #4F46E5, #06B6D4) !important;
            color: white !important;
            text-align: center !important;
            font-weight: bold !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 0.9rem !important;
            transition: background 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
            cursor: pointer !important;
        }

        .stFileUploader [data-testid="stFileUploadDropzone"]:hover {
            background: linear-gradient(135deg, #4338CA, #0891B2) !important;
            box-shadow: 0 0 12px rgba(79, 70, 229, 0.45);
        }

        .sidebar-footer {
            margin-top: 2rem;
            padding-top: 0.8rem;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            font-size: 0.9rem;
            color: #4F46E5;
            font-weight: 700;
        }

        </style>

        <!-- Upload Section -->
        <div class="upload-card">
            <h2>📁 Upload Chat</h2>
            <p>Export your WhatsApp chat as a <strong>.txt</strong> file and upload it here to begin analysis.</p>
        </div>
        <div class="sidebar-footer">Built by Charan Lawha</div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "📁",
        type=["txt"],
        help="Export your chat from WhatsApp and upload the .txt file here.",
        label_visibility="collapsed",
        key="chat_upload"
    )

    df = None
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        uploaded_bytes = uploaded_file.getvalue()
        fingerprint = hashlib.sha256(uploaded_bytes).hexdigest()
        if fingerprint != st.session_state.get("uploaded_data_fingerprint"):
            for key in (
                "context_sentiment_results",
                "roberta_sentiment_results",
                "roberta_sentiment_metadata",
                "transformer_summary_results",
                "transformer_authorship_results",
                "author_predictor",
                "predictor_trained",
                "analysis_date_range",
                "summary_date_range",
                "analysis_dataframe_fingerprint",
                "research_transformer_sentiment",
            ):
                st.session_state.pop(key, None)
            st.session_state.uploaded_data_fingerprint = fingerprint
        bytes_data = uploaded_bytes.decode("utf-8")
        df = preprocessor.preprocessor(bytes_data)

        # Assuming df has a 'user' column to select from, else adapt this
        users = df['user'].unique().tolist() if df is not None else []

        selected_user = st.sidebar.selectbox(
            "Select User",
            options=["Overall"] + users,
            index=0,
            help="Select a user to analyze specific chat data."
        )

        if "date_time" in df.columns and not df.empty:
            dates = pd.to_datetime(df["date_time"], errors="coerce").dropna().dt.date
            if not dates.empty:
                date_range = st.sidebar.date_input(
                    "Analysis date range",
                    value=(dates.min(), dates.max()),
                    key="analysis_date_range",
                )
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    start_date, end_date = date_range
                    timestamp_dates = pd.to_datetime(df["date_time"]).dt.date
                    df = df[
                        (timestamp_dates >= start_date)
                        & (timestamp_dates <= end_date)
                    ].copy()

        fingerprint_columns = [
            column for column in ("message_id", "date_time", "user", "message")
            if column in df.columns
        ]
        analysis_fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(
                df[fingerprint_columns], index=False
            ).values.tobytes()
        ).hexdigest()
        if (
            analysis_fingerprint
            != st.session_state.get("analysis_dataframe_fingerprint")
        ):
            for key in (
                "context_sentiment_results",
                "context_sentiment_metadata",
                "emotion_results",
                "emotion_metadata",
                "roberta_sentiment_results",
                "roberta_sentiment_metadata",
                "transformer_summary_results",
                "transformer_summary_metadata",
                "author_predictor",
                "predictor_trained",
            ):
                st.session_state.pop(key, None)
            st.session_state.analysis_dataframe_fingerprint = analysis_fingerprint

        if "show_analysis" not in st.session_state:
            st.session_state.show_analysis = False

        if st.sidebar.button("Show Analysis"):
            st.session_state.show_analysis = True

        if st.sidebar.button("Hide Analysis"):
            st.session_state.show_analysis = False

        return df, selected_user, st.session_state.show_analysis

    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    return df, None, st.session_state.show_analysis





# # —————————— USER SELECTION ——————————
def select_user(df):
    st.sidebar.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-header">👤 Select User</div>', unsafe_allow_html=True)
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("", user_list)
    return selected_user

def show_analysis_button():
    st.sidebar.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    return st.sidebar.button("🚀 Show Analysis", use_container_width=True)

# —————————— TOP METRICS ——————————
def display_stats(selected_user, df):
    num_messages, words, media_messages, links = helper.fetch_stats(df, selected_user)
    st.markdown('<div class="title">🔍 Top-Level Stats</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Total Messages")
        st.markdown(f"#### {num_messages}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Total Words")
        st.markdown(f"#### {words}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Media Shared")
        st.markdown(f"#### {media_messages}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Links Shared")
        st.markdown(f"#### {links}")
        st.markdown('</div>', unsafe_allow_html=True)

# —————————— TIMELINES (MONTHLY & DAILY) ——————————
def plot_timeline(title, timeline, x_col, y_col, marker_color):
    fig = px.line(
        timeline, x=x_col, y=y_col, 
        title=title, 
        markers=True
    )
    fig.update_traces(marker=dict(size=6, color=marker_color))
    fig.update_layout(
        title_font_size=20,
        xaxis_title=x_col.capitalize(),
        yaxis_title='Messages',
        xaxis_tickangle=-45,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# —————————— ACTIVITY MAP ——————————
def plot_activity_map(selected_user, df):
    st.markdown('<div class="title">📅 Activity Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Busy Days of Week")
        busy_day = helper.week_activity_map(selected_user, df)
        fig = go.Figure([go.Bar(x=busy_day.index, y=busy_day.values, marker_color='indigo')])
        fig.update_layout(
            xaxis_title='Day of Week',
            yaxis_title='Count',
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Busy Months")
        busy_month = helper.month_activity_map(selected_user, df)
        fig = go.Figure([go.Bar(x=busy_month.index, y=busy_month.values, marker_color='crimson')])
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Count',
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_weekly_activity_heatmap(selected_user, df):
    st.subheader("Weekly Heatmap")
    heatmap_data = helper.activity_heatmap(selected_user, df)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        heatmap_data, 
        cmap="YlGnBu", 
        linewidths=0.5, 
        linecolor="white",
        annot=False,
        cbar_kws={'shrink': 0.5}
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    st.pyplot(fig)

# —————————— BUSY USERS (ONLY “Overall”) ——————————

def display_busy_users(df):
    st.markdown('<div class="title">👥 Most Active Participants</div>', unsafe_allow_html=True)
    x, new_df = helper.most_busy_users(df)  # x is a Series: index = usernames, values = counts

    # Matplotlib Chart (Perfect as is)
    st.subheader("Matplotlib Chart")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x.index, x.values, color="#E74C3C")
    plt.xticks(rotation=45)
    ax.set_ylabel("**Message Count**")
    ax.set_title("Top Active Users")
    st.pyplot(fig)

    # Dataframe (Bottom)
    st.subheader("**User Rankings**")
    second_col = new_df.columns[1]
    st.dataframe(new_df.style.background_gradient(cmap="Blues", subset=[second_col]),use_container_width=True)

# —————————— WORDCLOUD & COMMON WORDS ——————————
def display_wordcloud(selected_user, df):
    if selected_user == 'group_notification':
        st.info("Wordcloud is not available for group notifications.")
        return

    st.markdown('<div class="title">☁️ Wordcloud</div>', unsafe_allow_html=True)
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(df_wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)

def display_common_words(selected_user, df):
    if selected_user == 'group_notification':
        st.info("Common words analysis is not available for group notifications.")
        return

    st.markdown('<div class="title">📝 Most Common Words</div>', unsafe_allow_html=True)
    common_words_df = helper.most_common_words(selected_user, df)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(common_words_df[0], common_words_df[1], color="#3498DB")
    ax.set_ylabel("Words")
    ax.set_xlabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# —————————— EMOJI ANALYSIS ——————————
def display_emoji_analysis(selected_user, df):
    st.markdown('<div class="title">😀 Emoji Analysis</div>', unsafe_allow_html=True)
    emoji_df = helper.emoji_helper(selected_user, df)

    if not isinstance(emoji_df, pd.DataFrame):
        st.write(emoji_df)
    else:
        col1, col2 = st.columns((1, 1), gap="large")
        with col1:
            st.dataframe(emoji_df.style.set_caption("Top Emojis & Counts"),use_container_width=True)
        with col2:
            pie_fig = px.pie(
                names=emoji_df[0].head(5), 
                values=emoji_df[1].head(5), 
                title='Top 5 Emojis',
                hole=0.4
            )
            pie_fig.update_layout(
                margin=dict(t=30, b=20, l=20, r=20),
                legend=dict(orientation="h", y=-0.1)
            )
            st.plotly_chart(pie_fig, use_container_width=True)

# —————————— AVERAGES & HOURLY ACTIVITY ——————————
def display_avg_messages(selected_user, df):
    avg_per_day = helper.average_messages_per_day(selected_user, df)
    st.markdown('<div class="metric-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.subheader("📈 Avg. Messages / Day")
    st.markdown(f"### {round(avg_per_day, 2)}")
    st.markdown('</div>', unsafe_allow_html=True)

def display_hourly_activity(selected_user, df):
    st.markdown('<div class="title">⏰ Hourly Activity</div>', unsafe_allow_html=True)
    hourly_df = helper.hourly_activity(selected_user, df)
    fig = go.Figure([go.Bar(x=hourly_df.index, y=hourly_df.values, marker_color='#1ABC9C')])
    fig.update_layout(
        xaxis_title='Hour',
        yaxis_title='Messages',
        margin=dict(t=30, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# —————————— DAY-OF-MONTH & LENGTH DISTRIBUTION ——————————
def plot_day_of_month_activity(selected_user, df):
    st.markdown('<div class="title">📅 Day of Month Activity</div>', unsafe_allow_html=True)
    df_day = helper.day_of_month_activity(selected_user, df)
    fig = go.Figure([go.Bar(x=df_day.index, y=df_day.values, marker_color='#8E44AD')])
    fig.update_layout(
        xaxis_title='Day of Month',
        yaxis_title='Messages',
        margin=dict(t=30, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def display_message_length_dist(selected_user, df):
    st.markdown('<div class="title">✍️ Message Length Distribution</div>', unsafe_allow_html=True)
    msg_len_df = helper.message_length_distribution(selected_user, df)
    fig = px.histogram(
        msg_len_df, 
        nbins=30, 
        title="Message Lengths (Number of Characters)"
    )
    fig.update_layout(
        margin=dict(t=30, b=20, l=20, r=20),
        xaxis_title="Length",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)

# —————————— LEXICAL RICHNESS & SENTIMENT ——————————
def display_lexical_richness(selected_user, df):
    richness = helper.lexical_richness(selected_user, df)
    st.markdown('<div class="metric-card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.subheader("🧠 Lexical Richness")
    st.markdown(f"### {richness:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)


def display_sentiment_analysis(selected_user, df):
    st.markdown('<div class="title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    sentiment_counts, sentiment_df = helper.sentiment_analysis(selected_user, df)

    # Pie chart section
    pie_fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Overall Sentiments",
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    )
    pie_fig.update_layout(
        margin=dict(t=30, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    # DataFrame section
    st.markdown("**Detailed Sentiment Data**")
    st.dataframe(sentiment_df.style.highlight_max(axis=0, color="#A3E4D7"),use_container_width=True)


# —————————— AUTHOR PREDICTION ——————————
def display_author_prediction(selected_user, df):
    """Display the existing authorship attribution baseline."""
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">✍️ Authorship Attribution</div>', unsafe_allow_html=True)
    
    # Initialize predictor if not exists
    if 'author_predictor' not in st.session_state:
        st.session_state.author_predictor = author_prediction.AuthorPredictor()
        st.session_state.predictor_trained = False
    
    predictor = st.session_state.author_predictor
    
    # Show data stats
    author_messages = df[
        (df['message'] != '<Media omitted>')
        & (df['user'] != 'group_notification')
    ]
    total_messages = len(author_messages)
    unique_users = author_messages['user'].nunique()
    
    st.info(f"📊 Available data: {total_messages} messages from {unique_users} users")
    
    if total_messages < 10:
        st.warning("⚠️ Need at least 10 messages for reliable training. Upload a larger chat file.")
        return
    
    if unique_users < 2:
        st.warning("⚠️ Need at least 2 different users for prediction. Upload a group chat.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎓 Train Predictor", key="train_predictor"):
            with st.spinner("Training author prediction model..."):
                result = predictor.train(df)
                if "error" in result:
                    st.error(f"Training failed: {result['error']}")
                else:
                    st.session_state.predictor_trained = True
                    st.success(f"✓ Model trained! Best: {result['best_model']}")
                    st.info(
                        "Exploratory message-level cross-validation accuracy "
                        f"(not held-out conversation evaluation): {result['accuracy']}"
                    )
                    with st.expander("All Models Performance"):
                        for model, acc in result['all_accuracies'].items():
                            st.write(f"- {model}: {acc}")
    
    if st.session_state.predictor_trained:
        with st.expander("Test Author Prediction", expanded=False):
            test_message = st.text_area("Enter a message to predict the author:", height=100)
            if test_message:
                author, confidence = predictor.predict(test_message)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Author", author, f"{confidence:.1%}")
                with col2:
                    st.metric("Model Used", predictor.model_name, predictor.get_stats()['num_users'])
        
        with st.expander("Top Predictions from Chat", expanded=False):
            top_predictions = predictor.get_top_predictions(df, top_n=10)
            if top_predictions is not None:
                st.dataframe(top_predictions, use_container_width=True)
            else:
                st.info("No predictions available")


# —————————— HIGH-PERFORMANCE SENTIMENT ANALYSIS ——————————
def display_roberta_sentiment(selected_user, df):
    """Display the existing RoBERTa model only after explicit user action."""
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">🤖 CardiffNLP RoBERTa Sentiment</div>', unsafe_allow_html=True)
    
    total_messages = len(df[df['message'] != '<Media omitted>'])
    st.info(f"📊 Available text messages: {total_messages}")
    
    if total_messages == 0:
        st.warning("⚠️ No messages to analyze. Upload a chat file with text messages.")
        return
    
    if st.button("Run CardiffNLP sentiment analysis", key="run_roberta_sentiment"):
        with st.spinner("Analyzing sentiment with RoBERTa..."):
            analyzer = sentiment_analyzer.get_sentiment_analyzer()
            result = analyzer.analyze_user_sentiment(df, selected_user)
        st.session_state.roberta_sentiment_results = result
        st.session_state.roberta_sentiment_metadata = {
            "participant": selected_user,
            "fingerprint": st.session_state.get("uploaded_data_fingerprint"),
        }

    metadata = st.session_state.get("roberta_sentiment_metadata")
    result = st.session_state.get("roberta_sentiment_results")
    if not result or metadata != {
        "participant": selected_user,
        "fingerprint": st.session_state.get("uploaded_data_fingerprint"),
    }:
        st.info("Not evaluated yet. Model weights download only when you run this analysis.")
        return
    if "error" in result:
        st.error(result["error"])
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Positive", result['counts']['POSITIVE'], result['percentages']['POSITIVE'])
    with col2:
        st.metric("Negative", result['counts']['NEGATIVE'], result['percentages']['NEGATIVE'])
    with col3:
        st.metric("Neutral", result['counts']['NEUTRAL'], result['percentages']['NEUTRAL'])
    with col4:
        st.metric("Positivity Score", f"{result['positive_score']:.2%}", "Overall mood")

    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [
            result['counts']['POSITIVE'],
            result['counts']['NEGATIVE'],
            result['counts']['NEUTRAL'],
        ],
    })
    fig = px.bar(
        sentiment_data,
        x='Sentiment',
        y='Count',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#95a5a6',
        },
        title="Sentiment Distribution (RoBERTa Analysis)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Analyzed {result['total_messages_analyzed']} messages")


def display_context_aware_sentiment(selected_user, df):
    st.markdown('<div class="title">🧠 Context-Aware Multilingual Sentiment</div>', unsafe_allow_html=True)
    st.caption(
        "Experimental context-conditioned inference: the selected transformer receives "
        "the current message and 2–5 preceding messages. This is not a fine-tuned "
        "context-aware architecture, and model confidence is not calibrated."
    )
    model_options = {
        "XLM-RoBERTa (multilingual)": sentiment_analyzer.XLM_ROBERTA_SENTIMENT_MODEL,
        "CardiffNLP Twitter-RoBERTa (English-oriented)": (
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ),
    }
    model_label = st.selectbox(
        "Sentiment model",
        list(model_options),
        key="context_sentiment_model",
    )
    context_messages = st.slider(
        "Previous messages to include",
        min_value=2,
        max_value=5,
        value=3,
        key="sentiment_context_window",
    )
    if st.button("Analyze conversation context", key="run_context_sentiment"):
        try:
            with st.spinner("Loading the selected model and analyzing locally..."):
                analyzer = sentiment_analyzer.ContextAwareSentimentAnalyzer(
                    model_options[model_label]
                )
                result = analyzer.analyze(df, context_messages=context_messages)
                st.session_state.context_sentiment_results = result
                st.session_state.context_sentiment_metadata = {
                    "model": model_options[model_label],
                    "context": context_messages,
                    "fingerprint": st.session_state.get(
                        "analysis_dataframe_fingerprint"
                    ),
                }
        except Exception as error:
            st.error(f"Context sentiment analysis failed: {error}")
            return

    results = st.session_state.get("context_sentiment_results")
    expected_metadata = {
        "model": model_options[model_label],
        "context": context_messages,
        "fingerprint": st.session_state.get("analysis_dataframe_fingerprint"),
    }
    if (
        results is None
        or st.session_state.get("context_sentiment_metadata") != expected_metadata
    ):
        st.info("Not evaluated yet. Model weights download only after you start analysis.")
        return
    if selected_user != "Overall":
        results = results[results["user"] == selected_user]
    if results.empty:
        st.info("No eligible text messages were found for this participant.")
        return

    display = results.copy()
    display["sentiment"] = display["sentiment"].str.title()
    st.dataframe(
        display[[column for column in (
            "message_id", "date_time", "user", "message", "sentiment", "confidence"
        ) if column in display.columns]],
        use_container_width=True,
    )
    counts = results["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Messages")
    st.plotly_chart(
        px.bar(counts, x="Sentiment", y="Messages", color="Sentiment"),
        use_container_width=True,
    )
    if "date_time" in results.columns:
        timeline = results.dropna(subset=["date_time"]).copy()
        if not timeline.empty:
            timeline["date"] = pd.to_datetime(timeline["date_time"]).dt.date
            trend = timeline.groupby(["date", "sentiment"]).size().reset_index(name="messages")
            st.plotly_chart(
                px.line(
                    trend,
                    x="date",
                    y="messages",
                    color="sentiment",
                    markers=True,
                    title="Contextual sentiment trend",
                ),
                use_container_width=True,
            )


def display_emotion_recognition(selected_user, df):
    st.markdown('<div class="title">🙂 Emotion Recognition</div>', unsafe_allow_html=True)
    st.caption(
        "Uses an English-language emotion classifier (anger, disgust, fear, joy, "
        "neutral, sadness, surprise). It is not a multilingual emotion model, and "
        "scores are uncalibrated."
    )
    if st.button("Run local emotion recognition", key="run_emotion_recognition"):
        try:
            with st.spinner("Loading the English emotion model and analyzing locally..."):
                results = emotion_analyzer.analyze_emotions(df)
            st.session_state.emotion_results = results
            st.session_state.emotion_metadata = st.session_state.get(
                "analysis_dataframe_fingerprint"
            )
        except Exception as error:
            st.error(f"Emotion recognition failed: {error}")
            return

    results = st.session_state.get("emotion_results")
    if (
        results is None
        or st.session_state.get("emotion_metadata")
        != st.session_state.get("analysis_dataframe_fingerprint")
    ):
        st.info("Not evaluated yet. Model weights download only after you start analysis.")
        return
    if selected_user != "Overall":
        results = results[results["user"] == selected_user]
    if results.empty:
        st.info("No eligible text messages were found.")
        return

    counts = results["emotion"].value_counts().rename_axis("Emotion").reset_index(name="Messages")
    st.plotly_chart(
        px.bar(counts, x="Emotion", y="Messages", color="Emotion"),
        use_container_width=True,
    )
    st.dataframe(
        results[[column for column in (
            "message_id", "date_time", "user", "message", "emotion", "confidence"
        ) if column in results.columns]],
        use_container_width=True,
    )


def display_metric_comparison(results, metrics, title):
    available_metrics = [column for column in metrics if column in results.columns]
    if not available_metrics:
        return
    chart_data = results[["model", *available_metrics]].copy()
    for metric in available_metrics:
        chart_data[metric] = pd.to_numeric(chart_data[metric], errors="coerce")
    chart_data = chart_data.dropna(subset=available_metrics, how="all")
    if chart_data.empty:
        return
    chart_data = chart_data.melt(
        id_vars="model",
        var_name="Metric",
        value_name="Score",
    )
    st.plotly_chart(
        px.bar(
            chart_data,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            title=title,
            range_y=[0, 1],
        ),
        use_container_width=True,
    )


def display_research_dashboard(df):
    st.markdown('<div class="title">🧪 Research Experiment Dashboard</div>', unsafe_allow_html=True)
    st.caption(
        "Only manually supplied annotations are used as ground truth. Models without "
        "completed evaluations are labeled Not evaluated yet."
    )
    st.caption(
        "Available ablation comparisons: XLM-RoBERTa with/without preceding context, "
        "and Sentence-BERT embeddings with/without writing-style features."
    )
    uploaded_annotations = st.file_uploader(
        "Import manually annotated dataset (.csv or .json)",
        type=["csv", "json"],
        key="annotation_upload",
    )
    if uploaded_annotations is not None:
        annotation_bytes = uploaded_annotations.getvalue()
        annotation_fingerprint = hashlib.sha256(annotation_bytes).hexdigest()
        if (
            annotation_fingerprint
            != st.session_state.get("annotation_dataset_fingerprint")
        ):
            for key in (
                "annotation_results",
                "authorship_results",
                "transformer_authorship_results",
            ):
                st.session_state.pop(key, None)
            st.session_state.annotation_dataset_fingerprint = annotation_fingerprint
        try:
            st.session_state.annotated_dataset = research_tools.load_annotated_dataset(
                annotation_bytes,
                uploaded_annotations.name,
            )
        except ValueError as error:
            st.session_state.pop("annotated_dataset", None)
            st.error(str(error))

    st.subheader("Create an annotation template")
    st.write(
        "Labels are intentionally blank. Annotate sentiment and, optionally, emotion. "
        "A conversation_id is required to keep messages from the same conversation "
        "together during evaluation."
    )
    template = df.copy()
    template["conversation_id"] = "conversation_1"
    template["sentiment"] = ""
    template["emotion"] = ""
    annotation_columns = [
        "conversation_id",
        "message_id",
        "date_time",
        "user",
        "message",
        "sentiment",
        "emotion",
    ]
    template = template[[column for column in annotation_columns if column in template.columns]]
    st.download_button(
        "Download blank annotation template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="whatsapp_annotations_template.csv",
        mime="text/csv",
    )
    if len(template) > 5000:
        st.info("Showing the first 5,000 messages in the editable preview; the template contains the full chat.")
    edited_template = st.data_editor(
        template.head(5000),
        hide_index=True,
        use_container_width=True,
        key="annotation_editor",
        disabled=[
            column for column in template.columns
            if column not in ("conversation_id", "sentiment", "emotion")
        ],
    )
    st.download_button(
        "Download edited annotations",
        data=edited_template.to_csv(index=False).encode("utf-8"),
        file_name="whatsapp_manual_annotations.csv",
        mime="text/csv",
    )
    if st.button("Use edited annotations for experiments", key="use_edited_annotations"):
        try:
            research_tools.validate_annotated_dataset(edited_template)
            st.session_state.annotated_dataset = edited_template.copy()
            for key in (
                "annotation_results",
                "authorship_results",
                "transformer_authorship_results",
            ):
                st.session_state.pop(key, None)
            st.success("Manually entered annotations are ready for evaluation.")
        except ValueError as error:
            st.error(f"Cannot use annotations yet: {error}")

    annotations = st.session_state.get("annotated_dataset")
    if annotations is None:
        st.info("No annotated dataset imported. Evaluation status: Not evaluated yet.")
    else:
        try:
            dataset_info = research_tools.validate_annotated_dataset(annotations)
            st.success(
                f"Validated {dataset_info['rows']} annotated messages from "
                f"{dataset_info['conversations']} conversations."
            )
            st.write("Sentiment class distribution")
            st.dataframe(
                pd.DataFrame(
                    dataset_info["class_distribution"].items(),
                    columns=["Sentiment", "Annotated messages"],
                ),
                hide_index=True,
            )
            if dataset_info["emotion_class_distribution"]:
                st.write("Emotion annotation distribution")
                st.dataframe(
                    pd.DataFrame(
                        dataset_info["emotion_class_distribution"].items(),
                        columns=["Emotion", "Annotated messages"],
                    ),
                    hide_index=True,
                )
        except ValueError as error:
            st.error(f"Dataset validation failed: {error}")
            return

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Evaluate sentiment baselines", key="evaluate_sentiment_baselines"):
                try:
                    with st.spinner("Evaluating on held-out conversations..."):
                        scores, matrices = research_tools.evaluate_tfidf_sentiment_baselines(
                            annotations
                        )
                    st.session_state.annotation_results = {
                        "scores": scores,
                        "matrices": matrices,
                    }
                except ValueError as error:
                    st.error(str(error))
        with col2:
            if st.button("Evaluate authorship baselines", key="evaluate_authorship_baselines"):
                try:
                    with st.spinner("Evaluating authorship on held-out conversations..."):
                        scores, matrices = research_tools.evaluate_authorship_baselines(
                            annotations
                        )
                    st.session_state.authorship_results = {
                        "scores": scores,
                        "matrices": matrices,
                    }
                except ValueError as error:
                    st.error(str(error))

        transformer_sentiment_models = {
            "CardiffNLP Twitter-RoBERTa": (
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                0,
            ),
            "XLM-RoBERTa multilingual": (
                sentiment_analyzer.XLM_ROBERTA_SENTIMENT_MODEL,
                0,
            ),
            "Context-conditioned XLM-RoBERTa": (
                sentiment_analyzer.XLM_ROBERTA_SENTIMENT_MODEL,
                3,
            ),
        }
        transformer_label = st.selectbox(
            "Optional transformer sentiment model (downloads weights on request)",
            list(transformer_sentiment_models),
            key="research_transformer_sentiment",
        )
        if st.button("Evaluate selected transformer", key="evaluate_transformer_sentiment"):
            model_name, context_size = transformer_sentiment_models[transformer_label]
            try:
                with st.spinner("Running the selected local model on held-out conversations..."):
                    model_result, confusion = research_tools.evaluate_transformer_sentiment(
                        annotations,
                        model_name=model_name,
                        context_messages=context_size,
                    )
                model_result["model"] = transformer_label
                current = st.session_state.get(
                    "annotation_results",
                    {"scores": pd.DataFrame(), "matrices": {}},
                )
                existing_scores = current["scores"]
                existing_scores = existing_scores[
                    existing_scores["model"] != model_result["model"]
                ] if "model" in existing_scores.columns else existing_scores
                current["scores"] = pd.concat(
                    [existing_scores, pd.DataFrame([model_result])],
                    ignore_index=True,
                )
                current["matrices"][model_result["model"]] = confusion
                st.session_state.annotation_results = current
            except Exception as error:
                st.error(f"Transformer sentiment evaluation failed: {error}")

        sentiment_results = st.session_state.get("annotation_results")
        st.subheader("Sentiment model comparison")
        if sentiment_results is None:
            st.info("No sentiment experiments completed. Select a baseline or transformer evaluation above.")
        else:
            sentiment_scores = sentiment_results["scores"]
            status_rows = pd.DataFrame([
                {"model": model, "status": "Not evaluated yet"}
                for model in (
                    "CardiffNLP Twitter-RoBERTa",
                    "XLM-RoBERTa multilingual",
                    "Context-conditioned XLM-RoBERTa",
                )
                if model not in sentiment_scores["model"].tolist()
            ])
            st.dataframe(
                pd.concat([sentiment_scores, status_rows], ignore_index=True),
                hide_index=True,
                use_container_width=True,
            )
            display_metric_comparison(
                sentiment_scores,
                ["accuracy", "precision_macro", "recall_macro", "macro_f1"],
                "Sentiment baseline and transformer metrics",
            )
            for model_name, matrix in sentiment_results["matrices"].items():
                with st.expander(f"{model_name} confusion matrix"):
                    matrix_frame = pd.DataFrame(
                        matrix["values"],
                        index=matrix["labels"],
                        columns=matrix["labels"],
                    )
                    st.dataframe(matrix_frame)
                    st.plotly_chart(
                        go.Figure(
                            data=go.Heatmap(
                                z=matrix["values"],
                                x=matrix["labels"],
                                y=matrix["labels"],
                                colorscale="Blues",
                            )
                        ),
                        use_container_width=True,
                    )
            st.download_button(
                "Export sentiment results (CSV)",
                research_tools.export_results(sentiment_results["scores"], "csv"),
                file_name="sentiment_experiment_results.csv",
                mime="text/csv",
            )
            st.download_button(
                "Export sentiment results (JSON)",
                research_tools.export_results(sentiment_results["scores"], "json"),
                file_name="sentiment_experiment_results.json",
                mime="application/json",
            )

        authorship_results = st.session_state.get("authorship_results")
        st.subheader("Authorship attribution comparison")
        if authorship_results is None:
            st.info("No authorship experiments completed yet.")
        else:
            st.dataframe(authorship_results["scores"], hide_index=True, use_container_width=True)
            display_metric_comparison(
                authorship_results["scores"],
                ["test_accuracy", "precision_macro", "recall_macro", "macro_f1"],
                "Authorship baseline metrics",
            )
            for model_name, matrix in authorship_results["matrices"].items():
                with st.expander(f"{model_name} per-author results"):
                    st.dataframe(pd.DataFrame(matrix["per_author"]).T)
                    st.dataframe(
                        pd.DataFrame(
                            matrix["values"],
                            index=matrix["labels"],
                            columns=matrix["labels"],
                        )
                    )
                    st.plotly_chart(
                        go.Figure(
                            data=go.Heatmap(
                                z=matrix["values"],
                                x=matrix["labels"],
                                y=matrix["labels"],
                                colorscale="Blues",
                            )
                        ),
                        use_container_width=True,
                    )
            st.download_button(
                "Export authorship results (CSV)",
                research_tools.export_results(authorship_results["scores"], "csv"),
                file_name="authorship_experiment_results.csv",
                mime="text/csv",
            )
        if st.button(
            "Evaluate multilingual Sentence-BERT authorship model",
            key="evaluate_transformer_authorship",
        ):
            try:
                with st.spinner("Loading Sentence-BERT and evaluating held-out conversations..."):
                    scores, matrices = research_tools.evaluate_transformer_authorship(
                        annotations
                    )
                st.session_state.transformer_authorship_results = {
                    "scores": scores,
                    "matrices": matrices,
                }
            except Exception as error:
                st.error(f"Transformer authorship evaluation failed: {error}")
        transformer_authorship = st.session_state.get(
            "transformer_authorship_results"
        )
        if transformer_authorship:
            st.dataframe(
                transformer_authorship["scores"],
                hide_index=True,
                use_container_width=True,
            )
            display_metric_comparison(
                transformer_authorship["scores"],
                ["test_accuracy", "precision_macro", "recall_macro", "macro_f1"],
                "Sentence-BERT authorship metrics",
            )
            st.download_button(
                "Export transformer authorship results",
                research_tools.export_results(
                    transformer_authorship["scores"], "json"
                ),
                file_name="transformer_authorship_results.json",
                mime="application/json",
            )
            for model_name, matrix in transformer_authorship["matrices"].items():
                with st.expander(f"{model_name} per-author results"):
                    st.dataframe(pd.DataFrame(matrix["per_author"]).T)
                    st.dataframe(
                        pd.DataFrame(
                            matrix["values"],
                            index=matrix["labels"],
                            columns=matrix["labels"],
                        )
                    )
                    st.plotly_chart(
                        go.Figure(
                            data=go.Heatmap(
                                z=matrix["values"],
                                x=matrix["labels"],
                                y=matrix["labels"],
                                colorscale="Blues",
                            )
                        ),
                        use_container_width=True,
                    )
        st.caption("Supervised contrastive learning: Not evaluated yet.")

    st.subheader("Summarization evaluation")
    st.write(
        "Upload a human-authored reference and candidate summaries for each model. "
        "Required CSV columns: `model`, `reference_summary`, `candidate_summary`. "
        "Optional human ratings: `human_relevance`, `human_factuality`, "
        "`human_coverage` (each on a 1 to 5 scale)."
    )
    summary_evaluation_file = st.file_uploader(
        "Import summary evaluation references (.csv)",
        type=["csv"],
        key="summary_evaluation_upload",
    )
    if summary_evaluation_file is not None:
        summary_bytes = summary_evaluation_file.getvalue()
        summary_fingerprint = hashlib.sha256(summary_bytes).hexdigest()
        if (
            summary_fingerprint
            != st.session_state.get("summary_dataset_fingerprint")
        ):
            st.session_state.pop("summary_experiment_results", None)
            st.session_state.summary_dataset_fingerprint = summary_fingerprint
        try:
            summary_scores = research_tools.evaluate_summary_dataset(
                pd.read_csv(BytesIO(summary_bytes))
            )
            st.session_state.summary_experiment_results = summary_scores
        except (ValueError, pd.errors.ParserError) as error:
            st.session_state.pop("summary_experiment_results", None)
            st.error(f"Could not evaluate summary references: {error}")
    summary_scores = st.session_state.get("summary_experiment_results")
    if summary_scores is None:
        st.info("ROUGE, BERTScore, and human evaluation: Not evaluated yet.")
    else:
        available_models = summary_scores["model"].tolist()
        status_rows = pd.DataFrame([
            {"model": model, "status": "Not evaluated yet"}
            for model in ("LDA", "TextRank", "BART", "T5", "Qwen")
            if model not in available_models
        ])
        st.dataframe(
            pd.concat([summary_scores, status_rows], ignore_index=True),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Export summary evaluation (CSV)",
            research_tools.export_results(summary_scores, "csv"),
            file_name="summary_evaluation_results.csv",
            mime="text/csv",
        )
        if not any(column.startswith("human_") for column in summary_scores.columns):
            st.caption("Human evaluation: Not evaluated yet. BERTScore: Not evaluated yet.")


# —————————— CHAT SUMMARIZATION ——————————
def display_chat_summarization(selected_user, df):
    """Display chat summarization feature."""
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="title">📝 Chat Summarization</div>', unsafe_allow_html=True)
    
    # Show data stats
    total_messages = len(df[df['message'] != '<Media omitted>'])
    st.info(f"📊 Available data: {total_messages} messages for summarization")
    
    if total_messages < 5:
        st.warning("⚠️ Need at least 5 messages for summarization. Upload a larger chat file.")
        return
    
    try:
        summarizer = chat_summarizer.get_chat_summarizer()
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            summary_type = st.radio(
                "Summary Type:",
                ["Full Chat", "Recent 50 msgs", "Topics", "TextRank"],
                horizontal=True
            )
        
        with col1:
            if st.button("📄 Generate Summary", key="gen_summary"):
                with st.spinner("Generating summary..."):
                    if summary_type == "Full Chat":
                        result = summarizer.summarize_user_messages(df, selected_user)
                    elif summary_type == "Recent 50 msgs":
                        result = summarizer.summarize_user_messages(df, selected_user, num_messages=50)
                    elif summary_type == "TextRank":
                        result = summarizer.textrank_summary(df, selected_user)
                    else:
                        result = summarizer.topic_model(df, selected_user, n_topics=4, n_top_words=6)
                    
                    if "error" not in result:
                        st.success("✓ Summary Generated!")
                        
                        if 'summary' in result:
                            st.markdown("### Summary:")
                            st.info(result['summary'])
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Messages Summarized", result.get('messages_summarized', result.get('total_messages')))
                            with col_b:
                                if 'compression_ratio' in result:
                                    st.metric("Compression", result['compression_ratio'])
                            if "supporting_messages" in result:
                                st.markdown("#### Verbatim supporting messages")
                                st.dataframe(
                                    pd.DataFrame(result["supporting_messages"]),
                                    use_container_width=True,
                                )
                        
                        elif 'topics' in result:
                            st.markdown("### Topic Model Results")
                            topic_df = pd.DataFrame(result['topics'])
                            topic_df = topic_df.rename(columns={'topic_id': 'Topic', 'keywords_text': 'Keywords'})
                            st.dataframe(topic_df[['Topic', 'Keywords']], use_container_width=True)
                            
                            dist_df = pd.DataFrame({
                                'Topic': [t['topic_id'] for t in result['topics']],
                                'Messages': result['distribution']
                            })
                            st.bar_chart(dist_df.set_index('Topic'))
                            st.metric("Total Messages Analyzed", result['total_documents'])
                        
                        elif 'topic_summary' in result:
                            st.markdown("### Topic Overview:")
                            st.info(result['topic_summary'])
                            st.metric("Total Messages", result['total_messages'])
                    else:
                        st.warning(f"Could not generate summary: {result['error']}")
        
        with st.expander("Key Sentences from Chat"):
            key_sentences = summarizer.get_key_sentences(df, selected_user, num_sentences=5)
            if key_sentences:
                for i, sentence in enumerate(key_sentences, 1):
                    source_id = sentence.get("message_id") or "ID unavailable"
                    st.write(
                        f"{i}. **{sentence['user']}** · {source_id} · "
                        f"{sentence.get('timestamp') or 'Timestamp unavailable'}: "
                        f"{sentence['message']}"
                    )
            else:
                st.info("No key sentences available")

        with st.expander("Topic and event candidates (extractive review)"):
            event_candidates = summarizer.get_event_candidates(df, selected_user)
            if event_candidates:
                st.caption(
                    "These are verbatim messages matched by simple event/decision cue "
                    "words, not verified event classifications."
                )
                st.dataframe(
                    pd.DataFrame(event_candidates),
                    use_container_width=True,
                )
            else:
                st.info("No candidate messages matched the current cue list.")

        with st.expander("Transformer-based hierarchical summary"):
            st.warning(
                "The selected model runs locally after its weights are downloaded. "
                "Generated text can contain errors; review the linked source messages."
            )
            model_label = st.selectbox(
                "Local summarization model",
                list(chat_summarizer.SUMMARY_MODELS),
                key="summary_transformer_model",
            )
            summary_scope = st.selectbox(
                "Summary scope",
                [
                    "Complete conversation",
                    "Participant",
                    "Topic summaries",
                    "Daily summaries",
                    "Weekly summaries",
                    "Date range",
                ],
                key="summary_scope",
            )
            summary_user = "Overall" if summary_scope == "Complete conversation" else selected_user
            topic_count = 4
            if summary_scope == "Topic summaries":
                topic_count = st.slider(
                    "Number of topic groups",
                    min_value=2,
                    max_value=8,
                    value=4,
                    key="summary_topic_count",
                )
            if summary_scope == "Participant":
                participants = sorted(
                    user for user in df["user"].dropna().unique()
                    if user != "group_notification"
                )
                if participants:
                    summary_user = st.selectbox(
                        "Participant summary",
                        participants,
                        key="summary_participant",
                    )
            start_date = end_date = None
            if summary_scope in ("Daily summaries", "Weekly summaries", "Date range"):
                available_dates = pd.to_datetime(df["date_time"]).dt.date
                chosen_dates = st.date_input(
                    "Summary date range",
                    value=(available_dates.min(), available_dates.max()),
                    key="summary_date_range",
                )
                if isinstance(chosen_dates, (list, tuple)) and len(chosen_dates) == 2:
                    start_date, end_date = chosen_dates

            if st.button("Generate local transformer summary", key="gen_transformer_summary"):
                model_name = chat_summarizer.SUMMARY_MODELS[model_label][0]
                with st.spinner("Generating a hierarchical summary locally..."):
                    if summary_scope == "Topic summaries":
                        topic_groups = summarizer.topic_message_groups(
                            df,
                            selected_user=summary_user,
                            n_topics=topic_count,
                        )
                        if "error" in topic_groups:
                            result = topic_groups
                        else:
                            topic_periods = []
                            for topic in topic_groups["topics"]:
                                topic_result = summarizer.summarize_transformer(
                                    topic["messages"],
                                    selected_user="Overall",
                                    model_name=model_name,
                                )
                                topic_result["topic_keywords"] = topic["keywords"]
                                topic_periods.append({
                                    "period": (
                                        f"Topic {topic['topic_id']}: "
                                        f"{', '.join(topic['keywords'])}"
                                    ),
                                    "result": topic_result,
                                })
                            result = {
                                "periods": topic_periods,
                                "model": model_name,
                                "truncated_period_count": 0,
                            }
                    elif summary_scope in ("Daily summaries", "Weekly summaries"):
                        period_frame = df.copy()
                        period_frame["date_time"] = pd.to_datetime(
                            period_frame["date_time"], errors="coerce"
                        )
                        if start_date is not None:
                            period_frame = period_frame[
                                period_frame["date_time"].dt.date >= start_date
                            ]
                        if end_date is not None:
                            period_frame = period_frame[
                                period_frame["date_time"].dt.date <= end_date
                            ]
                        if summary_scope == "Daily summaries":
                            period_frame["summary_period"] = (
                                period_frame["date_time"].dt.date
                            )
                        else:
                            period_frame["summary_period"] = (
                                period_frame["date_time"].dt.to_period("W").apply(
                                    lambda period: period.start_time.date()
                                )
                            )
                        grouped_periods = list(
                            period_frame.groupby("summary_period", sort=True)
                        )
                        periods = []
                        for period, period_messages in grouped_periods[:31]:
                            periods.append({
                                "period": str(period),
                                "result": summarizer.summarize_transformer(
                                    period_messages,
                                    selected_user=summary_user,
                                    model_name=model_name,
                                ),
                            })
                        result = {
                            "periods": periods,
                            "model": model_name,
                            "truncated_period_count": max(0, len(grouped_periods) - 31),
                        }
                    else:
                        result = summarizer.summarize_transformer(
                            df,
                            selected_user=summary_user,
                            model_name=model_name,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    st.session_state.transformer_summary_results = result
                    st.session_state.transformer_summary_metadata = {
                        "model": model_name,
                        "scope": summary_scope,
                        "participant": summary_user,
                        "start": str(start_date),
                        "end": str(end_date),
                        "fingerprint": st.session_state.get(
                            "analysis_dataframe_fingerprint"
                        ),
                    }

            summary_result = st.session_state.get("transformer_summary_results")
            expected_summary_metadata = {
                "model": chat_summarizer.SUMMARY_MODELS[model_label][0],
                "scope": summary_scope,
                "participant": summary_user,
                "start": str(start_date),
                "end": str(end_date),
                "fingerprint": st.session_state.get(
                    "analysis_dataframe_fingerprint"
                ),
            }
            if (
                summary_result
                and st.session_state.get("transformer_summary_metadata")
                == expected_summary_metadata
            ):
                if "error" in summary_result:
                    st.error(summary_result["error"])
                elif "periods" in summary_result:
                    st.caption(f"Generated summaries with {summary_result['model']}.")
                    if summary_result["truncated_period_count"]:
                        st.info(
                            "Showing the first 31 periods. Narrow the date range to "
                            "summarize the remaining periods."
                        )
                    for item in summary_result["periods"]:
                        with st.expander(f"Summary for {item['period']}"):
                            period_result = item["result"]
                            if "error" in period_result:
                                st.error(period_result["error"])
                            else:
                                if "topic_keywords" in period_result:
                                    st.caption(
                                        "Topic keywords: "
                                        + ", ".join(period_result["topic_keywords"])
                                    )
                                st.write(period_result["summary"])
                                st.caption(period_result["warning"])
                                st.dataframe(
                                    pd.DataFrame(period_result["supporting_messages"]),
                                    use_container_width=True,
                                )
                else:
                    st.markdown("#### Summary")
                    st.write(summary_result["summary"])
                    st.caption(summary_result["warning"])
                    st.caption(
                        f"Model: {summary_result['model']} · "
                        f"Messages summarized: {summary_result['messages_summarized']} · "
                        f"Hierarchical: {summary_result['hierarchical']}"
                    )
                    st.markdown("#### Supporting source messages")
                    st.dataframe(
                        pd.DataFrame(summary_result["supporting_messages"]),
                        use_container_width=True,
                    )
    
    except Exception as e:
        st.warning(f"Chat summarization not available: {str(e)}")
        st.info("💡 This feature uses local processing and should work offline.")


def display_privacy_management(df):
    st.markdown('<div class="title">🔒 Privacy Management</div>', unsafe_allow_html=True)
    st.info(
        "Chat parsing, sentiment inference, authorship, and summaries run in this app "
        "process. No chat text is sent to an external LLM API. Hugging Face model "
        "weights are downloaded only after you explicitly start a model analysis."
    )
    anonymize = st.checkbox("Anonymize participant names in the export", value=True)
    mask_contacts = st.checkbox("Mask phone numbers and email addresses", value=True)
    extra_terms = st.text_area(
        "Additional sensitive terms to redact (one per line)",
        help="For example, project names or addresses. Matching is case-insensitive.",
    )
    sanitized = privacy.sanitize_conversation(
        df,
        anonymize=anonymize,
        mask_contacts=mask_contacts,
        extra_terms=[
            term.strip() for term in extra_terms.splitlines() if term.strip()
        ],
    )
    st.download_button(
        "Download sanitized chat CSV",
        data=sanitized.to_csv(index=False).encode("utf-8"),
        file_name="sanitized_whatsapp_conversation.csv",
        mime="text/csv",
        key="download_sanitized_chat",
    )
    st.warning(
        "Deleting analysis data clears the uploaded chat and in-session annotations and "
        "results. Downloaded files or browser-managed upload contents may need separate "
        "deletion by you."
    )
    st.button(
        "Delete uploaded chat and analysis data",
        key="delete_uploaded_data",
        on_click=request_uploaded_data_deletion,
    )


# —————————— MAIN ——————————
def main():
    st.markdown(
        """
        <style>
        /* Main Title */
        .custom-title {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            color: #4F46E5 !important;
            text-shadow: 2px 2px 6px rgba(79, 70, 229, 0.18) !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            margin-bottom: 0.25rem !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.6rem !important;
        }
        /* Subheader */
        .custom-subheader {
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            color: #64748B !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            margin-top: 0 !important;
            margin-bottom: 0.8rem !important;
        }
        .custom-author {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            color: #0F766E !important;
            margin-top: 0 !important;
            margin-bottom: 0.9rem !important;
            letter-spacing: 0.03em !important;
        }
        /* Divider line */
        hr.custom-divider {
            height: 4px;
            background: linear-gradient(to right, #4F46E5, #06B6D4, #14B8A6);
            margin-top: 0.3rem;
            margin-bottom: 1rem;
            border: none;
            border-radius: 2px;
        }
        </style>

        <div>
            <h1 class="custom-title">📊 WhatsApp Chat Analyzer</h1>
            <h3 class="custom-subheader">Get insights from your WhatsApp conversations</h3>
            <p class="custom-author">Charan Lawha</p>
            <hr class="custom-divider" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write('')

    # Unpack new load_sidebar signature (df, selected_user, show_analysis)
    df, selected_user, show_analysis = load_sidebar()

    if df is None:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://cdn.pixabay.com/photo/2015/08/03/13/58/whatsapp-873316_1280.png", width=100)
        with col2:
            st.info("To export a chat, open WhatsApp > Chat > More > Export Chat (without media)")
            st.markdown(
                "*Note: The existing CardiffNLP sentiment model is English-oriented. "
                "The optional XLM-RoBERTa context mode is intended for multilingual text.*"
            )

        st.markdown(
            """
            <style>
            .upload-instructions {
                background-color: #f9f9f9;
                padding: 0.5rem;
                border-left: 3px solid #27ae60;
                border-radius: 6px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin-bottom: 0.5rem;
                box-shadow: 1px 1px 2px rgba(0,0,0,0.02);
                font-size: 0.75rem;
            }
            .upload-instructions h4 {
                margin: 0 0 0.3rem 0;
                font-size: 0.85rem;
            }
            .upload-instructions ul {
                margin: 0.3rem 0;
                padding-left: 1rem;
            }
            .upload-instructions li {
                margin-bottom: 0.2rem;
            }
            .upload-instructions code {
                background: #eefaf0;
                color: #2d8659;
                padding: 0.15em 0.3em;
                border-radius: 3px;
                font-size: 0.75em;
            }
            </style>

            <div class="upload-instructions">
                <h4>📁 Upload Instructions:</h4>
                <ul>
                    <li>The uploaded file should match WhatsApp export format.</li>
                    <li>Basic lexicon sentiment is English-oriented; multilingual transformer analysis is available on demand.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # New AI Features Note
        st.success("🚀 **New AI Features Available!** After uploading your chat:")
        st.markdown("""
        - **👤 Author Prediction**: Train AI to predict who wrote messages
        - **🤖 Advanced Sentiment**: RoBERTa-powered emotion analysis  
        - **📝 Chat Summarization**: Get conversation summaries
        """)

        st.sidebar.markdown(
            """
            <style>
            .upload-card {
                background: linear-gradient(135deg,  #e6f9f0, #f0fdf6);  /* soft yellow */
                border-radius: 14px;
                padding: 12px 15px;           /* reduced padding */
                box-shadow: 0 4px 14px rgba(37, 211, 102, 0.18);
                margin-bottom: 15px;          /* less margin */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #1c2b2d;
                border: 1px solid #c4efd2;
                max-width: 280px;             /* optional: limit max width */
            }

            .upload-card h2 {
                font-size: 1.2rem;            /* smaller font size */
                font-weight: 700;
                margin: 0 0 8px 0;            /* smaller margin */
                display: flex;
                align-items: center;
                gap: 8px;
                color: #128C7E;
            }

            .upload-card p {
                font-size: 0.85rem;           /* smaller paragraph font */
                margin-bottom: 10px;
                color: #444;
            }
            </style>

            <div class="upload-card">
                <h2>🔒 Privacy First!</h2>
                <p>Your chat is processed locally in this app session.</p>
                <p>Use the Privacy Management section to sanitize an export or delete in-session analysis data.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return


    custom_css = """
    <style>
    /* Change font size and font family of tab labels */
    .stTabs button[role="tab"] {
        font-size: 18px !important;
        font-weight: bold;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #075E54; /* WhatsApp green */
        background-color: #ECE5DD; /* WhatsApp light background */
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        border: none;
        margin-right: 5px;
        transition: background-color 0.3s, color 0.3s;
    }

    /* Change active tab color */
    .stTabs button[role="tab"][aria-selected="true"] {
        color: white;
        background-color: #25D366; /* WhatsApp bright green */

    }

    /* Hover effect on tabs */
    .stTabs button[role="tab"]:hover:not([aria-selected="true"]) {
        background-color: #D6E8E0;
        cursor: pointer;
    }

    /* Increase font size for all markdown texts inside tabs */
    .stTabs div[data-testid="stMarkdownContainer"] {
        font-size: 16px !important;
        font-family: 'Arial', sans-serif;
        color: #202C33; /* WhatsApp dark text */
    }

    /* Separator styling */
    .separator {
        margin: 20px 0;
        border-bottom: 2px solid #DDDCDC;
    }
    </style>
    """


    # Only show analysis if button clicked
    if show_analysis:
        st.markdown(custom_css, unsafe_allow_html=True)        
        display_stats(selected_user, df)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Timeline & Activity", 
            "Word & Emoji Analysis",
            "Message Statistics",
            "Authorship Attribution",
            "Sentiment Analysis",
            "AI Summarization",
            "Research Dashboard",
            "Privacy Management",
        ])

        with tab1:
            st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
            plot_timeline(
                "Monthly Timeline",
                helper.monthly_timeline(selected_user, df),
                x_col='time', y_col='message',
                marker_color='#E74C3C'
            )
            plot_timeline(
                "Daily Timeline",
                helper.daily_timeline(selected_user, df),
                x_col='date', y_col='message',
                marker_color='#27AE60'
            )

            plot_activity_map(selected_user, df)
            plot_weekly_activity_heatmap(selected_user, df)           
            
            # Additional info for 'Overall' selection
            if selected_user == 'Overall':
                st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                display_busy_users(df)

            # Optionally show message or skip logic for group notifications
            elif selected_user == 'group_notification':
                st.info("Group notifications are excluded from user-specific analysis.")

        with tab2:
            st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
            display_wordcloud(selected_user, df)
            display_common_words(selected_user, df)
            display_emoji_analysis(selected_user, df)

        with tab3:
            st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
            display_avg_messages(selected_user, df)
            display_hourly_activity(selected_user, df)
            plot_day_of_month_activity(selected_user, df)
            display_message_length_dist(selected_user, df)
            display_lexical_richness(selected_user, df)
            display_sentiment_analysis(selected_user, df)
        
        with tab4:
            display_author_prediction(selected_user, df)
        
        with tab5:
            display_roberta_sentiment(selected_user, df)
            display_context_aware_sentiment(selected_user, df)
            display_emotion_recognition(selected_user, df)
        
        with tab6:
            display_chat_summarization(selected_user, df)

        with tab7:
            display_research_dashboard(df)

        with tab8:
            display_privacy_management(df)


    else:
        st.info("Upload a chat file and click 'Show Analysis' to view insights.")



if __name__ == "__main__":
    main()
