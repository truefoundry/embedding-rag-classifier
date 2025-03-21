import warnings

warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
from tempfile import NamedTemporaryFile
import logging

# Import the main function from contrastive_learning.py
from src.contrastive_learning import main

# Add at the beginning of the file, after the imports
if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False

# Page configuration
st.set_page_config(
    page_title="Truefoundry Embedding Model Trainer", page_icon="🤖", layout="wide"
)

st.title("Truefoundry Embedding Model Trainer")
st.write(
    "Train a sentence-transformer model using contrastive learning for text classification"
)

# Sidebar for inputs
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type="csv")
use_default_csv = st.sidebar.checkbox(
    "Use default dataset", value=not bool(uploaded_file)
)

if use_default_csv:
    input_csv = "./data/dataset.csv"
    if os.path.exists(input_csv):
        st.sidebar.success(f"Using default dataset: {input_csv}")
    else:
        st.sidebar.error(f"Default dataset not found at {input_csv}")
        st.stop()
elif uploaded_file:
    # Save uploaded file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        input_csv = tmp_file.name
else:
    st.sidebar.warning("Please upload a CSV file or use the default dataset")
    st.stop()

# Model selection
model_name = st.sidebar.text_input(
    "Model name (Hugging Face repo)", value="mixedbread-ai/mxbai-embed-xsmall-v1"
)

# Training parameters with sliders
train_samples = st.sidebar.slider(
    "Train samples per class", min_value=4, max_value=16, value=8, step=1
)

test_samples = st.sidebar.slider(
    "Test samples per class", min_value=4, max_value=16, value=8, step=1
)

batch_size = st.sidebar.slider(
    "Batch size", min_value=16, max_value=4096, value=32, step=1
)

epochs = st.sidebar.slider(
    "Number of epochs", min_value=1, max_value=5, value=1, step=1
)

# Model suffix at the end of sidebar
model_suffix = st.sidebar.text_input(
    "Model suffix (required)",
    value="",
    help="Please enter a suffix to identify this training run (e.g., 'test-run-1')",
)

# Add a separator in sidebar
st.sidebar.markdown("---")

# Modify the train button section
train_button = st.sidebar.button(
    "Train Model",
    disabled=st.session_state.training_in_progress or not model_suffix.strip(),
    help="Model suffix is required to start training"
    if not model_suffix.strip()
    else "Training in progress..."
    if st.session_state.training_in_progress
    else "Start training",
)

if not model_suffix.strip():
    st.sidebar.warning("Please enter a model suffix to enable training")
elif st.session_state.training_in_progress:
    st.sidebar.warning("Training in progress... Please wait.")

# Display dataset preview
try:
    df = pd.read_csv(input_csv)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # # Show unique labels
    unique_labels = df["label"].unique()
    # st.subheader("Unique Labels")
    # st.write(unique_labels)

    # Show dataset statistics
    st.subheader("Dataset Statistics")
    st.write(f"Total examples: {len(df)}")
    st.write(f"Number of unique labels: {len(unique_labels)}")

    # Label distribution
    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    st.bar_chart(label_counts.set_index("Label"))

except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

# Training section
st.subheader("Training")


class StreamlitHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            if level == "ERROR":
                st.error(msg)
            elif level == "WARNING":
                st.warning(msg)
            elif level == "INFO":
                st.info(msg)
            else:
                st.text(msg)
        except Exception as e:
            st.error(f"Error in StreamlitHandler: {str(e)}")


if train_button:
    st.session_state.training_in_progress = True

    try:
        # Create a placeholder for the status
        status_text = st.empty()
        progress_bar = st.progress(0)

        # Create placeholders for metrics
        metrics_container = st.empty()
        progress_chart = st.empty()
        loss_values = []

        def progress_callback(stats):
            try:
                if "embedding_loss" in stats:
                    loss = stats["embedding_loss"]
                    loss_values.append(loss)
                    # Create a DataFrame for the chart
                    df = pd.DataFrame({"loss": loss_values})
                    progress_chart.line_chart(df)
                    metrics_container.json(stats)
            except Exception as e:
                logger.warning(f"Error in progress callback: {str(e)}")

        # Set up logging before training
        from src.logger import logger, formatter

        # Clear existing handlers
        logger.handlers.clear()

        # Add our streamlit handler
        streamlit_handler = StreamlitHandler()
        streamlit_handler.setFormatter(
            formatter
        )  # Use the same formatter from logger.py
        logger.addHandler(streamlit_handler)

        # Set up a progress indicator
        status_text.text("Starting training process...")

        # Create a Streamlit spinner while training
        with st.spinner("Training in progress... This might take a while."):
            # Call the main function with callback
            main(
                input_csv=input_csv,
                model_name=model_name,
                train_samples=train_samples,
                test_samples=test_samples,
                batch_size=batch_size,
                epochs=epochs,
                model_suffix=model_suffix,
                progress_callback=progress_callback,  # Pass the callback
            )

        # Update the progress
        progress_bar.progress(100)
        status_text.text("Training completed!")

        # Show success message
        st.success("Model training completed successfully!")

        # After training completes (success or failure), reset the state
        st.session_state.training_in_progress = False

    except Exception as e:
        # Make sure to reset the state even if there's an error
        st.session_state.training_in_progress = False
        st.error(f"Error during training: {str(e)}")
