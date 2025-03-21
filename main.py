import warnings

warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
from tempfile import NamedTemporaryFile
import logging
import asyncio

# Import the main functions
from src.contrastive_learning import main as train_main
from src.plot_embedding_map import main as plot_main

# StreamlitHandler class for logging
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


# Add at the beginning of the file, after the imports
if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False

# Page configuration
st.set_page_config(
    page_title="Truefoundry Classifier - Train & Visualize & Evaluate",
    page_icon="🤖",
    layout="wide",
)

st.title("Truefoundry Classifier - Train & Visualize & Evaluate")

# Create a more prominent navigation with radio buttons
st.markdown(
    """
    <style>
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    div.row-widget.stRadio > div > label {
        font-size: 24px !important;
        padding: 10px 30px !important;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-right: 10px;
    }
    div.row-widget.stRadio > div > label:hover {
        background-color: #e0e2e6;
    }
    div.row-widget.stRadio > div [data-baseweb="radio"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use radio buttons for tab selection
active_tab = st.radio("", ["Train Model", "Visualize Embeddings"], horizontal=True)


def render_training_sidebar():
    with st.sidebar:
        st.header("Training Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV dataset", type="csv", key="train_upload"
        )
        use_default_csv = st.checkbox(
            "Use default dataset", value=not bool(uploaded_file), key="train_default"
        )

        if use_default_csv:
            input_csv = "./data/dataset.csv"
            if not os.path.exists(input_csv):
                st.error(f"Default dataset not found at {input_csv}")
                st.stop()
            st.success(f"Using default dataset: {input_csv}")
        elif uploaded_file:
            with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_csv = tmp_file.name
        else:
            st.warning("Please upload a CSV file or use the default dataset")
            st.stop()

        # Model selection
        model_name = st.text_input(
            "Model name (Hugging Face repo)",
            value="mixedbread-ai/mxbai-embed-xsmall-v1",
        )

        # Training parameters
        train_samples = st.slider(
            "Train samples per class", min_value=4, max_value=16, value=8, step=1
        )

        test_samples = st.slider(
            "Test samples per class", min_value=4, max_value=16, value=8, step=1
        )

        batch_size = st.slider(
            "Batch size", min_value=16, max_value=4096, value=32, step=1
        )

        epochs = st.slider(
            "Number of epochs", min_value=1, max_value=5, value=1, step=1
        )

        model_suffix = st.text_input(
            "Model suffix (required)",
            value="",
            help="Please enter a suffix to identify this training run (e.g., 'test-run-1')",
        )

        st.markdown("---")

        train_button = st.button(
            "Train Model",
            disabled=st.session_state.training_in_progress or not model_suffix.strip(),
            help="Model suffix is required to start training"
            if not model_suffix.strip()
            else "Training in progress..."
            if st.session_state.training_in_progress
            else "Start training",
        )

        return (
            input_csv,
            model_name,
            train_samples,
            test_samples,
            batch_size,
            epochs,
            model_suffix,
            train_button,
        )


def render_visualization_sidebar():
    with st.sidebar:
        st.header("Visualization Configuration")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV dataset", type="csv", key="viz_upload"
        )
        use_default_csv = st.checkbox(
            "Use default dataset", value=not bool(uploaded_file), key="viz_default"
        )

        if use_default_csv:
            input_csv = "./data/dataset.csv"
            if not os.path.exists(input_csv):
                st.error(f"Default dataset not found at {input_csv}")
                st.stop()
            st.success(f"Using default dataset: {input_csv}")
        elif uploaded_file:
            with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_csv = tmp_file.name
        else:
            st.warning("Please upload a CSV file or use the default dataset")
            st.stop()

        # Parameters
        batch_size = st.slider(
            "Batch size for embedding generation", min_value=1, max_value=32, value=4
        )

        num_samples = st.slider(
            "Number of samples per class",
            min_value=-1,
            max_value=50,
            value=10,
            help="-1 means use all samples",
        )

        # Default models
        default_models = [
            "mixedbread-ai/mxbai-embed-xsmall-v1",
            "truefoundry/setfit-mxbai-embed-xsmall-v1-ivr-classifier",
        ]

        if "models" not in st.session_state:
            st.session_state.models = default_models.copy()

        st.subheader("Models to Compare")

        for i, model in enumerate(st.session_state.models):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.session_state.models[i] = st.text_input(
                    f"Model {i+1}", value=model, key=f"model_{i}"
                )
            with col2:
                if (
                    st.button("Remove", key=f"remove_{i}")
                    and len(st.session_state.models) > 1
                ):
                    st.session_state.models.pop(i)
                    st.rerun()

        if st.button("Add Model"):
            st.session_state.models.append("")
            st.rerun()

        st.markdown("---")

        viz_button = st.button("Generate Visualizations")

        return input_csv, batch_size, num_samples, viz_button


# Clear the sidebar before rendering the active tab's sidebar
st.sidebar.empty()

# Render content based on active tab
if active_tab == "Train Model":
    # Training Tab Content
    st.write(
        "Train a sentence-transformer model using contrastive learning for text classification"
    )

    # Get training parameters from sidebar
    (
        input_csv,
        model_name,
        train_samples,
        test_samples,
        batch_size,
        epochs,
        model_suffix,
        train_button,
    ) = render_training_sidebar()

    # Display dataset preview and statistics
    try:
        df = pd.read_csv(input_csv)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Statistics")
        st.write(f"Total examples: {len(df)}")
        unique_labels = df["label"].unique()
        st.write(f"Number of unique labels: {len(unique_labels)}")

        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        st.bar_chart(label_counts.set_index("Label"))

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

    # Training section
    st.subheader("Training")

    if train_button:
        st.session_state.training_in_progress = True

        try:
            status_text = st.empty()
            progress_bar = st.progress(0)
            metrics_container = st.empty()
            progress_chart = st.empty()
            loss_values = []

            def progress_callback(stats):
                try:
                    if "embedding_loss" in stats:
                        loss = stats["embedding_loss"]
                        loss_values.append(loss)
                        df = pd.DataFrame({"loss": loss_values})
                        progress_chart.line_chart(df)
                        metrics_container.json(stats)
                except Exception as e:
                    logger.warning(f"Error in progress callback: {str(e)}")

            # Set up logging
            from src.logger import logger, formatter

            logger.handlers.clear()
            streamlit_handler = StreamlitHandler()
            streamlit_handler.setFormatter(formatter)
            logger.addHandler(streamlit_handler)

            status_text.text("Starting training process...")

            with st.spinner("Training in progress... This might take a while."):
                train_main(
                    input_csv=input_csv,
                    model_name=model_name,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    batch_size=batch_size,
                    epochs=epochs,
                    model_suffix=model_suffix,
                    progress_callback=progress_callback,
                )

            progress_bar.progress(100)
            status_text.text("Training completed!")
            st.success("Model training completed successfully!")
            st.session_state.training_in_progress = False

        except Exception as e:
            st.session_state.training_in_progress = False
            st.error(f"Error during training: {str(e)}")
else:  # Visualize Embeddings tab
    # Visualization Tab Content
    st.write("Visualize embeddings from different models using UMAP projection")

    # Get visualization parameters from sidebar
    input_csv, batch_size, num_samples, viz_button = render_visualization_sidebar()

    # Generate visualizations when button is clicked
    if viz_button:
        with st.spinner("Generating embeddings and visualizations..."):
            try:
                figures = asyncio.run(
                    plot_main(
                        dataset_path=input_csv,
                        batch_size=batch_size,
                        num_samples=num_samples,
                        models=st.session_state.models,
                    )
                )

                for model_name, fig in figures.items():
                    st.subheader(f"Visualization for {model_name}")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
