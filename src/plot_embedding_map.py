from typing import List
import asyncio
import os
import argparse
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from src.services.infinity import InfinityEmbeddings
from tqdm.asyncio import tqdm

from src.config import settings
from src.logger import logger


async def generate_embeddings(
    texts: List[str], infinity_client: InfinityEmbeddings, model: str
):
    embeddings = await asyncio.gather(
        *[infinity_client.get_embeddings(text, model) for text in texts]
    )
    embeddings = np.array(embeddings)
    return embeddings[0] if len(texts) == 1 else embeddings


def create_infinity_client():
    return InfinityEmbeddings(
        url=settings.INFINITY_URL, api_key=settings.INFINITY_API_KEY
    )


def sample_dataset(dataset: pd.DataFrame, num_samples: int) -> pd.DataFrame:
    # Always group by label to maintain consistent behavior
    grouped_dataset = dataset.groupby("label", as_index=False)

    if num_samples == -1:
        # Return all samples but maintain grouping structure
        return grouped_dataset.apply(lambda x: x[["text", "label"]]).reset_index(
            drop=True
        )

    # Sample n items from each group
    return grouped_dataset.apply(
        lambda x: x[["text", "label"]].sample(n=min(len(x), num_samples))
    ).reset_index(drop=True)


async def process_dataset(
    dataset: pd.DataFrame, batch_size: int, infinity_client, models: List[str]
) -> pd.DataFrame:
    text_batches = [
        [row["text"] for _, row in dataset.iloc[i : i + batch_size].iterrows()]
        for i in range(0, len(dataset), batch_size)
    ]

    for model in models:
        all_embeddings = []
        for batch in tqdm(text_batches, desc=f"Generating embeddings for {model}"):
            embeddings = await generate_embeddings(batch, infinity_client, model)
            all_embeddings.extend(embeddings)

        dataset[f"embedding_{model.split('/')[-1]}"] = all_embeddings

    return dataset


def generate_projections(df: pd.DataFrame, results_dir: str, save_binary):
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    projection_dir = os.path.join(results_dir, f"projections-{timestamp}")
    os.makedirs(projection_dir, exist_ok=True)

    # Get all embedding columns
    embedding_cols = [col for col in df.columns if col.startswith("embedding_")]

    for embedding_col in embedding_cols:
        # Convert embeddings list to numpy array
        X = np.array(df[embedding_col].tolist())

        # Perform UMAP
        reducer = UMAP(random_state=42)
        X_umap = reducer.fit_transform(X)

        # Create a dataframe with the UMAP components and metadata
        plot_df = pd.DataFrame(
            {
                "UMAP1": X_umap[:, 0],
                "UMAP2": X_umap[:, 1],
                "Label": df["label"],
                "Text": df["text"],  # Original text for hover information
            }
        )

        # Create interactive plot
        fig = px.scatter(
            plot_df,
            x="UMAP1",
            y="UMAP2",
            color="Label",
            hover_data=["Text", "Label"],
            title=f"Interactive UMAP of Text Embeddings by Class - {embedding_col}",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )

        # Update layout for better visualization
        fig.update_layout(
            width=1200,
            height=800,
            title={"y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            legend_title_text="Class",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        )

        # Update traces for better visibility
        fig.update_traces(
            marker=dict(size=8),
            hovertemplate="<br>".join(
                ["Text: %{customdata[0]}", "Label: %{customdata[1]}", "<extra></extra>"]
            ),
        )

        # Save as HTML (interactive)
        output_file = os.path.join(projection_dir, f"{embedding_col}_projection.html")
        fig.write_html(output_file)

    # Save the entire dataframe if requested
    if save_binary:
        # Save the processed dataframe with embeddings
        output_file = os.path.join(projection_dir, "embeddings.pkl")
        df.to_pickle(output_file)


async def main(
    batch_size: int,
    use_tf: bool,
    num_samples: int,
    models: List[str],
    results_dir: str,
    save_binary: bool,
):
    # Load dataset
    dataset = pd.read_csv(os.path.abspath("./data/complete_dataset.csv"))

    # Initialize client
    infinity_client = create_infinity_client(use_tf)

    # Sample and process dataset
    sampled_dataset = sample_dataset(dataset, num_samples)
    processed_dataset = await process_dataset(
        sampled_dataset, batch_size, infinity_client, models
    )

    # Save results
    generate_projections(processed_dataset, results_dir, save_binary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings from text dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for processing texts"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples per label. Use -1 for all samples",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mixedbread-ai/mxbai-embed-xsmall-v1",
            "truefoundry/setfit-mxbai-embed-xsmall-v1-ivr-classifier",
        ],
        help="List of embedding models to use",
    )
    parser.add_argument(
        "--save-binary",
        action="store_true",
        help="Store binary of generated embeddings",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./eval_results",
        help="Directory to store results",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            batch_size=args.batch_size,
            use_tf=args.use_tf,
            num_samples=args.num_samples,
            models=args.models,
            results_dir=args.results_dir,
            save_binary=args.save_binary,
        )
    )
