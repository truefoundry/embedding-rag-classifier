from typing import List, Dict
import os
import tempfile
import argparse
import asyncio
import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib

from src.services.qdrant import QdrantVectorDB, EmbeddingCategory
from src.services.infinity import InfinityEmbeddings
from src.logger import logger
from src.config import settings


def create_infinity_client(url: str, api_key: str):
    """Create an InfinityEmbeddings client for generating embeddings.

    Args:
        url (str): URL of the Infinity service.
        api_key (str): API key for the Infinity service.
    Returns:
        InfinityEmbeddings: Client for generating embeddings from the Infinity service.
    """
    return InfinityEmbeddings(url=url, api_key=api_key)


async def create_qdrant_client(qdrant_url: str):
    """Create a Qdrant client for indexing documents.

    Args:
        qdrant_url: Url of Qdrant instance
    Returns:
        QdrantVectorDB: Client for indexing documents in Qdrant.
    """

    qdrant_vector_db = QdrantVectorDB.create(url=qdrant_url)
    await qdrant_vector_db.connect()
    return qdrant_vector_db


async def generate_embeddings(
    texts: List[str],
    infinity_client: InfinityEmbeddings,
    embedding_model: str,
) -> np.ndarray | List[Dict[str, np.ndarray]]:
    """Generate embeddings for a list of texts.

    Args:
        texts (List[str]): List of texts to generate embeddings for.
        infinity_client (InfinityEmbeddings): Client for generating embeddings from the Infinity service.
        embedding_model (str): Name of the embedding model to use.

    Returns:
        np.ndarray | List[Dict[str, np.ndarray]]: Embeddings for the texts.
    """

    embeddings = await asyncio.gather(
        *[infinity_client.get_embeddings(text, embedding_model) for text in texts]
    )
    embeddings = np.array(embeddings)
    return embeddings[0] if len(texts) == 1 else embeddings


async def create_collection(
    qdrant_vector_db: QdrantVectorDB,
    infinity_client: InfinityEmbeddings,
    collection: str,
    dense_model: str,
    sparse_model: str,
    late_interaction_model: str,
):
    """Create a Qdrant collection with the specified embedding models.

    Args:
        qdrant_vector_db (QdrantVectorDB): Client for indexing documents in Qdrant.
        infinity_client (InfinityEmbeddings): Client for generating embeddings from the Infinity service.
        collection (str): Name of the collection to create.
        dense_model (str): Name of the dense embedding model to use.
        sparse_model (str): Name of the sparse embedding model to use.
        late_interaction_model (str): Name of the late interaction embedding model to use.
    """
    try:
        # Get dimensions and configuration for each embedding model
        model_configs = {}
        if dense_model:
            embeddings = await generate_embeddings(
                ["test"], infinity_client, dense_model
            )
            model_configs[dense_model] = {
                "size": embeddings.shape[-1],
                "category": EmbeddingCategory.DENSE,
            }
        if sparse_model:
            model_configs[sparse_model] = {
                "size": None,
                "category": EmbeddingCategory.SPARSE,
            }
        if late_interaction_model:
            embeddings = await generate_embeddings(
                ["test"], infinity_client, late_interaction_model
            )
            model_configs[late_interaction_model] = {
                "size": embeddings.shape[-1],
                "category": EmbeddingCategory.LATE_INTERACTION,
            }

        await qdrant_vector_db.create_collection(
            collection_name=collection,
            model_configs=model_configs,
        )

        logger.info(
            f"Collection {collection} created with models: {model_configs.keys()}"
        )
    except Exception as e:
        logger.error(f"Error creating collection {collection}: {e}")
        raise e


async def index_documents(
    documents: List[List[str]],
    collection: str,
    dense_model: str,
    sparse_model: str,
    late_interaction_model: str,
    vector_db: QdrantVectorDB,
    infinity_client: InfinityEmbeddings,
    update: bool,
):
    """Index documents in Qdrant.

    Args:
        documents (List[List[str]]): List of documents to index.
        collection (str): Name of the collection to index documents in.
        dense_model (str): Name of the dense embedding model to use.
        sparse_model (str): Name of the sparse embedding model to use.
        late_interaction_model (str): Name of the late interaction embedding model to use.
        vector_db (QdrantVectorDB): Client for indexing documents in Qdrant.
        infinity_client (InfinityEmbeddings): Client for generating embeddings from the Infinity service.
    """
    document_chunks = []
    for document in documents:
        text, label, checksum = document

        # Skip if document already exists and update mode is True
        if update:
            exists = await vector_db.document_exists(collection, checksum)
            if exists:
                logger.info(
                    f"Skipping document with checksum {checksum} as it already exists"
                )
                continue

        document_chunk = {}
        vector_dict = {}
        if dense_model:
            embeddings = await generate_embeddings([text], infinity_client, dense_model)
            vector_dict[dense_model] = embeddings
        if sparse_model:
            vector_dict[sparse_model] = await infinity_client.get_embeddings(
                text, sparse_model
            )
        if late_interaction_model:
            vector_dict[late_interaction_model] = await infinity_client.get_embeddings(
                text, late_interaction_model
            )

        document_chunk["label"] = label
        document_chunk["text"] = text
        document_chunk["checksum"] = checksum
        document_chunk["metadata"] = {
            "label": label,
            "text": text,
            "checksum": checksum,
        }
        document_chunk["vector"] = vector_dict
        document_chunks.append(document_chunk)

    if document_chunks:  # Only call add_documents if there are documents to add
        await vector_db.add_documents(collection, document_chunks)


def load_csv_from_artifact(artifact_fqn: str, temp_dir: str):
    from truefoundry.ml import get_client

    client = get_client()
    artifact_version = client.get_artifact_version_by_fqn(fqn=artifact_fqn)

    # download the artifact contents to disk
    download_info = artifact_version.download(path=temp_dir)
    return download_info


async def main(
    csv_path: str,
    collection: str,
    dense_model: str,
    sparse_model: str = "Qdrant/bm25",
    late_interaction_model: str = "colbert-ir/colbertv2.0",
    delete_existing: bool = False,
    batch_size: int = 4,
    update: bool = False,
):
    # if csv_path begins with `artifact:` load from artifact
    if csv_path.startswith("artifact:"):
        # Create a temp directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Temp directory created at {temp_dir}")
        downloaded_dir = load_csv_from_artifact(csv_path, temp_dir)
        logger.info(f"CSV loaded from artifact to {downloaded_dir}")
        # Iterater over the directory and load all csv files
        for file in os.listdir(downloaded_dir):
            if file.endswith(".csv"):
                csv_path = os.path.join(downloaded_dir, file)
                logger.info(f"Loading CSV from {csv_path}")
                break

    # Create Infinity client
    infinity_client = create_infinity_client(
        url=settings.INFINITY_URL, api_key=settings.INFINITY_API_KEY
    )

    # Create Qdrant client
    qdrant_vector_db = await create_qdrant_client(qdrant_url=settings.QDRANT_URL)

    if delete_existing:
        try:
            await qdrant_vector_db.delete_collection(collection)
            logger.info(f"Deleting existing collection {collection} before indexing")
        except Exception as e:
            logger.error(f"Error deleting collection {collection}: {e}")

    if not update:
        try:
            # Check if collection exists, if so raise error
            if await qdrant_vector_db.collection_exists(collection):
                raise ValueError(
                    f"Collection {collection} already exists, delete it before indexing using --delete-existing flag or use --update flag to update the existing collection"
                )
        except Exception as e:
            logger.error(f"Error checking if collection {collection} exists: {e}")

        # Create collection
        await create_collection(
            qdrant_vector_db=qdrant_vector_db,
            infinity_client=infinity_client,
            collection=collection,
            dense_model=dense_model,
            sparse_model=sparse_model,
            late_interaction_model=late_interaction_model,
        )
        logger.info(f"Collection {collection} created successfully")

    # Iterate over dataframe in batches of 4
    dataset = pd.read_csv(csv_path)
    text_batches = [
        [
            [
                row["text"],
                row["label"],
                hashlib.sha256(row["text"].encode()).hexdigest(),
            ]
            for _, row in dataset.iloc[i : i + batch_size].iterrows()
        ]
        for i in range(0, len(dataset), batch_size)
    ]

    for batch in tqdm(text_batches, desc=f"Generating embeddings for {collection}"):
        await index_documents(
            documents=batch,
            collection=collection,
            dense_model=dense_model,
            sparse_model=sparse_model,
            late_interaction_model=late_interaction_model,
            vector_db=qdrant_vector_db,
            infinity_client=infinity_client,
            update=update,
        )
    logger.info(f"Collection {collection} indexed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./data/dataset.csv",
        help="Path to the CSV file to index or artifact fqn",
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Name of the collection to index documents in",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="truefoundry/setfit-mxbai-embed-xsmall-v1-ivr-classifier",
        help="Dense embedding model to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of documents to process at once",
    )
    parser.add_argument(
        "--delete-existing", action="store_true", help="Delete existing collection"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing collection with new data",
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
