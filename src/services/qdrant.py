import uuid
from typing import Any, Dict, List
import stamina
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from enum import Enum

from src.logger import logger


class EmbeddingCategory(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    LATE_INTERACTION = "late-interaction"


class QdrantVectorDB:
    def __init__(
        self,
        url: str,
    ):
        self.url = url
        self.client: AsyncQdrantClient = None
        self.batch_size = 4
        self.timeout = 100

    @classmethod
    def create(cls, **kwargs):
        return cls(kwargs["url"])

    async def connect(self):
        logger.info(f"Connecting to Qdrant at {self.url}")
        if self.url.startswith("http://"):
            self.client = AsyncQdrantClient(self.url, timeout=self.timeout)
        else:
            # Convert URL to host and prefix format programmatically
            url_parts = self.url.replace("https://", "").split("/")
            host = "https://" + url_parts[0]
            prefix = url_parts[1] if len(url_parts) > 1 else ""
            self.client = AsyncQdrantClient(
                url=host,
                prefix=prefix,
                port=443,
                timeout=self.timeout,
                prefer_grpc=False,
            )
        # Check if connection is successful
        try:
            await self.client.get_collections()
        except Exception as e:
            logger.exception(
                f"Failed to connect to Qdrant at {self.url} with error: {e}"
            )
            raise ConnectionError(
                f"Failed to connect to Qdrant at {self.url} with error: {e}"
            )

    async def disconnect(self):
        logger.info(f"Disconnecting from Qdrant at {self.url}")
        self.client = None

    async def collection_exists(self, collection_name: str):
        return await self.client.collection_exists(collection_name=collection_name)

    async def create_collection(
        self, collection_name: str, model_configs: Dict[str, Dict]
    ):
        """
        Create a new collection with specified vector configurations.

        Args:
            collection_name: Name of the collection
            model_configs: Dictionary mapping model names to their configurations
                {
                    "model-name": {
                        "size": dimension,
                        "category": "dense"|"late-interaction"|"sparse",
                    }
                }
        """
        logger.info(f"Creating collection {collection_name} in Qdrant")

        vectors_config = {}
        sparse_vectors_config = {}
        for model_name, config in model_configs.items():
            if config["category"] == "sparse":
                sparse_vectors_config[model_name] = models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(
                        on_disk=False,
                    ),
                )
                # Do not add sparse vectors to the vectors_config
                continue

            vector_params = {
                "size": config["size"],
                "distance": models.Distance.COSINE,
                "on_disk": True,
                "datatype": models.Datatype.FLOAT16,
            }

            # Add multivector config and quantization config for late-interaction models
            if config["category"] in ["late-interaction"]:
                vector_params.pop("datatype")
                vector_params["multivector_config"] = models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
                vector_params["quantization_config"] = models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                )
            vectors_config[model_name] = models.VectorParams(**vector_params)

        await self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            replication_factor=3,
            # HNSW (Hierarchical Navigable Small World) index configuration is required to:
            # 1. Enable fast approximate nearest neighbor search in high-dimensional vector spaces
            # 2. Balance between search speed, accuracy and memory usage
            # 3. Without an index, vector search would require exhaustive comparison with every vector
            #
            # The index builds a graph structure where:
            # - m=16: Each vector connects to 16 closest neighbors, forming search paths
            # - ef_construct=128: During building, consider 128 neighbors for optimal path creation
            # - full_scan_threshold=1000: For small collections (<1000KB), use full scan instead
            #   since building an index would be unnecessary overhead
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=1000,
                max_indexing_threads=8,
            ),
            on_disk_payload=True,
        )

        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="label",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="checksum",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    async def delete_collection(self, collection_name: str):
        await self.client.delete_collection(collection_name=collection_name)

    @stamina.retry(on=Exception, attempts=3)
    async def add_documents(
        self, collection_name: str, documents: List[Dict[str, Any]]
    ):
        """Add documents to the vector database. Each document is a dict with keys: vector, metadata"""
        try:
            logger.info(f"Adding {len(documents)} documents to Qdrant")
            points = []
            for idx, doc in enumerate(documents):
                logger.info(f"Adding document {idx}")
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id, vector=doc["vector"], payload=doc["metadata"]
                    )
                )

                # When we reach batch_size or it's the last batch, upload points
                if len(points) >= self.batch_size or idx == len(documents) - 1:
                    await self.client.upsert(
                        collection_name=collection_name,
                        points=points,
                        wait=False,
                    )
                    points = []  # Clear points for next batch

        except Exception as e:
            logger.exception(f"Failed to add documents to Qdrant: {e}")
            raise e

    async def delete_documents(self, collection_name: str, labels: List[str]):
        """Delete documents from the vector database. Each document is a dict with keys: vector, metadata. Used to delete documents by label."""
        filter_condition = models.Filter(
            must=[models.FieldCondition(key="label", match=models.MatchAny(any=labels))]
        )
        await self.client.delete(
            collection_name=collection_name,
            points_selector=filter_condition,
            wait=False,
        )

    async def generate_prefetch_vectors(self, embeddings_dict: Dict, limit: int):
        """Generate prefetch vectors for dense and sparse embeddings.

        Args:
            embeddings_dict (Dict): Dictionary containing embeddings with their categories and model names.
                Expected format:
                {
                    "model_name": {
                        "category": EmbeddingCategory,
                        "embedding": vector/sparse_vector
                    },
                    ...
                }
            limit (int): Number of results to return per vector.

        Returns:
            List[models.Prefetch]: List of prefetch vectors for querying.
        """
        prefetch_vectors = []
        for model_name, embedding_obj in embeddings_dict.items():
            if embedding_obj["category"] == EmbeddingCategory.DENSE:
                prefetch_vectors.append(
                    models.Prefetch(
                        query=embedding_obj["embedding"], using=model_name, limit=limit
                    )
                )
            elif embedding_obj["category"] == EmbeddingCategory.SPARSE:
                prefetch_vectors.append(
                    models.Prefetch(
                        query=models.SparseVector(**embedding_obj["embedding"]),
                        using=model_name,
                        limit=limit,
                    )
                )
        return prefetch_vectors

    async def generate_late_query_vector(self, embeddings_dict: Dict):
        """Generate a late query vector for a given embedding model.

        Args:
            embeddings_dict (Dict): Dictionary containing embeddings with their categories and model names.
                Expected format:
                {
                    "model_name": {
                        "category": EmbeddingCategory,
                        "embedding": vector/sparse_vector
                    },
                    ...
                }
        """
        late_vector = None
        for model_name, embedding_obj in embeddings_dict.items():
            if embedding_obj["category"] in [
                EmbeddingCategory.LATE_INTERACTION,
            ]:
                late_vector = {
                    "model": model_name,
                    "embedding": embedding_obj["embedding"],
                }
                break
        return late_vector

    @stamina.retry(on=Exception, attempts=3)
    async def search(
        self,
        collection_name: str,
        embeddings_dict: Dict,
        limit: int = 10,
    ):
        """Search for the nearest neighbors in the vector database.

        Args:
            collection_name (str): Name of the collection to search in.
            embeddings_dict (Dict): Dictionary containing embeddings with their categories and model names.
            limit (int): Number of results to return.
        """
        try:
            await self.connect()
            # Generate prefetch and late query vectors for 2x limit
            prefetch_vectors = await self.generate_prefetch_vectors(
                embeddings_dict, limit * 2
            )
            late_vector = await self.generate_late_query_vector(embeddings_dict)
            results = await self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_vectors,
                query=late_vector["embedding"],
                using=late_vector["model"],
                limit=5,  # Get top 5 results
                timeout=self.timeout,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
            )
            return results.points
        except Exception as e:
            logger.exception(f"Failed to search in Qdrant: {e}")
            raise e

    async def document_exists(self, collection_name: str, checksum: str) -> bool:
        """Check if a document with the given checksum exists in the collection.

        Args:
            collection_name (str): Name of the collection
            checksum (str): Checksum of the document

        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="checksum", match=models.MatchValue(value=checksum)
                    )
                ]
            )
            response = await self.client.scroll(
                collection_name=collection_name, scroll_filter=filter_condition, limit=1
            )
            return len(response[0]) > 0
        except Exception as e:
            logger.exception(f"Failed to check document existence: {e}")
            raise e
