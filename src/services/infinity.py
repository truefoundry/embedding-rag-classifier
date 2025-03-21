import logging
from typing import List

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fastembed.sparse.bm25 import Bm25
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class InfinityEmbeddings:
    def __init__(self, url: str, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=url)
        # Initialize BM25 model
        self.bm25 = Bm25("Qdrant/bm25")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, aiohttp.ServerTimeoutError)
        ),
        reraise=True,
    )
    async def get_embeddings(
        self, text: str, embedding_model: str
    ) -> List[List[float]]:
        try:
            if embedding_model == "Qdrant/bm25":
                sparse_vector = next(self.bm25.embed(text))
                return sparse_vector.as_object()
            else:
                response = await self.client.embeddings.create(
                    input=text, model=embedding_model, encoding_format="float"
                )
                return response.data[0].embedding
        except aiohttp.ClientError as e:
            logger.error(f"Network error while getting embeddings: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error while getting embeddings: {str(e)}")
            raise

    async def get_query_embedding(
        self, query: str, embedding_model: str
    ) -> List[float]:
        try:
            embedding = await self.get_embeddings([query], embedding_model)
            return embedding[0]
        except Exception as e:
            logger.error(f"Error while getting query embedding: {str(e)}")
            raise
