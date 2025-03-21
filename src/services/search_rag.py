from typing import List, Dict
import argparse
import asyncio
import numpy as np
from openai import AsyncOpenAI
from utils import QdrantVectorDB, EmbeddingCategory, InfinityEmbeddings
from src.common.service.logging.cvs_logger import info, error, signal
from src.common.config.app_config import get_application_config
from src.inference.constants import MAPPINGS, LLM_SUB_INTENT_PROMPT, INTENT_DESCRIPTION
import re
from src.common.service.prompt.prompt_service import PromptService, PromptDetail

CONFIG = get_application_config()


class RAGSearchService:
    _instance = None  # Class variable to hold the singleton instance

    @classmethod
    async def get_instance(
        cls,
        collection: str,
        dense_model: str,
        sparse_model: str,
        late_interaction_model: str,
        use_tf: bool = False,
        use_local: bool = False,
    ):
        """Get the singleton instance of SearchService."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.collection = collection
            cls._instance.dense_model = dense_model
            cls._instance.sparse_model = sparse_model
            cls._instance.late_interaction_model = late_interaction_model
            cls._instance.use_tf = use_tf
            cls._instance.use_local = use_local
            cls._instance.intent_prompt_id = CONFIG.get(
                "prompts.rag-llm-intent-classifier"
            )

            # Initialize clients immediately
            cls._instance.infinity_client = cls._instance._create_infinity_client()
            cls._instance.qdrant_vector_db = await cls._instance._create_qdrant_client()
            cls._instance.llm_client = await cls._instance._create_llm_client()
        return cls._instance

    def _create_infinity_client(self):
        """Create an InfinityEmbeddings client for generating embeddings."""
        url = (
            CONFIG.APP.get("client", {}).get("infinity_tfy", {}).get("base_url")
            if self.use_tf
            else CONFIG.APP.get("client", {}).get("infinity", {}).get("base_url")
        )
        api_key = CONFIG.APP.get("client", {}).get("infinity", {}).get("api_key")
        return InfinityEmbeddings(url=url, api_key=api_key)

    async def _create_qdrant_client(self):
        """Create a Qdrant client for indexing documents."""
        url = (
            CONFIG.APP.get("client", {}).get("qdrant_local", {}).get("base_url")
            if self.use_local
            else CONFIG.APP.get("client", {}).get("qdrant", {}).get("base_url")
        )
        qdrant_vector_db = QdrantVectorDB.create(url=url)
        await qdrant_vector_db.connect()
        return qdrant_vector_db

    async def _create_llm_client(self):
        """Create an OpenAI client for generating embeddings."""

        url = (
            CONFIG.APP.get("llm", {}).get("hermes_tfy", {}).get("base_url")
            if self.use_tf
            else CONFIG.APP.get("llm", {}).get("hermes", {}).get("base_url")
        )
        self.llm_model_name = (
            CONFIG.APP.get("llm", {}).get("hermes_tfy", {}).get("model_name")
            if self.use_tf
            else CONFIG.APP.get("llm", {}).get("hermes", {}).get("model_name")
        )
        api_key = (
            CONFIG.APP.get("llm", {}).get("hermes_tfy", {}).get("api_key")
            if self.use_tf
            else CONFIG.APP.get("llm", {}).get("hermes", {}).get("api_key")
        )
        client = AsyncOpenAI(base_url=url, api_key=api_key)
        # Test the connection
        try:
            await client.models.list()
            info("Successfully connected to LLM service")
        except Exception as e:
            error(f"Failed to connect to LLM service: {e}")
            raise

        return client

    async def get_sub_intent_from_llm(
        self, query: str, top_five: List[Dict[str, str]], multi_intent: bool = False
    ) -> str:
        """Get the sub-intent from the LLM."""
        # Convert top_five to a string
        top_five_str = "\n\n".join(
            [
                f"{i+1}. Query: {item['query']}\n   Intent: {item['intent']}"
                for i, item in enumerate(top_five)
            ]
        )

        prompt_detail: PromptDetail = await PromptService.get_prompt_template(
            self.intent_prompt_id,
            {
                "query": query,
                "top_five": top_five_str,
                # "multi_intent": str(multi_intent),
            },
        )

        prompt = prompt_detail.generatedPrompt
        # signal(f"Generated Prompt: {prompt}")
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Add temperature for some variability
        )
        intent_message: str = response.choices[0].message.content.lower()
        intent_message = intent_message.strip()

        signal(f"LLM response: {intent_message}")
        if len(intent_message) == 0:
            return "unknown"
        else:
            # In case intent message is a string, check if any of the keys in MAPPINGS are present in the intent message
            keys = []
            for key in MAPPINGS.keys():
                if key in intent_message:
                    keys.append(key)
            if keys:
                return ", ".join(keys)
            return "unknown"

    async def _search_single(
        self, query: str, limit: int = 10, multi: bool = False
    ) -> Dict:
        """Base search function for a single query."""
        embeddings_dict = {}

        # Get all embeddings in parallel
        dense_embedding, sparse_embedding, late_embedding = await asyncio.gather(
            self.infinity_client.get_embeddings(query, self.dense_model),
            self.infinity_client.get_embeddings(query, self.sparse_model),
            self.infinity_client.get_embeddings(query, self.late_interaction_model),
        )

        embeddings_dict[self.dense_model] = {
            "embedding": dense_embedding,
            "category": EmbeddingCategory.DENSE,
        }

        embeddings_dict[self.sparse_model] = {
            "embedding": sparse_embedding,
            "category": EmbeddingCategory.SPARSE,
        }

        embeddings_dict[self.late_interaction_model] = {
            "embedding": late_embedding,
            "category": EmbeddingCategory.LATE_INTERACTION,
        }

        results = await self.qdrant_vector_db.search(
            collection_name=self.collection,
            embeddings_dict=embeddings_dict,
            limit=limit,
        )

        # Filter out duplicate intents while preserving order
        # seen_intents = set()
        # top_five = []
        # for result in results:
        #     intent = result.payload["label"]
        #     if intent not in seen_intents:
        #         seen_intents.add(intent)
        #         top_five.append({"query": result.payload["text"], "intent": intent})

        top_five = [
            {"query": result.payload["text"], "intent": result.payload["label"]}
            for result in results
        ]

        sub_intent = top_five[0]["intent"]
        output = {
            "intent": MAPPINGS[sub_intent],
            "subIntent": sub_intent,
            "additionalData": {"similarQueries": top_five},
            "llmPrediction": sub_intent,
        }

        # Avoid calling llm when function is called from compound query
        if not multi:
            sub_intent = await self.get_sub_intent_from_llm(query, top_five)
            # Split compound intents and get the primary one
            primary_intent = sub_intent.split(", ")[0].strip()
            output["intent"] = (
                MAPPINGS[primary_intent] if "unknown" not in sub_intent else "Unknown"
            )
            output["subIntent"] = sub_intent
            output["additionalData"]["llmPrediction"] = sub_intent

        return output

    async def search(self, query: str, limit: int = 10) -> Dict:
        """Search with compound query support."""
        # query_parts = self._split_compound_sentence(query)

        # if len(query_parts) <= 1:
        return await self._search_single(query, limit=limit)

        # return await self._search_compound(query, query_parts, limit=limit)

    def _split_compound_sentence(self, text: str) -> List[str]:
        """Split a compound sentence into individual parts.

        Uses common conjunctions and punctuation as splitting points.
        """
        # Split on common conjunctions and punctuation
        separators = r"(?<=[.!?])\s+|\s+(?:and|but|or|because|however|moreover)\s+"
        parts = re.split(separators, text, flags=re.IGNORECASE)
        # Clean and filter out empty strings
        return [part.strip() for part in parts if part.strip()]

    async def _search_compound(
        self, query: str, query_parts: list, limit: int = 3
    ) -> Dict:
        """Search for compound queries and combine results in an interleaved manner."""

        # Use gather to parallelize searches
        results = await asyncio.gather(
            *[
                self._search_single(part, limit=limit, multi=True)
                for part in query_parts
            ]
        )

        # First, collect all unique sub_intents and their corresponding queries
        intent_to_queries = {}
        for result in results:
            for similar_query in result["additionalData"]["similarQueries"]:
                intent = similar_query["intent"]
                if intent not in intent_to_queries:
                    intent_to_queries[intent] = similar_query

        # Convert to list while preserving order of discovery
        combined_top_five = list(intent_to_queries.values())

        output = {
            "additionalData": {
                "similarQueries": combined_top_five,
                "queryParts": query_parts,
            },
        }

        sub_intent = await self.get_sub_intent_from_llm(
            query, combined_top_five, multi_intent=True
        )
        # Split compound intents and get the primary one
        primary_intent = sub_intent.split(", ")[0].strip()
        output["intent"] = (
            MAPPINGS[primary_intent] if "unknown" not in sub_intent else "Unknown"
        )
        output["subIntent"] = sub_intent
        output["additionalData"]["llmPrediction"] = sub_intent

        return output


async def main(
    collection: str,
    dense_model: str,
    sparse_model: str,
    late_interaction_model: str,
    use_tf: bool = False,
    use_local: bool = False,
):

    search_service = await RAGSearchService.get_instance(
        collection=collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        late_interaction_model=late_interaction_model,
        use_tf=use_tf,
        use_local=use_local,
    )

    while True:
        query = input("Enter query: ")

        results = await search_service.search(query=query)
        signal(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="ivr-classifier")
    parser.add_argument(
        "--dense-model",
        type=str,
        default="s1lv3rj1nx/setfit-mxbai-embed-xsmall-v1",
    )
    parser.add_argument(
        "--sparse-model",
        type=str,
        default="Qdrant/bm25",
    )
    parser.add_argument(
        "--late-interaction-model",
        type=str,
        default="colbert-ir/colbertv2.0",
    )
    parser.add_argument(
        "--use-tf", action="store_true", help="Use TF URL instead of CVS URL"
    )
    parser.add_argument(
        "--use-local", action="store_true", help="Use local Qdrant instead of cluster"
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
