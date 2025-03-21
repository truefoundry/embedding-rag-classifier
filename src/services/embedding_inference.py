from typing import Dict, List
import argparse
import asyncio
import torch
from setfit import SetFitModel
from openai import AsyncOpenAI
from src.config import settings
from src.logger import logger


class EmbeddingSearchService:
    _instance = None  # Singleton instance

    @classmethod
    async def get_instance(cls, model_name: str):
        """Get singleton instance of EmbeddingSearch."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.model_name = model_name
            # Load the model during initialization
            token = settings.HF_TOKEN
            if not token:
                raise ValueError("HF_TOKEN is not set in the configuration")
            cls._instance.model = cls._instance.load_model(model_name, token)
        return cls._instance

    def load_model(self, model_name: str, token: str) -> SetFitModel:
        """Load the SetFit model from HuggingFace."""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model = SetFitModel.from_pretrained(model_name, token=token, device="cpu")
        model = model.to(device)
        logger.info(f"Model {model_name} loaded on {device}")
        return model

    async def predict(self, query: str) -> Dict:
        """Predict intent for a given query."""
        try:
            # Get prediction from the model
            intent = self.model.predict([query])[0]
            return intent
        except Exception as e:
            logger.error(f"Error predicting intent: {str(e)}")
            raise


async def main(model_name: str):
    embedding_search = await EmbeddingSearchService.get_instance(model_name=model_name)

    while True:
        query = input("Enter query: ")
        result = await embedding_search.predict(query=query)
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict intents using embedding model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="truefoundry/setfit-mxbai-embed-xsmall-v1-ivr-classifier",
        help="Name of the model to use for predictions",
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
