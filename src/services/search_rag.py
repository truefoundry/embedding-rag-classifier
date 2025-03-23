from typing import List, Dict
import argparse
import asyncio
from openai import AsyncOpenAI
from src.services.infinity import InfinityEmbeddings
from src.services.qdrant import QdrantVectorDB, EmbeddingCategory
from src.config import settings
from src.logger import logger

INTENT_DESCRIPTIONS = """
1. store_address: User is asking for the address or location of a store.
2. store_hours: User is asking for the operating hours of a store. Please include all store hours and timing frontstore hours, clinical hours, store hours, Retail hours, photo hours etc
3. store_timezone: User is asking for the operating hours across multiple time zones and locations of a store in US.
4. store_directions: User is asking for directions to reach the specific CVS or Target store.
5. store_faq: User is asking for questions about the store related to store policies, store services, store phone and fax number, payment methods, reward programs and accepted insurance types at the store.
6. drug_information: User is asking for information about a particular drug or medication or prescriptions.
7. drug_side_effects: User is asking for information about the side effects of a particular drug or medication or prescriptions.
8. drug_usage: User is asking for information about the usage of a particular drug or medication basically wanting to understand if a drug is appropriate to use given their specific health condition.
9. drug_dosage_instructions: User is asking for information about the dosage instructions for prescribed medication.
10. drug_interactions: User is asking for information about the interactions of a particular drug with other drugs or substances.
11. drug_insurance_coverage: User is asking for information about the insurance coverage of a particular drug, to check if it is covered under the customer's insurance policy.
12. drug_availability: User is asking for information about the availability of a particular drug at a CVS store, or check the inventory or drug availability.
13. rx_status:   User is asking about the status of their last prescription fill, estimated timelines for preparation and pick-up, managing their expectations for future pickups. User may also ask about picking up prescriptions for others. keywords like order status, order, prescription, prescriptions, another prescription, different prescriptions etc
14. rx_action_notes_status: User is asking queries where he seeks any actions are required by the customer or doctor, if there are any insurance issues, if the request has been denied.
15. rx_refill:  User is asking queries about refilling or renewing an existing prescriptions or orders, including requests for refills, status updates, and inquiries about refill procedures.
16. rx_cancel:  User is asking queries about canceling existing prescriptions or orders, number of prescriptions for cancel, including inquiries about the cancellation process
17. rx_expedite:  User is asking queries about expediting the processing of a prescription, number of prescriptions for expedite, including requests for faster processing, and inquiries about expedited procedures.
18. rx_drug_price:  User is asking queries about the price of a drug, including inquiries about the price of a specific drug, or the price of a prescription.
19. live_pharmacist:  User is asking for any assistance, requesting to talk to a pharmacist, customer service, agent or any one from pharmacy in any case is categorized under live_pharmacist intent. User asking to connect to pharmacy store also falls within live_pharmacist intent. Please don't include the user requests that could be flagged for containing racial, danger, or legal concerns in a live pharmacist chat service. Could be customer service, agent, talk for help, connect with pharmacist, connect to agent, leave a voice message, Leave a voicemail etc are different keyword and examples. Query which have other intent( such as status, refill, prescription) and talk to pharmacist should not be be categorized in here.
20. get_vaccine_appointment:  User is asking for information about 'existing' vaccine appointments, including appointment time, date, location, and vaccine type. Any appointment schedule is consider as vaccine appointment. User query that asks to check the schedule or check the calendar fall under get_vaccine_appointment category.
21. available_vaccine:  User is asking for vaccine or immunization available or performed at store.
22. create_vaccine_appointment:  User is asking to book a new vaccine appointment.
23. cancel_vaccine_appointment: User is asking to cancel an existing vaccine appointment.
24. internal_rx_transfer:  User is asking to transfer a prescription from one CVS store to another CVS/Target pharmacy only. Reasons for this transfer could include changing the pick-up location for convenience or due to a move, preferring a different in-network location, or trying to find a location where the prescription is in stock. Queries like "transfer prescriptions to another pharmacy", "transfer medications to another store" fall in this category.
25. external_rx_transfer:  User is asking to transferring prescriptions to or from another EXTERNAL pharmacy like Walgreens, rite aid etc. Reasons for this kind of transfer might be the non-availability of a drug in CVS or Target stores, or the customer has moved to a location where the nearest pharmacy is not a CVS or Target store. The query should clearly state the intent of transferring from CVS to external pharmacy
26. rx_delivery: User is asking about delivering prescriptions to the customer's location, including requests for prescription delivery, inquiries about prescription delivery options.
27. thank:  User is expressing gratitude or saying goodbye, or saying thank you, I'm grateful, thanks, thank you or no further questions indication query has been resolved or the user has no questions, "nothing" or "no, i am good" or "I have nothing else" or "none" or "no"  or "stop" or "halt" or "abort" or "not now" or "not a good" or "don not bother" or "nothing wrong" or "leave me alone" or "do not want to talk" or "do not engage with me" or "do not talk to me" or "let me be, don't talk to me" queries.
28. hold_on:  The user has requested a brief pause to gather additional information.
29. healthcare: The user has identified themselves as a healthcare professional, such as a doctor, nurse, or physician assistant, healthcare staff or mentioned contacting a healthcare office or staff. The user's intent is to be connected with a pharmacist for direct communication. The user can identify as healthcare and ask for refill prescription, check order status or talk to live pharmacist. Queries such as "I'm a healthcare provider", "healthcare provider and need to refill", "healthcare and I need to check order status", "healthcare provider and connect to live pharmacist", "healthcare provider...hmm....i also need to refill prescription", "healthcare and I also need to cancel a prescription". Keywords could be doctor, prescriber, doctor office, nurse, provider, healthcare provider, medical assistant.
30. rephrase:  The user requests to have the bot's response repeated or rephrased, possibly due to misunderstanding or simply wanting to hear it again. Keywords include repeat, rephrase, say it again.
31. store_connect: The user is requesting to connect only to to front store or retail department or photo services  for issues regarding an items purchased at the front or retail store, to check availability of a product at the store, speaking to the store manager any store employee, any item loss at the store or enquiring about store holiday hours. User query related to pharmacy store is categorzied under live_pharmacist intent.
"""


PROMPT_TEMPLATE = """
You are an intent classifier. Your task is to identify user intent based on intent descriptions given below.

## Intent names with description:
{intent_descriptions}


## Rules:
a. If there is an approximate query match found in the examples from VectorDB, use the intent from the examples to determine the correct intent of the user query.
b. If the query consists of a compound sentence, i.e, two sentences joined by a conjunction, separately identify intent for each of the individual sentence. For example, if the user query is: 'Can you book me a vaccine appointment and get me the store address as well?'. You should identify the intent for 'Can you book me a vaccine appointment' and 'get me a store address as well?'  which is 'get_vaccine_appointment' and 'store_address'.
c. Return ONLY the intent name in the response nothing else, in case of multiple intents (for compound sentences) return a COMMA separated list of intents."""

APPENDING_PROMPT = """
## Approximate examples from Vector DB:
{rag_faq}


Input :
User query: {query}"""


class RAGSearchService:
    _instance = None  # Class variable to hold the singleton instance

    @classmethod
    async def get_instance(
        cls,
        collection: str,
        dense_model: str,
        sparse_model: str = "Qdrant/bm25",
        late_interaction_model: str = "colbert-ir/colbertv2.0",
    ):
        """Get the singleton instance of SearchService."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.collection = collection
            cls._instance.dense_model = dense_model
            cls._instance.sparse_model = sparse_model
            cls._instance.late_interaction_model = late_interaction_model

            # Initialize clients immediately
            cls._instance.infinity_client = cls._instance._create_infinity_client()
            cls._instance.qdrant_vector_db = await cls._instance._create_qdrant_client()
            cls._instance.llm_client = await cls._instance._create_llm_client()
        else:
            # Update the instance properties when called with potentially different params
            cls._instance.collection = collection
            cls._instance.dense_model = dense_model
            cls._instance.sparse_model = sparse_model
            cls._instance.late_interaction_model = late_interaction_model

            # Always recreate clients to ensure fresh connections
            cls._instance.infinity_client = cls._instance._create_infinity_client()
            cls._instance.qdrant_vector_db = await cls._instance._create_qdrant_client()
            cls._instance.llm_client = await cls._instance._create_llm_client()

        return cls._instance

    def _create_infinity_client(self):
        """Create an InfinityEmbeddings client for generating embeddings."""
        return InfinityEmbeddings(
            url=settings.INFINITY_URL, api_key=settings.INFINITY_API_KEY
        )

    async def _create_qdrant_client(self):
        """Create a Qdrant client for indexing documents."""
        try:
            qdrant_vector_db = QdrantVectorDB.create(url=settings.QDRANT_URL)
            await qdrant_vector_db.connect()
            return qdrant_vector_db
        except Exception as e:
            logger.error(f"Error creating Qdrant client: {e}")
            # Try again with a new client
            qdrant_vector_db = QdrantVectorDB.create(url=settings.QDRANT_URL)
            await qdrant_vector_db.connect()
            return qdrant_vector_db

    async def _create_llm_client(self):
        """Create an OpenAI client for generating embeddings."""
        client = AsyncOpenAI(
            base_url=settings.LLM_GATEWAY_URL, api_key=settings.LLM_GATEWAY_API_KEY
        )
        return client

    async def _get_intent_from_llm(
        self, query: str, top_five: List[Dict[str, str]], intent_descriptions: str = ""
    ) -> str:
        """Get the sub-intent from the LLM."""
        # Convert top_five to a string
        top_five_str = "\n\n".join(
            [
                f"{i+1}. Query: {item['query']}\n   Intent: {item['intent']}"
                for i, item in enumerate(top_five)
            ]
        )

        if not intent_descriptions:
            intent_descriptions = INTENT_DESCRIPTIONS
        prompt = PROMPT_TEMPLATE.format(
            intent_descriptions=intent_descriptions
        ) + APPENDING_PROMPT.format(rag_faq=top_five_str, query=query)

        response = await self.llm_client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Add temperature for some variability
        )
        intent_message: str = response.choices[0].message.content.lower()
        intent_message = intent_message.strip()

        return intent_message

    async def search(
        self, query: str, limit: int = 5, intent_descriptions: str = ""
    ) -> Dict:
        """Search for the given intent."""
        try:
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

            try:
                results = await self.qdrant_vector_db.search(
                    collection_name=self.collection,
                    embeddings_dict=embeddings_dict,
                    limit=limit,
                )
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Reconnect and try again
                    logger.warning("Event loop closed, reconnecting to Qdrant")
                    self.qdrant_vector_db = await self._create_qdrant_client()
                    results = await self.qdrant_vector_db.search(
                        collection_name=self.collection,
                        embeddings_dict=embeddings_dict,
                        limit=limit,
                    )
                else:
                    raise

            top_five = [
                {"query": result.payload["text"], "intent": result.payload["label"]}
                for result in results
            ]

            llm_intent = await self._get_intent_from_llm(
                query, top_five, intent_descriptions
            )
            output = {
                "intent": llm_intent,
                "additionalData": {"similarQueries": top_five},
            }

            return output
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise


async def main(
    collection: str,
    dense_model: str,
    sparse_model: str = "Qdrant/bm25",
    late_interaction_model: str = "colbert-ir/colbertv2.0",
):

    search_service = await RAGSearchService.get_instance(
        collection=collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        late_interaction_model=late_interaction_model,
    )

    while True:
        query = input("Enter query: ")

        results = await search_service.search(query=query)
        print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="ivr-classifier")
    parser.add_argument(
        "--dense-model",
        type=str,
        default="truefoundry/setfit-mxbai-embed-xsmall-v1-ivr-classifier",
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
