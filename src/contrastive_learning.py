import os
import shutil
import argparse
import datasets
from dotenv import load_dotenv
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from huggingface_hub import login
import torch
from datetime import datetime
from typing import Tuple, Dict
from transformers import TrainerCallback

from src.config import settings
from src.logger import logger


def load_and_split_dataset(input_csv: str) -> Tuple[datasets.Dataset, datasets.Dataset]:
    dataset = datasets.Dataset.from_csv(input_csv)
    logger.info(f"Total dataset: {len(dataset)}")

    train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()
    logger.info(f"Train dataset: {len(train_dataset)}")
    logger.info(f"Test dataset: {len(test_dataset)}")

    return train_dataset, test_dataset


def sample_datasets(
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    train_samples: int,
    test_samples: int,
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    sampled_train = sample_dataset(
        train_dataset, label_column="label", num_samples=train_samples
    )
    logger.info(f"Sampled train dataset: {len(sampled_train)}")

    sampled_test = sample_dataset(
        test_dataset, label_column="label", num_samples=test_samples
    )
    logger.info(f"Sampled test dataset: {len(sampled_test)}")

    return sampled_train, sampled_test


def initialize_model(model_name: str, token: str) -> SetFitModel:
    model = SetFitModel.from_pretrained(
        model_name, token=token, trust_remote_code=True, device="cpu"
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return model.to(device)


def get_training_args(batch_size: int, epochs: int) -> TrainingArguments:
    return TrainingArguments(
        batch_size=batch_size,
        num_epochs=epochs,
        loss=CosineSimilarityLoss,
        eval_strategy="epoch",
        save_strategy="epoch",
        output_dir="./results",
        load_best_model_at_end=True,
    )


class MetricsCallback(TrainerCallback):
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.progress_callback:
            self.progress_callback(logs)


def setup_trainer(
    model: SetFitModel,
    args: TrainingArguments,
    train_dataset: datasets.Dataset,
    test_dataset: datasets.Dataset,
    callback=None,
) -> Trainer:
    callbacks = []
    if callback:
        callbacks.append(MetricsCallback(callback))

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        metric="accuracy",
        callbacks=callbacks,
    )


def push_model_to_hub(
    trainer: Trainer, model_name: str, metrics: Dict, model_suffix: str = ""
):
    repo_name = f"truefoundry/setfit-{model_name.split('/')[-1]}-{model_suffix}"
    trainer.push_to_hub(
        repo_name,
        private=False,
        commit_message=f"Trained on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with accuracy {metrics.get('accuracy','')}",
    )
    return repo_name


def test_inference(model: SetFitModel, test_dataset: datasets.Dataset):
    # Choose 5 random samples from the test dataset
    test_texts = test_dataset["text"].shuffle(seed=42).select(range(5))
    preds = model.predict(test_texts)
    logger.info(f"Test predictions: {preds}")


def main(
    input_csv: str,
    model_name: str,
    train_samples: int,
    test_samples: int,
    batch_size: int,
    epochs: int,
    model_suffix: str = "",
    progress_callback=None,
):
    try:
        token = settings.HF_TOKEN
        login(token=token)

        # Load and prepare datasets
        train_dataset, test_dataset = load_and_split_dataset(input_csv)
        sampled_train, sampled_test = sample_datasets(
            train_dataset, test_dataset, train_samples, test_samples
        )

        # Initialize model and trainer
        model = initialize_model(model_name, token)
        training_args = get_training_args(batch_size, epochs)
        trainer = setup_trainer(
            model,
            training_args,
            sampled_train,
            sampled_test,
            callback=progress_callback,
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate(test_dataset)
        logger.info(f"Metrics: {metrics}")

        # Push to hub and test
        logger.info(f"Pushing model to hub: {model_name} with suffix: {model_suffix}")
        repo_name = push_model_to_hub(trainer, model_name, metrics, model_suffix)
        logger.info(f"Model pushed to hub: {repo_name}")

        # Delete the results directory
        try:
            shutil.rmtree("./results")
        except Exception as e:
            pass

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        default="./data/dataset.csv",
        help="Path to input CSV dataset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixedbread-ai/mxbai-embed-xsmall-v1",
        help="Name of the pretrained model to use",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4,
        help="Number of samples per class for training",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2,
        help="Number of samples per class for testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="",
        help="Suffix to add to the model name",
    )
    args = parser.parse_args()

    main(
        input_csv=args.input_csv,
        model_name=args.model_name,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
