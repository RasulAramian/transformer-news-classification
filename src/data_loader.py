from datasets import load_dataset


DATASET_NAME = "fancyzhx/ag_news"


def load_ag_news():
    """Load the AG News dataset from Hugging Face."""

    dataset = load_dataset(DATASET_NAME)

    return dataset
