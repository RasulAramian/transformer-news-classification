from datasets import load_dataset


def load_ag_news():
    """Load the AG News dataset from Hugging Face."""
    dataset = load_dataset("wangrongsheng/ag_news")
    return dataset


if __name__ == "__main__":
    dataset = load_ag_news()

    print(dataset)
    print("\nTrain split:")
    print(dataset["train"])

    print("\nTest split:")
    print(dataset["test"])
