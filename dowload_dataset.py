from datasets import load_dataset

queries = load_dataset(path="milistu/amazon-esci-data", name="queries", split=["train", "test"])