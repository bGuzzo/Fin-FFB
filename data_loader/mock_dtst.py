import yaml
from torch.utils.data import Dataset


class MockDataset(Dataset):
    """
    A mock dataset that loads sentences from a YAML configuration file.
    Useful for testing the training pipeline without large external datasets.
    """

    def __init__(self, config_path: str = "config/mock_data.yaml"):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        self.sentences = data.get("sentences", [])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {"text": self.sentences[idx]}