"""Dataset for California Housing Prices."""
from torch.utils.data import Dataset


class HousingDataset(Dataset):
    """Custom dataset class for housing data.

    This class provides an interface to access the housing dataset for training
    or evaluation purposes.
    """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)
