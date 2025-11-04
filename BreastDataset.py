
import torch
from torch.utils.data import Dataset
from data_utils import load_dicom_as_image

class BreastDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.items = []
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path = self.dataframe.iloc[idx]['new_path']
        y_text = str(self.dataframe.iloc[idx]['label']).strip().lower()
        label_map = {'negative': 0, 'suspicious': 1}
        if y_text not in label_map:
            raise ValueError(f"Unknown label: {y_text}")
        y = label_map[y_text]
        dicom = load_dicom_as_image(path)
        # im=(im-IM_MEAN)/IM_STD
        dicom = torch.from_numpy(dicom).repeat(3,1,1)
        if self.transform:
            dicom = self.transform(dicom)
        return dicom, torch.tensor(y,dtype=torch.long)