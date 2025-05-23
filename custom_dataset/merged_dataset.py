import os
import sys
sys.path.append(os.getcwd())
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

from custom_dataset.raw_dataset import CollectedGlyphDataset, CollectedGlyphDPODataset
from custom_dataset.synth_dataset import (
    SynthGlyphDataset,
    SynthGlyphSingleDataset,
    SynthGlyphRegionDataset,
)
from custom_dataset.utils.helper import general_collate_fn


class MergedGlyphDataset(data.Dataset):

    def __init__(
        self,
        datype,
        resolution,
        dataset_names,
        dataset_cfgs,
    ):
        self.datasets = []
        self.annos = []
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            dataset = eval(dataset_name)(
                datype=datype,
                resolution=resolution,
                **dataset_cfg,
            )
            self.datasets.append(dataset)
            if hasattr(dataset, "annos"):
                self.annos += dataset.annos

    def __len__(self):

        return sum([len(dataset) for dataset in self.datasets])
    
    def get_curr_dataset(self, anno_index):

        for dataset in self.datasets:
            if anno_index < len(dataset.annos):
                return dataset
            anno_index -= len(dataset.annos)
            
        raise IndexError
    
    def __getitem__(self, index):

        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)

        raise IndexError