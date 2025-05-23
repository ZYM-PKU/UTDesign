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
    

if  __name__ == "__main__":

    datype = "train"
    resolution = 256
    data_roots = ["/blob/data/JPoster/", "/blob/data/CPoster/", "/blob/data/YPoster/", "/blob/data/TPoster/"]
    data_thresholds = [0.4, 0.4, 0.4, 0.0]
    dataset_names = ["CollectedGlyphDataset"] * len(data_roots)
    dataset_cfgs = [
        dict(
            data_root=data_root,
            asset_root="/blob/proj/poster_gen/assets",
            std_font_path="/blob/data/all_fonts/NotoSansSC-Regular.ttf",
            min_box_size=72,
            min_distance=threshold,
            min_alpha_ratio=0.2,
        ) for data_root, threshold in zip(data_roots, data_thresholds)
    ]

    dataset = MergedGlyphDataset(
        datype=datype,
        resolution=resolution,
        dataset_names=dataset_names, 
        dataset_cfgs=dataset_cfgs
    )
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        collate_fn=general_collate_fn,
    )

    for dataset in dataset.datasets:
        print(f"samples in {dataset.data_root}: {len(dataset.index_map)}")

    for batch in tqdm(dataloader):
        img = batch["img"][0]
        bg = batch["bg"][0]
        gt_pil_list = batch["gt_pil"][0]
        s_refs_pil_list = batch["s_refs_pil"][0]
        crop_img = batch["crop_img"][0]
        crop_bg = batch["crop_bg"][0]
        labels = batch["labels"][0]
        prompt = batch["prompt"][0]

        gt_pil_combined = Image.new('RGBA', (gt_pil_list[0].width * len(gt_pil_list), gt_pil_list[0].height))
        s_refs_pil_combined = Image.new('RGBA', (s_refs_pil_list[0].width * len(s_refs_pil_list), s_refs_pil_list[0].height))
        for i, (gt_pil, s_refs_pil) in enumerate(zip(gt_pil_list, s_refs_pil_list)):
            gt_pil_combined.paste(gt_pil, (i * gt_pil.width, 0))
            s_refs_pil_combined.paste(s_refs_pil, (i * s_refs_pil.width, 0))

        img.save("img.png")
        bg.save("bg.png")
        crop_img.save("crop_img.png")
        crop_bg.save("crop_bg.png")
        gt_pil_combined.save("gt_pil_combined.png")
        s_refs_pil_combined.save("s_refs_pil_combined.png")
        print(f"labels: {labels}")
        print(f"prompt: {prompt}")
        pass