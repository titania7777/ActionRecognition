import os
import csv
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image # Try using the pillow-simd !!
from glob import glob

import math
import random
from torchvision.transforms import functional as F
# VideoDataset
class VideoDataset(Dataset):
    def __init__(self, frames_path:str, annotation_path:str, sampled_split:bool, train:bool, frame_size:int=112,
    # for self._index_sampler
    sequence_length:int=16, max_interval:int=-1, random_start_position:bool=False, uniform_frame_sample:bool=True,
    # for self._add_pads
    random_pad_sample:bool=False, channel_first:bool=True):

        self.train = train
        self.channel_first = channel_first
        self.frames_path = frames_path
        self.sampled_split = sampled_split
        self.sequence_length = sequence_length

        # arguments for self._index_sampler
        self.max_interval = max_interval
        self.random_start_position = random_start_position
        self.uniform_frame_sample = uniform_frame_sample

        # arguments for self._add_pads
        self.random_pad_sample = random_pad_sample

        if self.sampled_split:
            # =======================================
            # For sampled split annotations
            # =======================================
            # read a json file
            self.labels, self.categories = self._read_json(annotation_path)
            # =======================================
        else:
            # =======================================
            # For custom split annotations
            # =======================================
            # read a csv file
            self.labels, self.categories = self._read_csv(annotation_path)
            # =======================================
        
        self.num_classes = len(self.categories)

        # transformer
        if self.train:
            self.transform = transforms.Compose([
                RandomResizedCrop(size=frame_size, scale=(0.25, 1.0), ratio=(0.75, 1.0 / 0.75)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.4345, 0.4051, 0.3775],
                    std = [0.2768, 0.2713, 0.2737]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=frame_size),
                transforms.CenterCrop(size=frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.4345, 0.4051, 0.3775],
                    std = [0.2768, 0.2713, 0.2737]
                ),
            ])
    
    def __len__(self) -> int:
        return len(self.labels)

    def get_few_shot_sampler(self, iter_size:int, way:int, shot:int, query:int):
        return FewShotSampler(labels=[labels[1]for labels in self.labels], iter_size=iter_size, way=way, shot=shot, query=query)

    def _read_csv(self, csv_path: str) -> (list, dict):
        labels = [] # [[sub directory file path, label]]
        categories = {} # {label: category(str)}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for rows in reader:
                sub_file_path = rows[0]; label = int(rows[1]); category = rows[2]
                labels.append([sub_file_path, label])
                if label not in categories:
                    categories[label] = category

        return labels, categories
    
    def _read_json(self, json_path: str) -> (list, dict):
        labels = [] # [[sub directory file path, label]]]
        categories = {} # {label: category(str)}
        with open(json_path, "r") as f:
            self.jsons = json.load(f)

        for sub_file_path in self.jsons:
            label = int(self.jsons[sub_file_path]["label"])
            category = self.jsons[sub_file_path]["category"]
            labels.append([sub_file_path, label])
            if label not in categories:
                categories[label] = category

        return labels, categories

    """
    args: length_of_frames(int): length of current frames
        : sequence_length(int): number of require frames 
        : random_pad_sample(bool): select of sampling way
    
    returns: np.ndarray: resampled frames
    """
    def _add_pads(self, length_of_frames:int, sequence_length:int, random_pad_sample:bool) -> np.ndarray:
        # length -> array
        sequence = np.arange(length_of_frames)
        require_frames = sequence_length - length_of_frames

        if random_pad_sample:
            # random samples of frames
            add_sequence = np.random.choice(sequence, require_frames)
        else:
            # repeated a first frame
            add_sequence = np.repeat(sequence[0], require_frames)

        # sorting of the array
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        return sequence
    
    """
    args: length_of_frames(int): length of current frames
        : sequence_length(int): number of require frames
        : max_interval(int): set of maximum limit interval (-1 is no limitation)
        : random_start_position(bool): select of start point with randomly or 0
        : uniform_frame_sample(bool): select of sampling way

    returns: numpy.ndarray: sampled frames

    ex)
    input frames(index): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    args: length_of_frames: 20
        : sequence_length: 7
        : max_interval: 7
        : random_start_position: True
        : uniform_frame_sample: True

    interval = (length_of_frames // sequence_length) - 1 = (20 // 7) - 1 = 1
    require_frames = (interval + 1) x sequence_length - interval = (1 + 1) x 7 - 1 = 13
    range_of_start_position = length_of_frames - require_frames = 20 - 13 = 7
    start_position = 0 ~ 7

    start_position = 0
    result = 0, 2, 4, 6, 8, 10, 12

    start_position = 1
    result = 1, 3, 5, 7, 9, 11, 13

    .
    .
    .

    start_position = 7
    result = 7, 9, 11, 13, 15, 17, 19
    """
    def _index_sampler(self, length_of_frames:int, sequence_length:int, max_interval:int, random_start_position:bool, uniform_frame_sample:bool) -> np.ndarray:
        # sampling strategy(uniformly / randomly)
        if uniform_frame_sample:
            # set a default interval
            interval = (length_of_frames // sequence_length) - 1
            if max_interval != -1 and interval > max_interval:
                interval = max_interval
            
            # "require_frames" is number of requires frames to sampling with specified interval
            require_frames = ((interval + 1) * sequence_length - interval)
            
            # "range of start position" is range of start point of possible an sampling
            range_of_start_position = length_of_frames - require_frames

            # "random_start_position" is select a start position with randomly
            if random_start_position:
                start_position = np.random.randint(0, range_of_start_position + 1)
            else:
                start_position = 0
                
            sampled_index = list(range(start_position, require_frames + start_position, interval + 1))
        else:
            sampled_index = sorted(np.random.permutation(np.arange(length_of_frames))[:sequence_length])

        return sampled_index

    def __getitem__(self, index):
        sub_file_path, label = self.labels[index]

        # hmdb51 has some weird filenames that can't catch when using glob
        replaced_sub_file_path = sub_file_path.replace("]", "?")

        # get frames path
        images_path = np.array(sorted(glob(os.path.join(self.frames_path, replaced_sub_file_path, "*")), key=lambda file: int(file.split("/")[-1].split(".")[0])))

        # get index of samples
        length_of_frames = len(images_path)
        assert length_of_frames != 0, f"'{sub_file_path}' is not exists or empty."

        if length_of_frames >= self.sequence_length:
            if self.sampled_split:
                indices = sorted(self.jsons[sub_file_path]["index"][:self.sequence_length])
            else:
                indices = self._index_sampler(length_of_frames, self.sequence_length, self.max_interval, self.random_start_position, self.uniform_frame_sample)
                    
        else:
            indices = self._add_pads(length_of_frames, self.sequence_length, self.random_pad_sample)

        images_path = images_path[indices]
        
        # load frames
        if self.channel_first:
            # for 3D Conv
            data = torch.stack([self.transform(Image.open(image_path)) for image_path in images_path], dim=1)
        else:
            # for 2D Conv
            data = torch.stack([self.transform(Image.open(image_path)) for image_path in images_path], dim=0)

        return data, label

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.initialize = True

    def __call__(self, img):
        if self.initialize:
            self.params = self.get_params(img, self.scale, self.ratio)
            self.initialize = False

        i, j, h, w = self.params # Fixed Parameters
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

class FewShotSampler():
    def __init__(self, labels:list, iter_size:int, way:int, shot:int, query:int):
        self.iter_size = iter_size
        self.way = way
        self.shots = shot + query

        labels = np.array(labels)
        self.indices = []
        for i in np.unique(labels):
            index = np.argwhere(labels == i).reshape(-1)
            self.indices.append(index)

    def __len__(self):
        return self.iter_size
    
    def __iter__(self):
        for i in range(self.iter_size):
            batchs = []
            classes = np.random.permutation(len(self.indices))[:self.way]
            for c in classes:
                l = self.indices[c]
                pos = np.random.permutation(len(l))[:self.shots]
                batchs.append(l[pos])

            batchs = np.stack(batchs).T.reshape(-1)
            yield batchs