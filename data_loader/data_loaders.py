# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from torchvision import transforms

from base import BaseDataLoader, BaseInferenceDataset, BaseTrainDataset

random.seed(123)


class Landsat8TrainDataLoader(BaseDataLoader):
    """
    Dataloader to train, validate, and test on Landsat8 generated pickle data
    """

    def __init__(self, data_dir, data_split_lists_path, batch_size, model_input_size, bands, num_classes, one_hot,
                 train_split=0.8, mode='train', num_workers=0,
                 split_manifest=None, split_manifest_sha256=None,
                 normalization_path=None, normalization_sha256=None, input_clip=10.0):

        assert mode in (
            'train', 'val', 'test'), "Invalid value for train/val/test mode"

        # generated pickle data path
        self.data_dir = data_dir

        if split_manifest is not None or split_manifest_sha256 is not None:
            from utils.training_protocol import load_manifest, load_normalization, source_path
            manifest = load_manifest(split_manifest, split_manifest_sha256, data_dir, model_input_size)
            split = manifest['splits'][mode]
            paths = [str(source_path(data_dir, file_id)) for file_id in split['file_ids']]
            samples = [(paths[index], row, col) for index, row, col in split['samples']]
            normalization = load_normalization(normalization_path, normalization_sha256,
                                               split_manifest_sha256, bands)
            self.dataset = BaseTrainDataset(
                paths, None, 8 if mode == 'train' else model_input_size,
                model_input_size, bands, num_classes, one_hot,
                mode='train' if mode == 'train' else 'test',
                transforms=transforms.ToTensor(), frozen_samples=samples,
                normalization=normalization, input_clip=input_clip)
            super().__init__(self.dataset, batch_size, mode == 'train', num_workers)
            return

        # Read-only compatibility for historical evaluation. Training entry
        # points require v3 above; missing maps must never generate new splits.
        if normalization_path is not None or normalization_sha256 is not None:
            raise ValueError('Normalization requires a pinned split manifest')
        data_map_path = os.path.join(data_split_lists_path, f'{mode}_datamap.pkl')
        if not os.path.isfile(data_map_path):
            raise FileNotFoundError('No frozen split manifest or historical data map; refusing to create a random split')

        if not os.path.exists(data_split_lists_path):
            print('LOG: No saved data found. Making new data directory {}'.format(
                data_split_lists_path))
            os.mkdir(data_split_lists_path)

        full_examples_list = [os.path.join(
            self.data_dir, x) for x in os.listdir(self.data_dir)]
        random.shuffle(full_examples_list)
        train_split = int(train_split*len(full_examples_list))

        if mode == 'train':
            data_list = full_examples_list[:train_split]
        else:
            temp_list = full_examples_list[train_split:]
            if mode == 'val':
                data_list = temp_list[0:len(temp_list)//2]
            else:
                data_list = temp_list[len(temp_list)//2:]

        data_map_path = os.path.join(
            data_split_lists_path, f'{mode}_datamap.pkl')


        trsfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        if mode == 'train':
            self.dataset = BaseTrainDataset(data_list, data_map_path, 8, model_input_size,
                                            bands, num_classes, one_hot,
                                            transforms=trsfm)
            super().__init__(self.dataset, batch_size, True, num_workers)
        else:
            self.dataset = BaseTrainDataset(data_list, data_map_path, model_input_size, model_input_size,
                                            bands, num_classes, one_hot,
                                            mode='test', transforms=trsfm)
            super().__init__(self.dataset, batch_size, False, num_workers)


class Landsat8InferenceDataLoader(BaseDataLoader):
    """
    Dataloader to infer Landsat8 generated pickle data
    """

    def __init__(self, image_path, district,
                 rasterized_shapefiles_path, bands, model_input_size, num_classes, batch_size,
                 num_workers=0, transforms=None):
        # create dataset class instances
        self.dataset = BaseInferenceDataset(rasterized_shapefiles_path=rasterized_shapefiles_path,
                                            image_path=image_path,
                                            bands=bands,
                                            model_input_size=model_input_size,
                                            district=district,
                                            num_classes=num_classes,
                                            transformation=transforms)
        super().__init__(self.dataset, batch_size, False, num_workers)
