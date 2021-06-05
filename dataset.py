import cv2
import numpy as np
import os
import pandas as pd
import random
import torch

import config
import augmentations
import visualization


class SETIDataset(torch.utils.data.Dataset):
    def __init__(self, labels, npy_paths, in_channels, desired_image_size, augment, normalize, transforms=None):
        self.labels = labels
        self.npy_paths = npy_paths
        self.in_channels = in_channels
        self.desired_image_size = desired_image_size
        self.augment = augment
        self.transforms = transforms
        self.normalize = normalize

        self.signal_mean = np.load(os.path.join(config.data_dir, "train_mean.npy")).astype(np.float32)
        self.signal_std = np.load(os.path.join(config.data_dir, "train_std.npy")).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal = np.load(self.npy_paths[index]).astype(np.float32)  # [6 x 273 x 256]
        channel_amount, height, width = signal.shape

        if self.augment is True:
            # print("prbbly augmenting")

            # SpecAugment
            signal = torch.from_numpy(signal)
            if random.uniform(0, 1) < 0.5:
                # print("augmenting")
                num_masks_per_channel_time = [random.choice([0, 1, 2, 3]) for _ in range(channel_amount)]
                # print(num_masks_per_channel_time)
                signal = augmentations.time_mask(signal, num_masks_per_channel=num_masks_per_channel_time)

                num_masks_per_channel_freq = [random.choice([0, 1, 2, 3]) for _ in range(channel_amount)]
                # print(num_masks_per_channel_freq)
                signal = augmentations.freq_mask(signal, num_masks_per_channel=num_masks_per_channel_freq)
            signal = signal.numpy()

            # CoarseDropout for spectrogram
            if random.uniform(0, 1) < 0.5:
                signal = augmentations.coarse_dropout(
                    signal,
                    max_holes=10,
                    max_height=40,
                    max_width=40,
                    min_holes=5,
                    min_height=20,
                    min_width=20,
                    fill_value=0,
                    p=1  # prob to keep unchanged
                )

            # Random vertical flip
            if random.uniform(0, 1) < 0.5:
                # print("Flipping time")
                signal = np.flip(signal, axis=1)

            # Random horizontal flip
            if random.uniform(0, 1) < 0.5:
                # print("Flipping freq")
                signal = np.flip(signal, axis=2)

            # Random permutation of not important channels i.e, with indices [1, 3, 5]
            if random.uniform(0, 1) < 1:
                # 6 channels are - ABACAD,
                # where A - location that we are interested in
                # where B, C, D - other location
                permutation = 2 * np.random.permutation([1, 2, 3]) - 1
                permutation = [0, permutation[0], 2, permutation[1], 4, permutation[2]]
                signal = signal[permutation]
                # print("permutation", permutation)
            # exit()

        if self.normalize is True:
            # print("Normalizing")

            # wtf norm
            # signal = ((signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-7)).T
            # signal = ((signal - np.mean(signal, axis=0)) / (np.std(signal, axis=0) + 1e-7)).T
            # # signal = ((np.clip(signal, -1, 3) + 1) / 4 * 255).astype(np.uint8)
            # signal = (signal - signal.min()) / (signal.max() - signal.min())

            # meanstd norm - seems like better than wo normalization
            signal = (signal - self.signal_mean) / self.signal_std

            # log scale - worse than wo normalization
            # signal_mins_channelwise = signal.min(axis=(1, 2), keepdims=True)
            # signal_maxs_channelwise = signal.max(axis=(1, 2), keepdims=True)
            # signal = np.log(signal - signal_mins_channelwise + 1e-7)
            # signal = (signal - signal_mins_channelwise) / (signal_maxs_channelwise - signal_mins_channelwise)

        if self.in_channels == 6:
            # print("signal", signal.shape, signal.min(), signal.mean(), signal.max())
            # print()
            signal = np.pad(signal, ((0, 0), (0, 0), (0, self.desired_image_size - width)), 'constant')  # [6 x 273 x 273]
            # print("signal", signal.shape, signal.min(), signal.mean(), signal.max())
        elif self.in_channels == 1:
            # signal = np.vstack(signal).transpose((1, 0))
            signal = np.vstack(signal)
            # print(signal.shape)
            resized_shape = (self.desired_image_size, self.desired_image_size)
            signal = cv2.resize(signal, resized_shape, interpolation=cv2.INTER_AREA)
            # print("signal", signal.shape, signal.min(), signal.mean(), signal.max())

            signal = signal[np.newaxis, :, :]
            # print("signal", signal.shape, signal.min(), signal.mean(), signal.max())
            # exit()
        else:
            raise NotImplementedError

        sample = {
            "tensor": torch.from_numpy(signal),
            "label": self.labels[index],
            "npy_path": self.npy_paths[index],
        }
        return sample


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    labels_df = pd.read_csv(config.labels_csv_path)
    labels_df["target"] = labels_df["target"].apply(lambda x: int(x))
    labels = list(labels_df["target"])
    npy_paths = list(labels_df["path"])

    dataset = SETIDataset(labels, npy_paths, in_channels=1, desired_image_size=512, augment=True, normalize=False)
    showed_amount = 0
    for sample in dataset:
        # rand_index = random.randint(0, len(dataset) - 1)
        # sample = dataset[rand_index]
        if sample["label"] == 1:
            # print(sample["npy_path"])
            visualization.visualize_sample(sample)
            showed_amount += 1

        if showed_amount == 5:
            break