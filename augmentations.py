import albumentations as A
import copy
import pandas as pd
import random

import torch

import config
import dataset
import visualization


def freq_mask(spec, F=30, num_masks_per_channel=None, replace_with_zero=False):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    channels = spec.shape[0]

    if num_masks_per_channel is None:
        num_masks_per_channel = [1] * channels
    assert len(num_masks_per_channel) == channels

    for channel_num in range(channels):
        for i in range(num_masks_per_channel[channel_num]):
            f = random.randrange(0, F)
            f_zero = random.randrange(0, num_mel_channels - f)

            # avoids randrange error if values are equal and range is empty
            if f_zero == (f_zero + f):
                return cloned

            mask_end = random.randrange(f_zero, f_zero + f)

            if replace_with_zero is True:
                cloned[channel_num][f_zero:mask_end] = 0
            else:
                cloned[channel_num][f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=40, num_masks_per_channel=None, replace_with_zero=False):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]
    channels = spec.shape[0]

    if num_masks_per_channel is None:
        num_masks_per_channel = [1] * channels
    assert len(num_masks_per_channel) == channels

    for channel_num in range(channels):
        for i in range(num_masks_per_channel[channel_num]):
            t = random.randrange(0, T)
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == (t_zero + t):
                return cloned

            mask_end = random.randrange(t_zero, t_zero + t)

            if replace_with_zero is True:
                cloned[channel_num][:, t_zero:mask_end] = 0
            else:
                cloned[channel_num][:, t_zero:mask_end] = cloned.mean()
    return cloned


def coarse_dropout(spec, max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8, fill_value=0, p=0.5):
    cloned = copy.deepcopy(spec)
    transform = A.CoarseDropout(
        always_apply=False,
        max_holes=max_holes,
        max_height=max_height,
        max_width=max_width,
        min_holes=min_holes,
        min_height=min_height,
        min_width=min_width,
        fill_value=fill_value,
        p=p,
    )
    channels = spec.shape[0]
    for channel_num in range(channels):
        cloned[channel_num] = transform(image=cloned[channel_num])["image"]
    return cloned


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    labels_df = pd.read_csv(config.labels_csv_path)
    labels_df["target"] = labels_df["target"].apply(lambda x: int(x))
    labels = list(labels_df["target"])
    npy_paths = list(labels_df["path"])

    ds = dataset.SETIDataset(
        labels,
        npy_paths,
        in_channels=6,
        desired_image_size=273,
        augment=True,
        normalize=False,
    )

    sample = ds[0]
    visualization.visualize_sample(sample)

    signal = sample["tensor"]
    signal_time_masked = time_mask(signal, num_masks_per_channel=[2] * 6)
    signal_freq_masked = freq_mask(signal, num_masks_per_channel=[2] * 6)
    signal_timefreq_masked = time_mask(freq_mask(signal, num_masks_per_channel=[2] * 6), num_masks_per_channel=[2] * 6)
    print(signal.shape)

    # sample["tensor"] = signal_time_masked
    # visualization.visualize_sample(sample)
    # sample["tensor"] = signal_freq_masked
    # visualization.visualize_sample(sample)
    # sample["tensor"] = signal_timefreq_masked
    # visualization.visualize_sample(sample)

    signal_coarsed = coarse_dropout(signal.numpy(), max_height=32, max_width=32, min_height=32, min_width=32, fill_value=255)
    sample["tensor"] = torch.from_numpy(signal_coarsed)
    visualization.visualize_sample(sample)

    # showed_amount = 0
    # for sample in ds:
    #     rand_index = random.randint(0, len(ds) - 1)
    #     sample = ds[rand_index]
    #     if sample["label"] == 1:
    #         signal = sample["tensor"]
    #         signal_time_masked = time_mask(signal)
    #         signal_freq_masked = freq_mask(signal)
    #         signal_timefreq_masked = time_mask(freq_mask(signal))
    #         print(signal.shape)
    #
    #         sample["tensor"] = signal_time_masked
    #         visualization.visualize_sample(sample)
    #         sample["tensor"] = signal_freq_masked
    #         visualization.visualize_sample(sample)
    #         sample["tensor"] = signal_timefreq_masked
    #         visualization.visualize_sample(sample)
    #
    #         showed_amount += 1
    #
    #     if showed_amount == 3:
    #         break
