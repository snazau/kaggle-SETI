import albumentations as A
import copy
import FMix.fmix as FMix
import numpy as np
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


def random_cadence_permutation(spec):
    cloned = copy.deepcopy(spec)
    # 6 channels are - ABACAD,
    # where A - location that we are interested in
    # where B, C, D - other location
    permutation = 2 * np.random.permutation([1, 2, 3]) - 1
    permutation = [0, permutation[0], 2, permutation[1], 4, permutation[2]]
    cloned = cloned[permutation]
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


def motion_blur(spec, p=0.5):
    cloned = copy.deepcopy(spec)
    transform = A.MotionBlur(p=p)
    channels = spec.shape[0]
    for channel_num in range(channels):
        cloned[channel_num] = transform(image=cloned[channel_num])["image"]
    return cloned


def shift_scale_rotate(spec, p=0.5):
    cloned = copy.deepcopy(spec)
    transform = A.ShiftScaleRotate(
        p=p,
        shift_limit_x=(-0.1, 0.1),
        shift_limit_y=(-0.1, 0.1),
        scale_limit=(-0.1, 0.1),
        rotate_limit=(-10, 10),
        interpolation=1,
        border_mode=0,
        value=0,
    )
    cloned[0] = transform(image=cloned[0])["image"]
    return cloned


def flip(spec, p=0.5):
    cloned = copy.deepcopy(spec)
    transform = A.Flip(p=p)
    cloned[0] = transform(image=cloned[0])["image"]
    return cloned


def fmix(data, targets, alpha, decay_power, shape, device, max_soft=0.0):
    lam, mask = FMix.sample_mask(alpha, decay_power, shape, max_soft)

    indices = torch.randperm(data.size(0)).to(device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    x1 = torch.from_numpy(mask) * data
    x2 = torch.from_numpy(1 - mask) * shuffled_data
    mixed_data = x1 + x2

    return mixed_data, targets, shuffled_targets, lam


def mixup(data, targets, alpha=1.0):
    # when alpha = 1 then Beta(1, 1) = U(0, 1)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]
    if data.is_cuda is True:
        indices = torch.randperm(batch_size).cuda()
    else:
        indices = torch.randperm(batch_size)

    mixed_data = lam * data + (1 - lam) * data[indices, :]
    shuffled_targets = targets[indices]
    return mixed_data, targets, shuffled_targets, lam


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    labels_df = pd.read_csv(config.train_csv_path)
    labels_df["target"] = labels_df["target"].apply(lambda x: int(x))
    labels = list(labels_df["target"])
    npy_paths = list(labels_df["path"])

    ds = dataset.SETIDataset(
        labels,
        npy_paths,
        in_channels=1,
        desired_image_size=256,
        augment=False,
        normalize=False,
    )

    sample = ds[0]
    visualization.visualize_sample(sample)

    signal = sample["tensor"]

    # SpecAugment check
    # signal_time_masked = time_mask(signal, num_masks_per_channel=[2] * 6)
    # signal_freq_masked = freq_mask(signal, num_masks_per_channel=[2] * 6)
    # signal_timefreq_masked = time_mask(freq_mask(signal, num_masks_per_channel=[2] * 6), num_masks_per_channel=[2] * 6)
    # print(signal.shape)

    # sample["tensor"] = signal_time_masked
    # visualization.visualize_sample(sample)
    # sample["tensor"] = signal_freq_masked
    # visualization.visualize_sample(sample)
    # sample["tensor"] = signal_timefreq_masked
    # visualization.visualize_sample(sample)

    # CoarseDropout check
    # signal_coarsed = coarse_dropout(signal.numpy(), max_height=32, max_width=32, min_height=32, min_width=32, fill_value=255)
    # sample["tensor"] = torch.from_numpy(signal_coarsed)
    # visualization.visualize_sample(sample)

    # MotionBlur check
    # signal_blurred = motion_blur(signal.numpy(), p=1)
    # sample["tensor"] = torch.from_numpy(signal_blurred)
    # visualization.visualize_sample(sample)

    # ShiftScaleRotate check
    signal_transformed = shift_scale_rotate(signal.numpy(), p=1)
    sample["tensor"] = torch.from_numpy(signal_transformed)
    visualization.visualize_sample(sample)

    # Flip check
    signal_transformed = flip(signal.numpy(), p=1)
    sample["tensor"] = torch.from_numpy(signal_transformed)
    visualization.visualize_sample(sample)

    # # FMix check - prbbly not good for that task
    # # mixup check
    # device = torch.device("cpu")
    # bs = 8
    # dataloader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    # for index, sample in enumerate(dataloader):
    #     inputs = sample["tensor"].to(device, config.dtype)
    #     labels = sample["label"].to(device, config.dtype)
    #     npy_paths = sample["npy_path"]
    #
    #     # mixed_inputs, labels, labels_shuffled, lam = fmix(inputs, labels, alpha=1.0, decay_power=5.0, shape=(256, 256), device=device)
    #     mixed_inputs, labels, labels_shuffled, lam = mixup(inputs, labels, alpha=1.0)
    #     print("mixed_inputs", mixed_inputs.shape)
    #     print("labels", labels.shape)
    #     print("labels_shuffled", labels_shuffled.shape)
    #     print("lam", lam)
    #
    #     for i in range(bs):
    #         kek = {
    #             "tensor": inputs[i],
    #             "label": labels[i].item()
    #         }
    #         visualization.visualize_sample(kek)
    #
    #         kek = {
    #             "tensor": mixed_inputs[i],
    #             "label": "orig=" + str(labels[i].item()) + "_shuffled=" + str(labels_shuffled[i].item()) + "_lam=" + str(lam)
    #         }
    #         visualization.visualize_sample(kek)
    #     exit()

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
