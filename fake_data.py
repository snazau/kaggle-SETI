import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import sklearn
import torch

import config
import visualization


def dummy_signal_augmentation(img, target=0):
    """
    Original code - https://www.kaggle.com/yururoi/dummy-signal-data-augmentation
    """
    # signal shape
    x = int(np.random.uniform(50, 250))
    y = int(np.random.uniform(800, 1300))
    ax_x = int(np.random.uniform(15, 30))
    ax_y = int(np.random.uniform(500, 650))
    angle = np.random.uniform(-5, -1.5)
    start_angle = np.random.uniform(115, 130)
    end_angle = np.random.uniform(220, 235)
    color = np.random.uniform(0.8, 2.5)

    # signal_value
    signal_value = np.random.uniform(4, 4.5)

    signal = np.full((1638, 256), 0, dtype=np.uint8)
    signal = cv2.ellipse(signal, (x, y), (ax_x, ax_y), angle, start_angle, end_angle, color, thickness=1)
    signal = np.where(signal > 0, signal_value, signal)

    augmentation_img = np.copy(img)
    augmentation_img = np.vstack(augmentation_img.astype(np.float))
    augmentation_img = augmentation_img + signal

    if target == 0:
        return np.array(np.vsplit(augmentation_img, 6))

    split_img = np.vsplit(augmentation_img, 6)

    return np.array([split_img[0], img[1], split_img[2], img[3], split_img[4], img[4]])


def create_additional_data_csv():
    additional_data_dir = os.path.join(".", "temp", "some_data", "archive", "primary_small")
    set_names = ["test", "train", "valid"]
    anomaly_types = [
        "narrowband",
        "narrowbanddrd",
        "squarepulsednarrowband",
        "squiggle",
        "squigglesquarepulsednarrowband",
    ]

    anomaly_ids = []
    anomaly_paths = []
    for set_name in set_names:
        set_dir = os.path.join(additional_data_dir, set_name)
        for anomaly_type in anomaly_types:
            anomaly_dir = os.path.join(set_dir, anomaly_type)

            for filename in os.listdir(anomaly_dir):
                anomaly_name, anomaly_ext = os.path.splitext(filename)

                signal_path = os.path.join(anomaly_dir, filename)
                # print(signal_path)

                anomaly_ids.append(anomaly_name)
                anomaly_paths.append(signal_path)
                continue

    anomaly_data = {
        'id': anomaly_ids,
        'path': anomaly_paths
    }

    anomaly_df = pd.DataFrame(anomaly_data, columns=['id', 'path'])
    anomaly_csv_path = os.path.join(".", "temp", "additional_data.csv")
    anomaly_df.to_csv(anomaly_csv_path, index=False)


def mixup_with_additional_data(cadence):
    additional_csv_path = os.path.join(".", "temp", "additional_data.csv")
    df_additional = pd.read_csv(additional_csv_path)
    additional_paths = df_additional["path"].values

    additional_anomaly_path = random.choice(additional_paths)
    anomaly = cv2.imread(additional_anomaly_path, cv2.IMREAD_GRAYSCALE)
    print("anomaly", anomaly.shape, anomaly.min(), anomaly.mean(), anomaly.max())

    presented_on_n_A_observations = random.choice([2, 3])
    anomaly_width, anomaly_height = 256, 273 * presented_on_n_A_observations
    anomaly = cv2.resize(anomaly, (anomaly_width, anomaly_height))
    print("anomaly", anomaly.shape, anomaly.min(), anomaly.mean(), anomaly.max())

    anomaly_mean, anomaly_std = anomaly.mean(), anomaly.std()
    anomaly = (anomaly - anomaly_mean) / anomaly_std
    print("anomaly", anomaly.shape, anomaly.min(), anomaly.mean(), anomaly.max())

    cadence_mean, cadence_std = cadence.mean(), cadence.std()
    print("cadence", cadence.shape, cadence.min(), cadence.mean(), cadence.max(), cadence.std())
    cadence_copy = (cadence - cadence_mean) / cadence_std
    print("cadence_copy", cadence_copy.shape, cadence_copy.min(), cadence_copy.mean(), cadence_copy.max())

    lam = random.uniform(0.1, 0.5)
    random_start = int(random.uniform(0, 273 // 2))
    random_end = anomaly_height if presented_on_n_A_observations == 1 else int(random.uniform(anomaly_height - 273 // 2, anomaly_height))

    signal = np.vstack(cadence_copy[[0, 2, 4, 1, 3, 5]])  # vstack AAABCD
    mixed_signal = signal
    mixed_signal[random_start:random_end] = (1 - lam) * signal[random_start:random_end] + lam * anomaly[random_start:random_end]

    mixed_cadence = np.zeros_like(cadence)
    for i in range(6):
        tmp = i // 2 + (3 if i % 2 == 1 else 0)
        mixed_cadence[i] = mixed_signal[tmp * 273: (tmp + 1) * 273]

    sample = {
        "label": 1,
        "tensor": torch.from_numpy(cadence)
    }
    visualization.visualize_sample(sample, "original")

    sample["tensor"] = torch.from_numpy(mixed_cadence)
    visualization.visualize_sample(sample, "mixed")


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    # create_additional_data_csv()
    # exit(0)

    data_df = pd.read_csv(config.train_csv_path)
    data_df = sklearn.utils.shuffle(data_df)
    data_df["target"] = data_df["target"].apply(lambda x: int(x))
    labels = list(data_df["target"])
    npy_paths = list(data_df["path"])

    show_amount = 5
    showed = 0
    class_to_show = 0
    for index, row in data_df.iterrows():
        npy_path = row["path"]
        label = int(row["target"])
        npy_name = row["id"]

        if label != class_to_show:
            continue

        if showed == show_amount:
            break

        print(npy_name)
        title = npy_name + "_label=" + str(label) + "_#" + str(showed + 1)

        cadence = np.load(npy_path).astype(np.float32)
        mixup_with_additional_data(cadence)
        exit(0)

        # sample = {"label": label, "tensor": torch.from_numpy(signal)}
        # visualization.visualize_sample(sample, title=title)
        #
        # signal_fake = dummy_signal_augmentation(signal, label)
        # sample["tensor"] = torch.from_numpy(signal_fake)
        # visualization.visualize_sample(sample, title=title)
        showed += 1
