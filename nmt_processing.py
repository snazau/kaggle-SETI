import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.utils
import seaborn as sns
import time
import torch

import config
import dataset
import visualization

# sns.set(style="whitegrid")


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocess_cadence(cadence, model_nmf):
    cadence_processed = np.zeros_like(cadence)
    for i in range(0, 6, 2):
        temp_on = cadence[i] + 50
        temp_off = cadence[i + 1] + 50

        W_on = model_nmf.fit_transform(temp_on)
        H_on = model_nmf.components_

        W_off = model_nmf.fit_transform(temp_off)
        H_off = model_nmf.components_

        # on_clean = sklearn.preprocessing.normalize(temp_on - np.matmul(W_on, H_on))
        on_clean = sklearn.preprocessing.normalize(temp_on - np.matmul(W_on, H_off))
        off_clean = sklearn.preprocessing.normalize(temp_off - np.matmul(W_on, H_off))
        # off_clean = sklearn.preprocessing.normalize(temp_off - np.matmul(W_off, H_off))

        cadence_processed[i] = on_clean
        cadence_processed[i + 1] = off_clean

    return cadence_processed


def show_cleaned_image_individually(image, title=None):
    """
    Original code - https://www.kaggle.com/manabendrarout/rfi-reduction-ideas-examples-seti
    """

    image_on = None
    image_off = None
    clean_image = None
    wtf_image = None
    wtf2_image = None
    wtf3_image = None

    for i in range(0, 6, 2):
        temp_on = image[i]
        temp_off = image[i + 1]

        print("temp_on", temp_on.shape, temp_on.min(), temp_on.mean(), temp_on.max())
        print("temp_off", temp_off.shape, temp_off.min(), temp_off.mean(), temp_off.max())

        temp_on = temp_on + 50
        temp_off = temp_off + 50

        print("temp_on", temp_on.shape, temp_on.min(), temp_on.mean(), temp_on.max())
        print("temp_off", temp_off.shape, temp_off.min(), temp_off.mean(), temp_off.max())

        model = sklearn.decomposition.NMF(init='random', n_components=2, solver='mu', alpha=0.01, random_state=8)

        W_on = model.fit_transform(temp_on)
        H_on = model.components_
        print("W_on", W_on.shape, W_on.min(), W_on.mean(), W_on.max())
        print("H_on", H_on.shape, H_on.min(), H_on.mean(), H_on.max())

        W_off = model.fit_transform(temp_off)
        H_off = model.components_
        print("W_off", W_off.shape, W_off.min(), W_off.mean(), W_off.max())
        print("H_off", H_off.shape, H_off.min(), H_off.mean(), H_off.max())

        temp_wtf = sklearn.preprocessing.normalize(temp_off - np.matmul(W_off, H_on))
        temp_wtf2 = sklearn.preprocessing.normalize(temp_off - np.matmul(W_on, H_off))
        temp_wtf3 = sklearn.preprocessing.normalize(temp_off - np.matmul(W_off, H_off))
        print("temp_wtf", temp_wtf.shape, temp_wtf.min(), temp_wtf.mean(), temp_wtf.max())

        temp_clean = sklearn.preprocessing.normalize(temp_on - np.matmul(W_on, H_off))
        print("temp_clean", temp_clean.shape, temp_clean.min(), temp_clean.mean(), temp_clean.max())

        if image_off is None:
            image_off = image[i + 1]
        else:
            image_off = np.concatenate((image_off, image[i + 1]))

        if image_on is None:
            image_on = image[i]
        else:
            image_on = np.concatenate((image_on, image[i]))

        if clean_image is None:
            clean_image = temp_clean
        else:
            clean_image = np.concatenate((clean_image, temp_clean))

        if wtf_image is None:
            wtf_image = temp_wtf
        else:
            wtf_image = np.concatenate((wtf_image, temp_wtf))

        if wtf2_image is None:
            wtf2_image = temp_wtf2
        else:
            wtf2_image = np.concatenate((wtf2_image, temp_wtf2))

        if wtf3_image is None:
            wtf3_image = temp_wtf2
        else:
            wtf3_image = np.concatenate((wtf3_image, temp_wtf3))

        print("image_off", image_off.shape, image_off.min(), image_off.mean(), image_off.max())
        print("image_on", image_on.shape, image_on.min(), image_on.mean(), image_on.max())
        print("clean_image", clean_image.shape, clean_image.min(), clean_image.mean(), clean_image.max())

        print()

    # # plot AAA and BCD separately
    # plt.figure(figsize=(16, 10))
    #
    # plt.subplot(3, 2, 1)
    # plt.imshow(image_on.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    # plt.text(5, 100, 'ON', bbox={'facecolor': 'white'})
    # plt.grid(b=None)
    #
    # plt.subplot(3, 2, 2)
    # plt.imshow(image_off.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    # plt.text(5, 100, 'OFF', bbox={'facecolor': 'white'})
    # plt.grid(b=None)
    #
    # plt.subplot(3, 2, 3)
    # diff = sklearn.preprocessing.normalize(image_on - image_off)
    # plt.imshow(diff.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    # plt.text(5, 100, 'ON Difference', bbox={'facecolor': 'white'})
    # plt.grid(b=None)
    #
    # plt.subplot(3, 2, 5)
    # plt.imshow(clean_image.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    # plt.text(5, 100, 'ON NMF', bbox={'facecolor': 'white'})
    # plt.grid(b=None)
    #
    # plt.subplot(3, 2, 4)
    # plt.imshow(wtf_image.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    # plt.text(5, 100, 'OFF NMF', bbox={'facecolor': 'white'})
    # plt.grid(b=None)
    #
    # plt.subplot(3, 2, 6)
    # plt.imshow(wtf2_image.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    # plt.text(5, 100, 'OFF NMF2', bbox={'facecolor': 'white'})
    # plt.grid(b=None)
    #
    # plt.show()

    # plot full cadence
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    cadence = np.concatenate((image_on[0:273], image_off[0:273], image_on[273:273*2], image_off[273:273*2], image_on[273*2:273*3], image_off[273*2:273*3]))
    # print("cadence", cadence.shape, cadence.min(), cadence.mean(), cadence.max())
    # exit()
    plt.imshow(cadence.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    plt.text(5, 100, 'original', bbox={'facecolor': 'white'})
    plt.grid(b=None)

    plt.subplot(2, 2, 2)
    cadence = np.concatenate((clean_image[0:273], wtf_image[0:273], clean_image[273:273*2], wtf_image[273:273*2], clean_image[273*2:273*3], wtf_image[273*2:273*3]))
    plt.imshow(cadence.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    plt.text(5, 100, 'ON NMF / OFF NMF', bbox={'facecolor': 'white'})
    plt.grid(b=None)

    plt.subplot(2, 2, 3)
    cadence = np.concatenate((clean_image[0:273], wtf2_image[0:273], clean_image[273:273*2], wtf2_image[273:273*2], clean_image[273*2:273*3], wtf2_image[273*2:273*3]))
    plt.imshow(cadence.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    plt.text(5, 100, 'ON NMF / OFF NMF2', bbox={'facecolor': 'white'})
    plt.grid(b=None)

    plt.subplot(2, 2, 4)
    cadence = np.concatenate((clean_image[0:273], wtf3_image[0:273], clean_image[273:273 * 2], wtf3_image[273:273 * 2], clean_image[273 * 2:273 * 3], wtf3_image[273 * 2:273 * 3]))
    plt.imshow(cadence.astype(float), interpolation='antialiased', aspect='auto', cmap='viridis');
    plt.text(5, 100, 'ON NMF / OFF NMF3', bbox={'facecolor': 'white'})
    plt.grid(b=None)

    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    set_seed(8)

    # avg_preprocess_time 0.04009617352485657
    # avg_get_item_time 0.0029651055335998535

    # npy_names = [
    #     "b3fa35ccc4fe",  # 1 - worst
    #     "e22a04c5637b",  # 1 - worst
    #     "32d36db5de2c",  # 0 - worst
    #     "c3fb7abd6a6f",  # 0 - best
    #     "6aa62db73d92",  # 0 - best
    #     "b93c5db0a0de",  # 0 - best
    #     "0a09a787e15e",  # 1 - idk
    #     "0a47af6a917c",  # 1 - idk
    #     "0aaab08f8780",  # 1 - idk
    #     "0ac344e4dfe6",  # 1 - idk
    #
    # ]
    # for npy_name in npy_names:
    #     npy_path = os.path.join(config.train_data_dir, npy_name[0], npy_name + ".npy")
    #     signal = np.load(npy_path)
    #     show_cleaned_image_individually(signal)

    data_df = pd.read_csv(config.train_csv_path)
    data_df = sklearn.utils.shuffle(data_df)
    data_df["target"] = data_df["target"].apply(lambda x: int(x))
    labels = list(data_df["target"])
    npy_paths = list(data_df["path"])

    ds = dataset.SETIDataset(
        labels,
        npy_paths,
        in_channels=1,
        desired_image_size=256,
        interpolation=cv2.INTER_AREA,
        augment=True,
        normalize=False,
    )

    model_nmf = sklearn.decomposition.NMF(init='random', n_components=2, solver='mu', alpha=0.01, random_state=8)

    smth_dir = os.path.join("temp", "smth")
    if not os.path.exists(smth_dir):
        os.makedirs(smth_dir)

    show_amount = 5
    showed = 0
    class_to_show = 0
    avg_preprocess_time = 0
    avg_get_item_time = 0
    for index, row in data_df.iterrows():
        npy_path = row["path"]
        label = int(row["target"])
        npy_name = row["id"]

        if label != class_to_show:
            continue

        if showed == show_amount:
            avg_preprocess_time = avg_preprocess_time / showed
            print("avg_preprocess_time", avg_preprocess_time)

            avg_get_item_time = avg_get_item_time / showed
            print("avg_get_item_time", avg_get_item_time)
            break

        print(npy_name)
        signal = np.load(npy_path)
        title = npy_name + "_label=" + str(label) + "_#" + str(showed + 1)

        sample = {"label": label, "tensor": torch.from_numpy(signal)}
        save_path = os.path.join(smth_dir, title + "_orig.png")
        visualization.visualize_sample(sample, title=title, save_path=save_path)

        start = time.time()
        signal_transformed = preprocess_cadence(signal, model_nmf)
        end = time.time()
        elapsed_time = end - start
        avg_preprocess_time += elapsed_time

        sample["tensor"] = torch.from_numpy(signal_transformed)
        save_path = os.path.join(smth_dir, title + "_processed.png")
        visualization.visualize_sample(sample, title=title, save_path=save_path)

        start = time.time()
        sample = ds[index]
        end = time.time()
        elapsed_time = end - start
        avg_get_item_time += elapsed_time

        # show_cleaned_image_individually(signal, title=title)

        showed += 1
