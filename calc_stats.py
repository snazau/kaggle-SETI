import numpy as np
import os
import pandas as pd
import torch

import config
import visualization

if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    ndarray_mean = np.zeros((6, 273, 256))
    ndarray_std = np.zeros((6, 273, 256))

    labels_df = pd.read_csv(config.labels_csv_path)
    # print(np.unique(labels_df["target"], return_counts=True))
    labels_df["group"] = labels_df["id"].apply(lambda x: x[0])

    # df = labels_df[["id", "target", "group"]].groupby("group")["target"].nunique()
    # print(df)
    print(labels_df.pivot_table(index=['group'], columns='target', aggfunc='size', fill_value=0))

    # labels_df[["id", "target"]].hist(by=labels_df["group"])
    import matplotlib.pyplot as plt
    plt.show()
    exit()

    ndarray_amount = labels_df.shape[0]

    print("Calculating mean ndarray")
    for index, row in labels_df.iterrows():
        npy_path = row["path"]
        ndarray = np.load(npy_path).astype(np.float32)
        ndarray_mean += ndarray

        print('\r', "Progress {}".format(index), end='')
    print()

    ndarray_mean /= ndarray_amount
    print("ndarray_mean", ndarray_mean.shape, ndarray_mean.min(), ndarray_mean.mean(), ndarray_mean.max())
    print()

    visualization.visualize_sample({
        "tensor": torch.from_numpy(ndarray_mean),
        "label": 0,
        "npy_path": "qwerty"
    })

    ndarray_mean_path = os.path.join(config.data_dir, "train_mean.npy")
    np.save(ndarray_mean_path, ndarray_mean)

    print("Calculating std ndarray")
    for index, row in labels_df.iterrows():
        npy_path = row["path"]
        ndarray = np.load(npy_path).astype(np.float32)
        ndarray_std += (ndarray - ndarray_mean) ** 2

        print('\r', "Progress {}".format(index), end='')
    print()

    ndarray_std /= (ndarray_amount - 1)
    ndarray_std = np.sqrt(ndarray_std)
    print("ndarray_std", ndarray_std.shape, ndarray_std.min(), ndarray_std.mean(), ndarray_std.max())
    print()

    visualization.visualize_sample({
        "tensor": torch.from_numpy(ndarray_std),
        "label": 0,
        "npy_path": "qwerty"
    })

    ndarray_std_path = os.path.join(config.data_dir, "train_std.npy")
    np.save(ndarray_std_path, ndarray_std)
