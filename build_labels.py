import os
import pandas as pd
import numpy as np
import shutil

import config


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    labels_df = pd.read_csv("./data/train_labels.csv")
    print(labels_df.shape)

    npy_paths = []
    subdir_names = [str(i) for i in range(10)] + ["a", "b", "c", "d", "e", "f"]
    # for index, row in labels_df.iterrows():
    #     npy_id = row["id"]
    #
    #     for subdir_name in subdir_names:
    #         possible_npy_name = npy_id + ".npy"
    #         possible_npy_path = os.path.join(config.train_data_dir, subdir_name, possible_npy_name)
    #
    #         if os.path.exists(possible_npy_path):
    #             npy_paths.append(possible_npy_path)
    #             # print(possible_npy_path)
    #             # exit(0)
    #
    # labels_df["path"] = npy_paths
    # csv_path = os.path.join(config.data_dir, "train.csv")
    # labels_df.to_csv(csv_path)

    # Create test.csv
    npy_paths = []
    npy_ids = []
    test_dir = os.path.join(config.data_dir, "test")
    for subdir_name in subdir_names:
        dir_path = os.path.join(test_dir, subdir_name)
        print("Processing", dir_path)
        for index, filename in enumerate(os.listdir(dir_path)):
            npy_name, ext = os.path.splitext(filename)
            npy_path = os.path.join(dir_path, filename)

            if ext == ".npy":
                npy_paths.append(npy_path)
                npy_ids.append(npy_name)
            else:
                print("There is not *.npy file", npy_path)

            print('\r', "Progress {}".format(index), end='')
        print()

    data_test = {
        'id': npy_ids,
        'target': [0.5] * len(npy_ids),
        "path": npy_paths,
    }
    df_test = pd.DataFrame(data_test, columns=['id', 'target', 'path'])
    df_test.to_csv(os.path.join(config.data_dir, "test.csv"))

