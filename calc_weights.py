import os
import pandas as pd
import ast
import numpy as np

import config


if __name__ == '__main__':
    import __main__
    print("Run of", __main__.__file__)

    class_amount = 2

    calc_type = "max"  # {"mean", "max"}

    labels_df = pd.read_csv(config.labels_csv_path)

    images_per_class = {
        1: 0,
        0: 0,
    }

    for index, row in labels_df.iterrows():
        images_per_class[int(row["target"])] += 1

    weights = []

    if calc_type == "max":
        for class_id in images_per_class.keys():
            weight = max(images_per_class.values()) / images_per_class[class_id]
            weights.append(weight)
            print(class_id, images_per_class[class_id], weight)

        print(weights)

        divisor = sum(weights) / class_amount
        for i in range(class_amount):
            weights[i] /= divisor
        print(weights)
        print(sum(weights))
    elif calc_type == "mean":
        image_values = np.array(list(images_per_class.values()))
        weights = ([image_values.mean()] * class_amount) / image_values
        print(weights)
        print(weights.sum())
    else:
        print("smth goes wrong")
        print("check calc_type variable")
        print()
