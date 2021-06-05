import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import plotly.express as px

import config


def pseudo_label(submission_path, upper_threshold, lower_threshold, only_positive=True):
    assert lower_threshold < upper_threshold
    assert 0 < lower_threshold < 1
    assert 0 < upper_threshold < 1

    df = pd.read_csv(submission_path)

    fig = px.histogram(df, x="target")
    fig.show()

    zeros_amount = 0
    ones_amount = 0

    base_image_path = config.test_data_dir
    pseudo_targets = []
    pseudo_ids = []
    pseudo_npy_paths = []
    pseudo_pred = []
    test_size = df.shape[0]
    for index, row in df.iterrows():
        id = row["id"]
        pred = row["target"]

        if pred > upper_threshold or pred < lower_threshold:
            npy_path = os.path.join(base_image_path, id[0], id + ".npy")
            pseudo_target = 1 if pred > upper_threshold else 0

            df.loc[index, "target"] = pseudo_target  # threshold submission
            # print('df["target"][index]', df["target"][index])
            # print('df["target"][id]', df["id"][index])
            # exit(0)

            if only_positive is True and pred < lower_threshold:
                continue

            if pseudo_target == 0:
                zeros_amount += 1
            if pseudo_target == 1:
                ones_amount += 1

            pseudo_targets.append(pseudo_target)
            pseudo_ids.append(id)
            pseudo_npy_paths.append(npy_path)
            pseudo_pred.append(pred)

            print('\r', "Progress {}/{} ones={} zeros={}".format(index, test_size, ones_amount, zeros_amount), end='')
    print()

    pseudo_data = {
        "id": pseudo_ids,
        "target": pseudo_targets,
        "pred": pseudo_pred,
        "path": pseudo_npy_paths,
    }
    pseudo_df = pd.DataFrame(pseudo_data, columns=["id", "target", "pred", "path"])

    thresholded_df = df
    smth = list(thresholded_df["target"])
    print("zero amount", len([x for x in smth if x == 0]))
    print("ones amount", len([x for x in smth if x == 1]))
    print("(0, 1)", len([x for x in smth if 0 < x < 1]))

    print('pseudo_df["target"] stats:')
    print(np.unique(pseudo_df["target"].values, return_counts=True))

    return pseudo_df, thresholded_df


def merge_test_with_pseudo(df_train, df_pseudo, dataset_columns=["id", "target", "path"]):
    df_merged = pd.concat([df_train[dataset_columns], df_pseudo[dataset_columns]])
    return df_merged


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    curr_date = datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    # run_date = "May21_20-12-15"
    # run_name = "May21_20-12-15_model=tf_efficientnetv2_s_in21k_pretrained=True_aug=True_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    run_date = "May31_23-03-41"
    run_name = "May31_23-03-41_model=tf_efficientnetv2_s_in21k_pretrained=T_c=1_size=256_aug=T_nrmlz=meanstd_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR_MixUp"

    submission_name = "best_loss_val_" + run_name
    submission_path = os.path.join(".", "submissions", "submitted",  submission_name + ".csv")

    upper_threshold = 0.9
    lower_threshold = 0.002
    only_positive = False
    pseudo_df, thresholded_df = pseudo_label(submission_path, upper_threshold=upper_threshold, lower_threshold=lower_threshold, only_positive=only_positive)

    # submission_thresholded_path = submission_path = os.path.join(".", "submissions",  "threshold=" + str(confidence_threshold) + "_" + submission_name + ".csv")
    # thresholded_df.to_csv(submission_thresholded_path)

    thresholds_str = str(upper_threshold) + (("," + str(lower_threshold)) if only_positive is False else "")
    pseudo_df_path = os.path.join(config.data_dir, "pseudo_at=" + curr_date + "_with=" + run_date + "_threshold=" + thresholds_str + ".csv")
    pseudo_df.to_csv(pseudo_df_path, index=False)

    train_df = pd.read_csv(config.train_csv_path)
    merged_df = merge_test_with_pseudo(train_df, pseudo_df, dataset_columns=["id", "target", "path"])
    merged_df_path = os.path.join(config.data_dir, "train_pseudo_merged_at=" + curr_date + "_with=" + run_date + "_threshold=" + thresholds_str + ".csv")
    merged_df.to_csv(merged_df_path, index=False)
