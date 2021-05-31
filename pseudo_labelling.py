import os
import pandas as pd

import config


def pseudo_label(submission_path, confidence_threshold, only_positive=True):
    assert confidence_threshold > 0.5, "confidence_threshold must be greater than 0.5 but its {}".format(confidence_threshold)

    df = pd.read_csv(submission_path)

    base_image_path = config.test_data_dir
    pseudo_targets = []
    pseudo_ids = []
    pseudo_npy_paths = []
    pseudo_pred = []
    for index, row in df.iterrows():
        id = row["id"]
        pred = row["target"]

        if pred > confidence_threshold or pred < (1 - confidence_threshold):
            npy_path = os.path.join(base_image_path, id[0], id + ".npy")
            pseudo_target = 1 if pred > confidence_threshold else 0

            df.loc[index, "target"] = pseudo_target  # threshold submission
            # print('df["target"][index]', df["target"][index])
            # print('df["target"][id]', df["id"][index])
            # exit(0)

            if only_positive is True and pred < (1 - confidence_threshold):
                continue

            pseudo_targets.append(pseudo_target)
            pseudo_ids.append(id)
            pseudo_npy_paths.append(npy_path)
            pseudo_pred.append(pred)

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

    return pseudo_df, thresholded_df


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    run_name = "May21_20-12-15_model=tf_efficientnetv2_s_in21k_pretrained=True_aug=True_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    submission_name = "best_loss_val_" + run_name
    submission_path = os.path.join(".", "submissions", "submitted",  submission_name + ".csv")
    confidence_threshold = 0.99
    pseudo_df, thresholded_df = pseudo_label(submission_path, confidence_threshold=confidence_threshold, only_positive=True)

    submission_thresholded_path = submission_path = os.path.join(".", "submissions",  "threshold=" + str(confidence_threshold) + "_" + submission_name + ".csv")
    thresholded_df.to_csv(submission_thresholded_path)

    pseudo_df_path = os.path.join(config.data_dir, "pseudo_" + "threshold=" + str(confidence_threshold) + "_" + run_name + ".csv")
    pseudo_df.to_csv(pseudo_df_path, index=False)
