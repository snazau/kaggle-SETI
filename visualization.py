import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import config
import dataset
import predict
import seti_model


def visualize_sample(sample, title=None, save_path=None):
    tensor = sample["tensor"]
    channel_amount, height, width = tensor.shape
    ndarray = tensor.cpu().numpy()

    if title is None:
        title = str(sample["label"])

    if channel_amount == 2:
        ndarray = np.vstack(ndarray)
        resized_shape = (512, 512)
        ndarray = cv2.resize(ndarray, resized_shape, interpolation=cv2.INTER_AREA)
        ndarray = ndarray[np.newaxis, :, :]
        channel_amount = 1

    if channel_amount == 6:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(title)
        for i in range(channel_amount):
            plt.subplot(channel_amount, 1, i+1)
            plt.imshow(ndarray[i].astype(float), aspect='auto')
        fig.text(0.5, 0.04, 'Frequency ➡', ha='center', fontsize=16)
        fig.text(0.04, 0.5, '⬅ Time', va='center', rotation='vertical', fontsize=16)
    elif channel_amount == 1:
        fig = plt.figure()
        fig.suptitle(title)
        plt.imshow(ndarray[0])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def visualize_train_data(labels_csv_path, visualization_dir):
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    train_df = pd.read_csv(labels_csv_path)
    npy_paths_train = train_df["path"].values
    labels_train = train_df["target"].values
    npy_ids_train = train_df["id"].values

    print("labels", type(labels_train), labels_train.shape)

    ds = dataset.SETIDataset(
        labels_train,
        npy_paths_train,
        in_channels=6,
        desired_image_size=273,
        normalize=False,
        augment=False
    )
    samples_amount = len(ds)
    for index, sample in enumerate(ds):
        label = sample["label"]
        npy_path = sample["npy_path"]

        class_dir = os.path.join(visualization_dir, str(label))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        filename = os.path.basename(npy_path)
        image_name, ext = os.path.splitext(filename)

        title = "label={}".format(label)
        visualization_path = os.path.join(class_dir, image_name + "_" + title + ".jpg")
        if not os.path.exists(visualization_path):
            visualize_sample(sample, title=title, save_path=visualization_path)
        print('\r', "Progress {}/{}".format(index, samples_amount), end='')
        # break
    print()


def visualize_top_preds(checkpoint_path, labels_csv_path, visualization_dir, visualize_best=False):
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    df = pd.read_csv(labels_csv_path)
    npy_paths = df["path"].values
    labels = df["target"].values
    npy_ids = df["id"].values

    pred_probs, pred_labels, pred_npy_paths = predict.get_preds_from_checkpoint(checkpoint_path, labels, npy_paths)
    print("pred_probs", pred_probs)
    print("pred_labels", pred_labels)
    # exit(0)

    sorted_indices = np.argsort(np.abs(pred_labels - pred_probs))
    sorted_indices = sorted_indices[::-1]

    pred_npy_paths = [pred_npy_paths[i] for i in sorted_indices]
    pred_probs = pred_probs[sorted_indices]
    pred_labels = pred_labels[sorted_indices]

    # Worst preds
    print("Visualizing worst predictions")
    worst_dir = os.path.join(visualization_dir, "worst")
    if not os.path.exists(worst_dir):
        os.makedirs(worst_dir)

    amount_to_visualize = int(0.1 * len(pred_npy_paths))
    for index, (pred_npy_path, pred_prob, pred_label) in enumerate(zip(pred_npy_paths, pred_probs, pred_labels)):
        if index == amount_to_visualize:
            break

        signal = np.load(pred_npy_path).astype(np.float32)
        channel_amount, height, width = signal.shape
        signal = np.pad(signal, ((0, 0), (0, 0), (0, config.desired_image_size - width)), 'constant')  # [6 x 273 x 273]
        sample = {
            "tensor": torch.from_numpy(signal),
            "label": pred_label,
            "npy_path": pred_npy_path,
        }

        filename = os.path.basename(pred_npy_path)
        image_name, ext = os.path.splitext(filename)

        # print("pred_prob", pred_prob)
        # print("pred_label", pred_label)
        # print("npy_path", pred_npy_path)
        # exit(0)

        # assert pred_label in [0, 1]

        title = "error={:09.8f}_label={:09.8f}_pred={:09.8f}".format(abs(pred_label - pred_prob), pred_label, pred_prob)
        visualization_path = os.path.join(worst_dir, title + "_" + image_name + ".jpg")
        if not os.path.exists(visualization_path):
            visualize_sample(sample, title=title, save_path=visualization_path)
        print('\r', "Progress {}/{}".format(index, amount_to_visualize), end='')
    print()

    if visualize_best is False:
        return

    # Best preds
    print("Visualizing best predictions")

    pred_npy_paths = pred_npy_paths[::-1]
    pred_probs = pred_probs[::-1]
    pred_labels = pred_labels[::-1]

    best_dir = os.path.join(visualization_dir, "best")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    amount_to_visualize = int(0.1 * len(pred_npy_paths))
    for index, (pred_npy_path, pred_prob, pred_label) in enumerate(zip(pred_npy_paths, pred_probs, pred_labels)):
        if index == amount_to_visualize:
            break

        signal = np.load(pred_npy_path).astype(np.float32)
        channel_amount, height, width = signal.shape
        signal = np.pad(signal, ((0, 0), (0, 0), (0, config.desired_image_size - width)), 'constant')  # [6 x 273 x 273]
        sample = {
            "tensor": torch.from_numpy(signal),
            "label": pred_label,
            "npy_path": pred_npy_path,
        }

        filename = os.path.basename(pred_npy_path)
        image_name, ext = os.path.splitext(filename)

        # print("pred_prob", pred_prob)
        # print("pred_label", pred_label)
        # print("npy_path", pred_npy_path)
        # exit(0)

        # assert pred_label in [0, 1]

        title = "error={:09.8f}_label={:09.8f}_pred={:09.8f}".format(abs(pred_label - pred_prob), pred_label, pred_prob)
        visualization_path = os.path.join(best_dir, title + "_" + image_name + ".jpg")
        if not os.path.exists(visualization_path):
            visualize_sample(sample, title=title, save_path=visualization_path)
        print('\r', "Progress {}/{}".format(index, amount_to_visualize), end='')
    print()


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    labels_csv_path = config.labels_csv_path
    # visualization_dir = os.path.join(".", "visualization", "train_data")
    # visualize_train_data(labels_csv_path, visualization_dir)

    run_name = "May21_20-12-15_model=tf_efficientnetv2_s_in21k_pretrained=True_aug=True_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    checkpoints_dir = "/media/tower/nvme/kaggle/SETI_bin_Class/checkpoints/" + run_name + "/fold5"
    checkpoint_path = os.path.join(checkpoints_dir, "best_loss_val_e12_" + run_name + ".pth.tar")
    visualization_dir = os.path.join(".", "visualization", run_name)
    visualize_top_preds(checkpoint_path, labels_csv_path, visualization_dir, visualize_best=True)
