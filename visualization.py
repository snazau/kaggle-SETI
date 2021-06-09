import cv2
import cuml
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import config
import dataset
import predict


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
            plt.imshow(ndarray[i].astype(float), aspect="auto")
        fig.text(0.5, 0.04, "Frequency ➡", ha="center", fontsize=16)
        fig.text(0.04, 0.5, "⬅ Time", va="center", rotation="vertical", fontsize=16)
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
        print("\r", "Progress {}/{}".format(index, samples_amount), end="")
        # break
    print()


def visualize_top_preds(checkpoints_paths, labels_csv_path, visualization_dir, visualize_best=False):
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    df = pd.read_csv(labels_csv_path)
    npy_paths = df["path"].values
    labels = df["target"].values
    npy_ids = df["id"].values

    # pred_probs, pred_labels, pred_npy_paths = predict.get_preds_from_checkpoint(checkpoint_path, labels, npy_paths)
    pred_probs, pred_labels, pred_npy_paths, df_submission = predict.get_ensemble_preds(checkpoints_paths, labels_csv_path)
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
        signal = np.pad(signal, ((0, 0), (0, 0), (0, config.desired_image_size - width)), "constant")  # [6 x 273 x 273]
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
        print("\r", "Progress {}/{}".format(index, amount_to_visualize), end="")
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
        signal = np.pad(signal, ((0, 0), (0, 0), (0, config.desired_image_size - width)), "constant")  # [6 x 273 x 273]
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
        print("\r", "Progress {}/{}".format(index, amount_to_visualize), end="")
    print()


def visualize_embeddings(checkpoint_path, visualization_dir, tsne_perplexity):
    # Core code was taken from:
    # https://www.kaggle.com/ttahara/eda-seti-e-t-train-v-s-test-by-cnn-embeddings#Extract-Embeddings-(and-Prediction)

    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # Get train embeddings
    df_train = pd.read_csv(config.train_csv_path)
    npy_paths_train = df_train["path"].values
    labels_train = df_train["target"].values
    npy_ids_train = df_train["id"].values

    probs_train, labels_train, npy_paths_train, embeddings_train = predict.get_preds_from_checkpoint(
        checkpoint_path,
        labels_train,
        npy_paths_train,
        calculate_embeddings=True
    )
    print("embeddings_train", embeddings_train.shape)

    # Get test embeddings
    df_test = pd.read_csv(config.test_csv_path)
    npy_paths_test = df_test["path"].values
    labels_test = df_test["target"].values
    npy_ids_test = df_test["id"].values

    probs_test, labels_test, npy_paths_test, embeddings_test = predict.get_preds_from_checkpoint(
        checkpoint_path,
        labels_test,
        npy_paths_test,
        calculate_embeddings=True
    )
    print("embeddings_test", embeddings_test.shape)

    # Grouping
    all_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    all_df["target"].value_counts()
    all_df["data_type"] = ""
    all_df.loc[all_df["target"] == 1.0, "data_type"] = "train_pos"
    all_df.loc[all_df["target"] == 0.0, "data_type"] = "train_neg"
    all_df.loc[all_df["target"] == 0.5, "data_type"] = "test"
    all_df["data_type"].value_counts()

    # tsne
    all_embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)
    all_probs = np.concatenate([probs_train, probs_test], axis=0)
    print("all_embeddings", all_embeddings.shape)
    print("all_probs", all_probs.shape)

    tsne = cuml.TSNE(n_components=2, perplexity=tsne_perplexity)
    all_embeddings_2d = tsne.fit_transform(all_embeddings)
    print("all_embeddings_2d", all_embeddings_2d.shape)

    neg_embeddings_2d = all_embeddings_2d[all_df.query("data_type == 'train_neg'").index.values]
    pos_embeddings_2d = all_embeddings_2d[all_df.query("data_type == 'train_pos'").index.values]
    test_embeddings_2d = all_embeddings_2d[all_df.query("data_type == 'test'").index.values]

    # Plot train 2d embeddings
    fig = plt.figure(figsize=(30, 10))
    ax_neg = fig.add_subplot(1, 3, 1)
    ax_pos = fig.add_subplot(1, 3, 2)
    ax_posneg = fig.add_subplot(1, 3, 3)

    ax_neg.scatter(neg_embeddings_2d[:, 0], neg_embeddings_2d[:, 1], color="red", s=10, label="train_negative", alpha=0.3)
    ax_neg.legend(fontsize=13)
    ax_neg.set_title("train_negative", fontsize=18)

    ax_pos.scatter(pos_embeddings_2d[:, 0], pos_embeddings_2d[:, 1], color="blue", s=10, label="", alpha=0.3)
    ax_pos.legend(fontsize=13)
    ax_pos.set_title("train_positive", fontsize=18)

    ax_posneg.scatter(neg_embeddings_2d[:, 0], neg_embeddings_2d[:, 1], color="red", s=10, label="train_negative", alpha=0.3)
    ax_posneg.scatter(pos_embeddings_2d[:, 0], pos_embeddings_2d[:, 1], color="blue", s=10, label="train_positive", alpha=0.3)
    ax_posneg.legend(fontsize=13)
    ax_posneg.set_title("train_all", fontsize=18)

    visualization_path = os.path.join(visualization_dir, "train_tsne_perplexity=" + str(tsne_perplexity) + ".png")
    plt.savefig(visualization_path)

    # plot train and test 2d embeddings
    fig = plt.figure(figsize=(50, 10))

    ax_posneg = fig.add_subplot(1, 5, 1)
    ax_test = fig.add_subplot(1, 5, 2)
    ax_negtest = fig.add_subplot(1, 5, 3)
    ax_postest = fig.add_subplot(1, 5, 4)
    ax_all = fig.add_subplot(1, 5, 5)

    ax_posneg.scatter(neg_embeddings_2d[:, 0], neg_embeddings_2d[:, 1], color="red", s=10, label="train_negative", alpha=0.3)
    ax_posneg.scatter(pos_embeddings_2d[:, 0], pos_embeddings_2d[:, 1], color="blue", s=10, label="train_positive", alpha=0.3)
    ax_posneg.legend(fontsize=13)
    ax_posneg.set_title("train_all", fontsize=18)

    ax_test.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], color="limegreen", s=10, label="test_examples", alpha=0.3)
    ax_test.legend(fontsize=13)
    ax_test.set_title("examples in Test", fontsize=18)

    ax_negtest.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], color="limegreen", s=10, label="test_examples", alpha=0.3)
    ax_negtest.scatter(neg_embeddings_2d[:, 0], neg_embeddings_2d[:, 1], color="red", s=10, label="train_negative", alpha=0.3)
    ax_negtest.legend(fontsize=13)
    ax_negtest.set_title("test vs train_negative", fontsize=18)

    ax_postest.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], color="limegreen", s=10, label="test_examples", alpha=0.3)
    ax_postest.scatter(pos_embeddings_2d[:, 0], pos_embeddings_2d[:, 1], color="blue", s=10, label="train_positive", alpha=0.3)
    ax_postest.legend(fontsize=13)
    ax_postest.set_title("test vs train_positive", fontsize=18)

    ax_all.scatter(test_embeddings_2d[:, 0], test_embeddings_2d[:, 1], color="limegreen", s=10, label="test_examples", alpha=0.3)
    ax_all.scatter(neg_embeddings_2d[:, 0], neg_embeddings_2d[:, 1], color="red", s=10, label="train_negative", alpha=0.3)
    ax_all.scatter(pos_embeddings_2d[:, 0], pos_embeddings_2d[:, 1], color="blue", s=10, label="train_positive", alpha=0.3)
    ax_all.legend(fontsize=13)
    ax_all.set_title("test vs train_all", fontsize=18)

    visualization_path = os.path.join(visualization_dir, "traintest_tsne_perplexity=" + str(tsne_perplexity) + ".png")
    plt.savefig(visualization_path)


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    labels_csv_path = config.train_csv_path
    # visualization_dir = os.path.join(".", "visualization", "train_data")
    # visualize_train_data(labels_csv_path, visualization_dir)

    # run_name = "May21_20-12-15_model=tf_efficientnetv2_s_in21k_pretrained=True_aug=True_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    run_name = "May31_23-03-41_model=tf_efficientnetv2_s_in21k_pretrained=T_c=1_size=256_aug=T_nrmlz=meanstd_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR_MixUp"

    # Visualize embeddings
    checkpoints_dir = "/media/tower/nvme/kaggle/SETI_bin_Class/checkpoints/" + run_name + "/fold5"
    checkpoint_path = os.path.join(checkpoints_dir, "best_loss_val_e10_" + run_name + ".pth.tar")
    # checkpoints_paths = [checkpoint_path]

    visualization_dir = os.path.join(".", "visualization", "embeddings_" + run_name)
    visualize_embeddings(checkpoint_path, visualization_dir, tsne_perplexity=50.0)

    # Visualize top preds
    # best_metric = "best_loss_val"
    # cv_checkpoints_dir = os.path.join(".", "checkpoints", run_name)
    # checkpoints_paths = predict.get_best_k_checkpoint_paths(cv_checkpoints_dir, best_metric=best_metric)
    #
    # visualization_dir = os.path.join(".", "visualization", run_name)
    # visualize_top_preds(checkpoints_paths, labels_csv_path, visualization_dir, visualize_best=True)
