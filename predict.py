import os
import numpy as np
import pandas as pd
import torch


import config
import dataset
import seti_model


def predict(dataloader, model):
    model.eval()

    # Pseudo labelling stats
    pseudo_1_95 = 0
    pseudo_0_05 = 0
    pseudo_1_99 = 0
    pseudo_0_01 = 0

    all_probs = np.array([])
    all_labels = np.array([])
    all_npy_paths = []
    for index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device, config.dtype)
        labels = sample["label"]
        npy_paths = sample["npy_path"]

        all_labels = np.concatenate([all_labels, labels])
        all_npy_paths = all_npy_paths + list(npy_paths)

        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
        all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])

        pseudo_1_95 += (probs[probs > 0.95] > 0).sum()
        pseudo_1_99 += (probs[probs > 0.99] > 0).sum()
        pseudo_0_05 += (probs[probs < 0.05] < 1).sum()
        pseudo_0_01 += (probs[probs < 0.01] < 1).sum()

        print('\r', "Progress {}".format(index), end='')
        if config.debug is True:
            break
    print()
    print("pseudo_1_95", pseudo_1_95)
    print("pseudo_1_99", pseudo_1_99)
    print("pseudo_0_05", pseudo_0_05)
    print("pseudo_0_01", pseudo_0_01)

    return all_probs, all_labels, all_npy_paths


def get_preds_from_checkpoint(checkpoint_path, labels_test, npy_paths_test):
    # Get checkpoint and dataset settings
    checkpoint = load_checkpoint(checkpoint_path)
    in_channels = checkpoint["model"]["in_channels"] if "in_channels" in checkpoint["model"] else 6
    desired_image_size = checkpoint["model"]["desired_image_size"] if "desired_image_size" in checkpoint["model"] else 273
    normalize = checkpoint["model"]["normalize"] if "normalize" in checkpoint["model"] else False

    # Get dataset and dataloader
    dataset_test = dataset.SETIDataset(
        labels_test,
        npy_paths_test,
        in_channels=in_channels,
        desired_image_size=desired_image_size,
        normalize=normalize,
        augment=False,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.n_loader_workers,
        pin_memory=True
    )
    print("Images in test: " + str(len(dataset_test)))
    print("Dataloader test len: " + str(len(dataloader_test)))
    print()

    # Get model
    model = load_model(checkpoint)
    print("Model on GPU:", next(model.parameters()).is_cuda)

    # Get preds
    probs, labels, npy_paths = predict(dataloader_test, model)
    return probs, labels, npy_paths


def get_ensemble_preds(checkpoint_paths, test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    npy_paths_test = test_df["path"].values
    labels_test = test_df["target"].values
    npy_ids_test = test_df["id"].values

    models_amount = 0
    ensemble_probs = np.zeros_like(labels_test)
    for checkpoint_path in checkpoint_paths:
        models_amount += 1
        probs, labels, npy_paths = get_preds_from_checkpoint(checkpoint_path, labels_test, npy_paths_test)
        ensemble_probs += probs
    ensemble_probs /= float(models_amount)

    data_submission = {
        'id': npy_ids_test,
        'target': ensemble_probs,
    }
    df_submission = pd.DataFrame(data_submission, columns=['id', 'target'])

    return ensemble_probs, df_submission


def get_best_k_checkpoint_paths(cv_checkpoints_dir, best_metric="best_loss_val"):
    checkpoint_paths = []
    for dir_name in os.listdir(cv_checkpoints_dir):
        if "fold" not in dir_name:
            continue

        fold_dir = os.path.join(cv_checkpoints_dir, dir_name)
        for checkpoint_name in os.listdir(fold_dir):
            if best_metric not in checkpoint_name:
                continue

            if not checkpoint_name.endswith(".pth.tar"):
                continue

            checkpoint_path = os.path.join(fold_dir, checkpoint_name)
            checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    print("Loaded checkpoint", checkpoint_path)
    return checkpoint


def load_model(checkpoint):
    model_name = checkpoint["model"]["name"]

    in_channels = checkpoint["model"]["in_channels"] if "in_channels" in checkpoint["model"] else 6
    model = seti_model.SETIModel(model_name, in_channels=in_channels, num_classes=1, pretrained=False)
    model = model.to(device=config.device)
    print("Model loaded successfully")

    model.load_state_dict(checkpoint["model"]["state_dict"])

    desired_image_size = checkpoint["model"]["desired_image_size"] if "desired_image_size" in checkpoint["model"] else 273
    normalize = checkpoint["model"]["normalize"] if "normalize" in checkpoint["model"] else False
    print("Loaded weights into model")
    print("model_name", model_name)
    print("epoch", checkpoint["epoch"])
    print("in_channels", in_channels)
    print("desired_image_size", desired_image_size)
    print("normalize", normalize)
    print("val_loss", checkpoint["val_loss"])
    print("train_loss", checkpoint["train_loss"])
    print("auc_val", checkpoint["auc_val"])
    print("auc_train", checkpoint["auc_train"])

    return model


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    # run_name = "May17_17-54-12_model=dm_nfnet_f0_lr=0.0001_posweights=[9.687047294418406]_loss=BCE"
    # run_name = "May21_01-00-37_model=dm_nfnet_f0_pretrained=True_aug=True_lr=0.0001_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    # run_name = "May20_19-20-31_model=efficientnet_b0_pretrained=True_aug=True_lr=0.0005_bs=64_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    # run_name = "May21_20-12-15_model=tf_efficientnetv2_s_in21k_pretrained=True_aug=True_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    # run_name = "May30_00-15-15_model=tf_efficientnetv2_s_in21k_pretrained=T_c=1_size=256_aug=T_nrmlz=meanstd_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR"
    # run_name = "May30_16-58-16_model=tf_efficientnetv2_s_in21k_pretrained=T_c=1_size=256_aug=T_nrmlz=meanstd_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR_addCoarseDropout"
    run_name = "May30_23-34-30_model=tf_efficientnetv2_s_in21k_pretrained=T_c=1_size=256_aug=T_nrmlz=meanstd_lr=0.0005_bs=32_weights=[1.0]_loss=BCE_scheduler=CosineAnnealingLR_CoarseDropout_RndPermut"

    best_metric = "best_loss_val"
    cv_checkpoints_dir = os.path.join(".", "checkpoints", run_name)
    cv_checkpoint_paths = get_best_k_checkpoint_paths(cv_checkpoints_dir, best_metric=best_metric)
    ensemble_probs, df_submission = get_ensemble_preds(cv_checkpoint_paths, config.test_csv_path)

    # Save submission
    if not os.path.exists(config.submissions_dir):
        os.makedirs(config.submissions_dir)

    df_submission_path = os.path.join(config.submissions_dir, best_metric + "_" + run_name + ".csv")
    df_submission.to_csv(df_submission_path, index=False)
    print("Submission saved to:", df_submission_path)



