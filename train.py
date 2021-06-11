import numpy as np
import os
import pandas as pd
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold

import augmentations
from binary_focal_loss import BinaryFocalWithLogitsLoss
import config
import dataset
import seti_model


def train(dataloader, model, criterion, optimizer, scheduler, epoch, fold, writer):
    model.train()

    loss_avg = 0
    all_labels = np.array([])
    all_probs = np.array([])

    for index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device, config.dtype)
        labels = sample["label"].to(config.device, config.dtype)
        npy_paths = sample["npy_path"]

        optimizer.zero_grad()
        if config.mix_strategy is not None:
            if config.mix_strategy is "MixUp":
                mixed_inputs, labels, labels_shuffled, lam = augmentations.mixup(inputs, labels, alpha=config.mixup_alpha)
            elif config.mix_strategy is "FMix":
                mixed_inputs, labels, labels_shuffled, lam = augmentations.fmix(inputs, labels, alpha=1.0, decay_power=5.0, shape=(256, 256), device=config.device)
            else:
                raise NotImplementedError
            mixed_outputs = model(mixed_inputs)
            loss = lam * criterion(mixed_outputs, labels.unsqueeze(1)) + (1 - lam) * criterion(mixed_outputs, labels_shuffled.unsqueeze(1))
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if config.lr_scheduler_name == "OneCycleLR":
            scheduler.step()

        loss_avg += loss.item()

        if config.mix_strategy is not None:
            with torch.no_grad():
                outputs = model(inputs)
        probs = torch.sigmoid(outputs)

        all_labels = np.concatenate([all_labels, labels.cpu().detach().numpy()])
        all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])

        if index % config.print_freq == 0:
            print("Fold {:2d} Epoch: {:3d} Batch: {:3d}/{:3d} avg_loss: {:10.6f} lr: {:10.8f}".format(
                fold + 1,
                epoch + 1,
                index + 1,
                len(dataloader),
                loss_avg / (index + 1),
                optimizer.param_groups[0]["lr"],
            ))

        if config.debug is True:
            break

    loss_avg /= len(dataloader)
    writer.add_scalars("fold" + str(fold + 1) + "/loss", {"train": loss_avg}, epoch)

    metrics = calc_metrics(all_probs, all_labels)
    for metric_name in metrics.keys():
        metric_value = metrics[metric_name]
        writer.add_scalars("fold" + str(fold + 1) + "/" + metric_name, {"train": metric_value}, epoch)

    return loss_avg, metrics


def validate(dataloader, model, criterion, optimizer, epoch, fold, writer):
    model.eval()

    loss_avg = 0
    all_labels = np.array([])
    all_probs = np.array([])

    for index, sample in enumerate(dataloader):
        inputs = sample["tensor"].to(config.device, config.dtype)
        labels = sample["label"].to(config.device, config.dtype)
        npy_paths = sample["npy_path"]

        with torch.no_grad():
            if config.mix_strategy is not None:
                if config.mix_strategy is "MixUp":
                    mixed_inputs, labels, labels_shuffled, lam = augmentations.mixup(inputs, labels, alpha=config.mixup_alpha)
                elif config.mix_strategy is "FMix":
                    mixed_inputs, labels, labels_shuffled, lam = augmentations.fmix(inputs, labels, alpha=1.0, decay_power=5.0, shape=(256, 256), device=config.device)
                else:
                    raise NotImplementedError
                mixed_outputs = model(mixed_inputs)
                loss = lam * criterion(mixed_outputs, labels.unsqueeze(1)) + (1 - lam) * criterion(mixed_outputs, labels_shuffled.unsqueeze(1))

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)

        if config.mix_strategy is None:
            loss = criterion(outputs, labels.unsqueeze(1))
        loss_avg += loss.item()

        all_labels = np.concatenate([all_labels, labels.cpu().detach().numpy()])
        all_probs = np.concatenate([all_probs, probs[:, 0].cpu().detach().numpy()])

        if index % config.print_freq == 0:
            print("Fold {:2d} Epoch: {:3d} Batch: {:3d}/{:3d} avg_loss: {:10.6f} lr: {:10.8f}".format(
                fold + 1,
                epoch + 1,
                index + 1,
                len(dataloader),
                loss_avg / (index + 1),
                optimizer.param_groups[0]["lr"],
            ))

        if config.debug is True:
            break
    # print()

    loss_avg /= len(dataloader)
    writer.add_scalars("fold" + str(fold + 1) + "/loss", {"validation": loss_avg}, epoch)
    writer.add_scalars("fold" + str(fold + 1) + "/lr", {"validation": optimizer.param_groups[0]["lr"]}, epoch)

    metrics = calc_metrics(all_probs, all_labels)
    for metric_name in metrics.keys():
        metric_value = metrics[metric_name]
        writer.add_scalars("fold" + str(fold + 1) + "/" + metric_name, {"validation": metric_value}, epoch)

    return loss_avg, metrics


def binarize(probabilities, threshold):
    return probabilities > threshold


def calc_metrics(probs, labels):
    fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
    auc_roc = sklearn.metrics.auc(fpr, tpr)

    precision_pr, recall_pr, _ = sklearn.metrics.precision_recall_curve(labels, probs)
    auc_pr = sklearn.metrics.auc(recall_pr, precision_pr)

    preds = binarize(probs, threshold=0.5)
    accuracy_combined = sklearn.metrics.accuracy_score(labels, preds)
    f1_score = sklearn.metrics.f1_score(labels, preds)
    precision = sklearn.metrics.precision_score(labels, preds)
    recall = sklearn.metrics.recall_score(labels, preds)
    cohen_kappa = sklearn.metrics.cohen_kappa_score(labels, preds)

    accuracy_class = np.sum(labels[labels == 1] * preds[labels == 1]) / np.sum(labels)

    metric_dict = {
        "accuracy_combined": accuracy_combined,
        "accuracy_class": accuracy_class,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
        "cohen_kappa": cohen_kappa,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }

    return metric_dict


def save_checkpoint(save_checkpoint_path, epoch, model, loss_avg_train, loss_avg_val, auc_train, auc_val):
    checkpoint = {
        "run_description": config.run_description,
        "epoch": epoch + 1,
        "date": config.curr_date,
        "description": config.run_description,
        "val_loss": loss_avg_val,
        "train_loss": loss_avg_train,
        "auc_val": auc_val,
        "auc_train": auc_train,
        "model": {
            "include_drop_block": config.include_drop_block,
            "normalize": config.normalize,
            "in_channels": config.in_channels,
            "desired_image_size": config.desired_image_size,
            "name": config.model_name,
            "state_dict": model.state_dict()
        },
    }
    torch.save(checkpoint, save_checkpoint_path)


def set_seed(seed=8, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)
    print("Run description:", config.run_description)

    # Seeds
    if config.deterministic is True:
        set_seed(config.seed, config.deterministic)

    # Init worker
    writer = SummaryWriter(config.writer_dir)

    # Get all paths and labels
    labels_df = pd.read_csv(config.train_csv_path)
    all_npy_paths = labels_df["path"].values
    all_labels = labels_df["target"].values

    # CV
    avg_val_auc = 0
    avg_val_loss = 0
    skf = StratifiedKFold(n_splits=config.cv_splits_amount, random_state=config.seed if config.deterministic is True else None)
    for fold, (train_indices, val_indices) in enumerate(skf.split(all_npy_paths, all_labels)):
        print("Starting fold #" + str(fold + 1))

        # Checkpoints foldwise dir
        checkpoints_dir = os.path.join(config.checkpoints_dir, "fold" + str(fold + 1))
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Data settings
        npy_paths_train = all_npy_paths[train_indices]
        labels_train = all_labels[train_indices]
        dataset_train = dataset.SETIDataset(
            labels_train,
            npy_paths_train,
            in_channels=config.in_channels,
            desired_image_size=config.desired_image_size,
            augment=config.augment,
            normalize=config.normalize
        )
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_loader_workers,
            pin_memory=True
        )

        npy_paths_val = all_npy_paths[val_indices]
        labels_val = all_labels[val_indices]
        dataset_val = dataset.SETIDataset(
            labels_val,
            npy_paths_val,
            in_channels=config.in_channels,
            desired_image_size=config.desired_image_size,
            augment=False,
            normalize=config.normalize
        )
        dataloader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.n_loader_workers,
            pin_memory=True
        )

        print("Images in training: " + str(len(dataset_train)))
        print("Dataloader train len: " + str(len(dataloader_train)))
        print("Images in validating: " + str(len(dataset_val)))
        print("Dataloader val len: " + str(len(dataloader_val)))
        print()

        # Model
        # model = seti_model.DenseNet(in_channels=6, num_classes=1, pretrained=False)
        model = seti_model.SETIModel(
            model_name=config.model_name,
            in_channels=config.in_channels,
            num_classes=1,
            pretrained=config.model_pretrained,
            include_drop_block=config.include_drop_block,
        )
        model = model.to(device=config.device)
        print(config.model_name, "loaded successfully")

        # Criterion
        if config.criterion_name == "BCE":
            perclass_weights_train = torch.tensor(config.pos_weights_train).to(dtype=config.dtype, device=config.device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=perclass_weights_train).to(config.device)
        elif config.criterion_name == "Focal":
            criterion = BinaryFocalWithLogitsLoss(reduction="mean").to(config.device)
        else:
            raise NotImplementedError

        # Optimizer
        if config.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), config.lr, weight_decay=config.weight_decay)
        elif config.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), config.lr, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        # Scheduler
        scheduler = None
        if config.lr_scheduler_name is not None:
            if config.lr_scheduler_name == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=config.lr_reduce_on_plateau_factor,
                    patience=config.lr_reduce_on_plateau_patience,
                    verbose=True
                )
            elif config.lr_scheduler_name == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=config.lr_cosine_annealing_T_max,
                    eta_min=config.lr_cosine_annealing_min_lr,
                    last_epoch=-1
                )
            elif config.lr_scheduler_name == "OneCycleLR":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    epochs=config.lr_cycle_epochs,
                    steps_per_epoch=len(dataloader_train),
                    max_lr=config.lr_cycle_max_lr,
                    pct_start=config.lr_cycle_pct_start,
                    anneal_strategy=config.lr_cycle_anneal_strategy,
                    div_factor=config.lr_cycle_div_factor,
                    final_div_factor=config.lr_cycle_final_div_factor,
                )
            else:
                error_message = "Only reduce step-wise or reduce on plateau is supported but {} was found".format(
                    config.lr_scheduler_name)
                raise NotImplemented(error_message)

        # Validate model before training
        print("Initial validation")
        loss_avg_val, metrics_val = validate(dataloader_val, model, criterion, optimizer, -1, fold, writer)
        auc_val = metrics_val["auc_roc"]
        print("Avg loss:", loss_avg_val)
        print("auc:", auc_val)

        # Train loop
        min_val_loss = 1e10
        min_val_loss_epoch = -1
        max_val_auc = -1
        max_val_auc_epoch = -1
        for epoch in range(config.epochs_amount):
            print("Training started")
            loss_avg_train, metrics_train = train(dataloader_train, model, criterion, optimizer, scheduler, epoch, fold, writer)
            auc_train = metrics_train["auc_roc"]
            print("Avg loss:", loss_avg_train)
            print("auc:", auc_train)

            print("Validation started")
            loss_avg_val, metrics_val = validate(dataloader_val, model, criterion, optimizer, epoch, fold, writer)
            auc_val = metrics_val["auc_roc"]
            print("Avg loss:", loss_avg_val)
            print("auc:", auc_val)

            # Save best checkpoints
            if loss_avg_val < min_val_loss:
                checkpoint_name_old = "best_loss_val_e" + str(min_val_loss_epoch) + "_" + config.run_description + ".pth.tar"
                checkpoint_path_old = os.path.join(checkpoints_dir, checkpoint_name_old)
                if os.path.exists(checkpoint_path_old):
                    os.remove(checkpoint_path_old)

                min_val_loss_epoch = epoch + 1
                min_val_loss = loss_avg_val
                checkpoint_name = "best_loss_val_e" + str(epoch + 1) + "_" + config.run_description + ".pth.tar"
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                save_checkpoint(checkpoint_path, epoch, model, loss_avg_train, loss_avg_val, auc_train, auc_val)
            if auc_val > max_val_auc:
                checkpoint_name_old = "best_auc_val_e" + str(max_val_auc_epoch) + "_" + config.run_description + ".pth.tar"
                checkpoint_path_old = os.path.join(checkpoints_dir, checkpoint_name_old)
                if os.path.exists(checkpoint_path_old):
                    os.remove(checkpoint_path_old)

                max_val_auc_epoch = epoch + 1
                max_val_auc = auc_val
                checkpoint_name = "best_auc_val_e" + str(epoch + 1) + "_" + config.run_description + ".pth.tar"
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                save_checkpoint(checkpoint_path, epoch, model, loss_avg_train, loss_avg_val, auc_train, auc_val)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(min_val_loss)
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
            print()

            if config.debug is True:
                break

        avg_val_auc += max_val_auc
        avg_val_loss += min_val_loss

        if config.debug is True:
            break
        print()

    avg_val_auc /= config.cv_splits_amount
    avg_val_loss /= config.cv_splits_amount
    print("avg_val_auc", avg_val_auc)
    print("avg_val_loss", avg_val_loss)
