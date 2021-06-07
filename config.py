import datetime
import os
import torch

curr_date = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
# model_name = "dm_nfnet_f0"  # 32 128
model_name = "tf_efficientnetv2_s_in21k"  # 32
# model_name = "efficientnet_b4"  # 16
# model_name = "efficientnet_b0"  # 64 256
model_pretrained = True

# Dataset settings
in_channels = 1  # {1, 2, 6}
desired_image_size = 256  # 273 for 6 channels, anything for 1 channel
augment = True
normalize = True

# Training process
debug = False
deterministic = True
seed = 8
print_freq = 300
cv_splits_amount = 5
epochs_amount = 10
batch_size = 32
test_batch_size = 128
n_loader_workers = 4
# pos_weights_train = [9.687047294418406]
pos_weights_train = [1.0]
mix_strategy = "MixUp"  # {None, "FMix", "MixUp"}
mixup_alpha = 1.0
criterion_name = "BCE"  # {"BCE", "Focal"}

# lr
lr_scheduler_name = "CosineAnnealingLR"  # ["ReduceLROnPlateau" | "CosineAnnealingLR"]
weight_decay = 1e-6
lr = 5e-4
lr_reduce_on_plateau_factor = 0.5
lr_reduce_on_plateau_patience = 3
lr_cosine_annealing_min_lr = 5e-6
lr_cosine_annealing_T_max = epochs_amount

run_description = "{}_model={}_pretrained={}_c={}_size={}_aug={}_nrmlz={}_lr={}_bs={}_weights={}_loss={}_scheduler={}_{}".format(
    curr_date,
    model_name,
    str(model_pretrained)[0],
    str(in_channels),
    str(desired_image_size),
    str(augment)[0],
    "meanstd" if normalize is True else str(normalize)[0],
    str(lr),
    str(batch_size),
    str(pos_weights_train),
    criterion_name,
    lr_scheduler_name,
    mix_strategy + str(mixup_alpha) + "_SpecAugWZeros",
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

# Directories
data_dir = os.path.join(".", "data")
train_data_dir = os.path.join(data_dir, "train")
test_data_dir = os.path.join(data_dir, "test")
train_csv_path = os.path.join(data_dir, "train.csv")
test_csv_path = os.path.join(data_dir, "test.csv")

checkpoints_dir = os.path.join(".", "checkpoints", run_description)
writer_dir = os.path.join("runs", run_description)
submissions_dir = os.path.join(".", "submissions")
