from dropblock import DropBlock2D
import timm
import torch
import torchvision


def extend_module(model, old_module, extension_module):
    for child_name, child in model.named_children():
        if isinstance(child, type(old_module)):
            # print("FOUND", child_name)
            extended_module = torch.nn.Sequential(child, extension_module)
            setattr(model, child_name, extended_module)
        else:
            # print(child_name)
            extend_module(child, old_module, extension_module)


class DenseNet(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=False):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=pretrained)
        self.features = preloaded.features
        self.features.conv0 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.classifier = torch.nn.Linear(1024, num_classes, bias=True)
        del preloaded

    def forward(self, x):
        features = self.features(x)
        out = torch.nn.functional.relu(features, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class SETIModel(torch.nn.Module):
    def __init__(self, model_name, in_channels=3, num_classes=1, pretrained=False, include_drop_block=False):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(self.model_name, pretrained=pretrained, in_chans=in_channels)
        self.dropout = torch.nn.Dropout(0.5)

        self.include_drop_block = include_drop_block
        self.drop_block = DropBlock2D(block_size=5, drop_prob=0.15)

        # dm_nfnet_f#
        if "nfnet" in self.model_name:
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = torch.nn.Identity()
            self.head = torch.nn.Linear(self.n_features, num_classes)
            if self.include_drop_block is True:
                raise NotImplementedError
        elif "efficientnet" in self.model_name:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Identity()
            self.head = torch.nn.Linear(self.n_features, num_classes)
            if self.include_drop_block is True:
                extend_module(self.model.blocks, torch.nn.Sequential(), self.drop_block)
        elif "resnet" in self.model_name:
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Identity()
            self.head = torch.nn.Linear(self.n_features, num_classes)
            if self.include_drop_block is True:
                extend_module(self.model, torch.nn.Sequential(), self.drop_block)
        else:
            raise NotImplementedError

    def forward(self, x):
        feature_vector = self.model(x)
        output = self.head(self.dropout(feature_vector))
        return output


def overfit(model):
    import config
    import numpy as np
    import pandas as pd

    model = model.to(config.device)

    input_tensor = torch.randn((1, 6, 273, 256))
    label = torch.tensor([1.0])
    model.train()

    labels_df = pd.read_csv(config.train_csv_path)
    all_npy_paths = labels_df["path"].values
    all_labels = labels_df["target"].values
    for i in range(len(all_npy_paths)):
        npy_label = all_labels[i]
        if npy_label == 1:
            label = torch.tensor([float(npy_label)])
            npy_path = all_npy_paths[i]
            input_tensor[0] = torch.tensor(np.load(npy_path).astype(np.float32))
            print("Picked", npy_path)
            break

    criterion = torch.nn.BCEWithLogitsLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    input_tensor = input_tensor.to(config.device, config.dtype)
    label = label.to(config.device, config.dtype)

    outputs = model(input_tensor)
    loss = criterion(outputs, label.unsqueeze(1))
    print("Initial loss", loss)

    loss_prev = 1e9
    iteration_amount = 0
    while True:
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = criterion(outputs, label.unsqueeze(1))
        loss.backward()
        optimizer.step()

        print('\r', "Progress {}".format(loss), end='')
        loss_prev = loss
        iteration_amount += 1

        if loss < 1e-4:
            print()
            print("Overfitted in {} iterations!".format(iteration_amount))
            break
    print()


def get_model_parameter_amount(model):
    parameters_amount = 0
    for parameter in model.parameters():
        value = 1
        for num in parameter.shape:
            value *= num

        parameters_amount += value

    return parameters_amount


if __name__ == "__main__":
    import __main__
    print("Run of", __main__.__file__)

    input_tensor = torch.randn((1, 6, 273, 256))
    # model = DenseNet(in_channels=6, num_classes=1, pretrained=False)
    # model = SETIModel(model_name=config.model_name, in_channels=6, num_classes=1, pretrained=True)
    # model = SETIModel(model_name="dm_nfnet_f0", in_channels=6, num_classes=1, pretrained=True)
    model = SETIModel(model_name="tf_efficientnetv2_s_in21k", in_channels=6, num_classes=1, pretrained=True, include_drop_block=True)
    print(model)
    print("Model", model.model_name)
    output = model(input_tensor)
    print("output", output)
    model_parameter_amount = get_model_parameter_amount(model)
    print("parameters_amount:", model_parameter_amount)
    print("size in Gb", model_parameter_amount * 32 / 8 / 1024 / 1024 / 1024)

    # overfit(model)

    # from pprint import pprint
    # model_names = timm.list_models(pretrained=True)
    # pprint(model_names)
    #
    # model = SETIModel("efficientnet_b0")
    #
    # input_tensor = torch.randn((1, 3, 273, 256))
    # output = model(input_tensor)
    # print("output", output.shape)
