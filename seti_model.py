import torch
import torchvision
import timm

import config


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
    def __init__(self, model_name, in_channels=3, num_classes=1, pretrained=False):
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(self.model_name, pretrained=pretrained, in_chans=in_channels)
        self.dropout = torch.nn.Dropout(0.5)
        # dm_nfnet_f#
        if self.model_name.startswith("dm_nfnet"):
            self.n_features = self.model.head.fc.in_features
            self.model.head.fc = torch.nn.Linear(self.n_features, num_classes)
        elif self.model_name.startswith("tf_efficientnetv2"):
            # tf_efficientnetv2_b#
            self.n_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(self.n_features, num_classes)
        elif self.model_name.startswith("efficientnet"):
            # tf_efficientnetv2_b#
            self.n_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(self.n_features, num_classes)
        elif "resnet" in self.model_name:
            self.n_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(self.n_features, num_classes)

        print(self.model)

    def forward(self, x):
        output = self.model(x)
        return output


def overfit(model):
    import config
    import numpy as np
    import pandas as pd

    model = model.to(config.device)

    input_tensor = torch.randn((1, 6, 273, 256))
    label = torch.tensor([1.0])
    model.train()

    labels_df = pd.read_csv(config.labels_csv_path)
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
    # model = SETIModel(model_name="resnet18d", in_channels=6, num_classes=1, pretrained=True)
    model = SETIModel(model_name="tf_efficientnet_b1_ns", in_channels=6, num_classes=1, pretrained=True)
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
