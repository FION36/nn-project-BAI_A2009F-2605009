import torch
import torch.nn as nn


class DisasterCNN(nn.Module):

    def __init__(self, num_classes=6, dropout_p=0.3):
        super(DisasterCNN, self).__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # Convolution Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # Convolution Block 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.relu3 = nn.ReLU()


        # Adaptive Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten
        self.flatten = nn.Flatten()


        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)


        # Fully Connected Layer
        self.fc1 = nn.Linear(128, 64)

        self.relu4 = nn.ReLU()


        # Output Layer
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):

        # Convolution Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Convolution Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Convolution Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Adaptive Average Pooling
        x = self.avgpool(x)

        # Flatten
        x = self.flatten(x)

        # Dropout
        x = self.dropout(x)

        # Fully Connected
        x = self.fc1(x)
        x = self.relu4(x)

        # Output
        x = self.fc2(x)

        return x


# Test model creation
if __name__ == "__main__":

    model = DisasterCNN(num_classes=6)

    dummy_input = torch.randn(1, 3, 224, 224)

    output = model(dummy_input)

    print("Output shape:", output.shape)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total trainable parameters:", total_parameters)
