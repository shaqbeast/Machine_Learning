import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        """
        Instantiates the CNN model

        HINT: Here's an outline of the function you can use. Fill in the "..." with the appropriate code:
        """
        super(CNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope = 0.01),
            nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.02, inplace=False),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.02, inplace=False)
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # Linear layers
            nn.Linear(in_features=3136, out_features=256, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.02, inplace=False),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.02, inplace=False),
            nn.Linear(in_features=128, out_features=4, bias=True)
        )


    def forward(self, x):
        """
        Runs the forward method for the CNN model

        Args:
            x (torch.Tensor): input tensor to the model

        Returns:
            torch.Tensor: output classification tensor of the model
        """
        x = self.feature_extractor(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        
        return x
        