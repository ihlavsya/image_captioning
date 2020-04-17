"""EncoderCNN is a feature extractor that uses all weights of resnext50_32x4d
except for class classification layer"""
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """EncoderCNN is a feature extractor that uses all weights of resnext50_32x4d
    except for class classification layer"""
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnext50_32x4d(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, embed_size)
        # self.resnet = self.resnet.to(device)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.resnet.fc.weight.data.normal_(0.0, 0.02)
        self.resnet.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = self.bn(features)
        return features
