"""module for running complete process of loading data and training models"""
import time
import os
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence

from torchvision import datasets, models, transforms
from gensim.models import KeyedVectors

from vocabulary_utils import build_word2vec
from coco_dataset import CocoDataset, collate_fn
from encoder_cnn import EncoderCNN
from decoder_rnn import DecoderRNN
from model_trainer import ModelTrainer
import settings

plt.ion()

DATA_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(50) 


def main():
    """function for experimenting and running complete pipeline"""
    model, num_layers = build_word2vec("dataset1/annotations/train.json")
    wv = model.wv
    num_layers = 20
    images_dir = "dataset1/images/val"
    ann_filename = "dataset1/annotations/val.json"
    dataset_val = CocoDataset(images_dir, ann_filename,
                              wv, transform=DATA_TRANSFORMS)
    images_dir = "dataset1/images/train"
    ann_filename = "dataset1/annotations/train.json"
    dataset_train = CocoDataset(
        images_dir, ann_filename, wv, transform=DATA_TRANSFORMS)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=2,
                                                  shuffle=True, collate_fn=collate_fn)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2,
                                                    shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders = {"train": data_loader_train, "val": data_loader_val}
    num_embeddings = len(wv.vectors)
    embed_size = settings.EMBED_SIZE
    hidden_size = settings.EMBED_SIZE

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, wv.vectors, hidden_size,
                         num_embeddings, num_layers).to(device)
    learning_rate = 0.001
    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)
    # for now lets try default step_size and gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2)
    i = iter(dataloaders["train"])
    b = iter(data_loader_train)
    path = "models_weights"  # you have to use custom dataset and custom collate-fn
    trainer = ModelTrainer(encoder, decoder, optimizer,
                           scheduler, loss_function, dataloaders, device, path)
    encoder, decoder, losses, accuracies = trainer.train()


if __name__ == "__main__":
    main()
