"""class that performs model training with checkpoints saving"""
import time

import torch

import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence

from helper import save_checkpoint, get_checkpoint_path
from encoder_cnn import EncoderCNN
from decoder_rnn import DecoderRNN
from image_captioner import ImageCaptioner


class ModelTrainer():
    """class that performs model training with checkpoints saving"""
    # pass hyperparams-dict and not optimizer, scheduler, model,
    # init everything here

    def __init__(self, dataloaders, hyperparams_dict, wv_wrapper, path=None):
        """Set parameters for training"""
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataloaders = dataloaders
        self.dataset_sizes = {"train": len(dataloaders["train"].dataset),
                              "val": len(dataloaders["val"].dataset)}
        self.hyperparams_dict = hyperparams_dict

        self.model = self.__get_init_model(wv_wrapper)
        self.optimizer, self.scheduler = self.__get_init_optimization_means()

        self.loss_function = nn.CrossEntropyLoss()
        self.path = path
        self.best_acc = 0.0
        self.losses = {"train": [], "val": []}
        self.accuracies = {"train": [], "val": []}

    def __get_init_model(self, wv_wrapper):
        """init image_captioner
        that consists of decoder and encoder"""
        embed_size = self.hyperparams_dict["embed_size"]

        encoder = EncoderCNN(embed_size).to(self.device)
        decoder = DecoderRNN(wv_wrapper, embed_size).to(self.device)
        model = ImageCaptioner(encoder, decoder)
        return model

    def __get_init_optimization_means(self):
        """init optimizer and corresponding
        scheduler"""
        learning_rate = self.hyperparams_dict["learning_rate"]
        step_size = self.hyperparams_dict["step_size"]
        gamma = self.hyperparams_dict["gamma"]
        params = self.model.parameters()

        optimizer = optim.Adam(params, lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
        return optimizer, scheduler

    def __train_one_epoch(self, epoch):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                self.model.train()  # Set self.model to training mode
            else:
                self.model.eval()   # Set self.model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for images, captions, lengths in self.dataloaders[phase]:
                # Set mini-batch dataset
                images = images.to(self.device)
                captions = captions.to(self.device)
                targets = pack_padded_sequence(
                    captions, lengths, batch_first=True)[0]

                # zero the parameter gradients
                self.model.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # Forward, backward and optimize
                    outputs = self.model(images, captions, lengths)
                    preds = torch.argmax(outputs, 1)
                    loss = self.loss_function(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        self.optimizer.step()

                # statistics
                running_loss += loss.item() * captions.size(0)
                running_corrects += torch.sum(preds == targets)

            if phase == "train":
                self.scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes[phase]
            epoch_acc = running_corrects / self.dataset_sizes[phase]

            self.losses[phase].append(epoch_loss)
            self.accuracies[phase].append(epoch_acc)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.__create_checkpoint(loss.item(), epoch)

    def __create_checkpoint(self, loss_item, epoch):
        """create checkpoints for decoder
        and encoder parts"""
        checkpoint_path = get_checkpoint_path(self.path, "encoder", epoch)
        save_checkpoint(
            loss_item, self.model.get_encoder_state_dict(), checkpoint_path)
        checkpoint_path = get_checkpoint_path(self.path, "decoder", epoch)
        save_checkpoint(
            loss_item, self.model.get_decoder_state_dict(), checkpoint_path)

    def train(self, epochs=10):
        """perform update weights for two models,
        track statistics, save progress"""
        since = time.time()

        # you can not to return models
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 10)

            self.__train_one_epoch(epoch)
            print()

        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(self.best_acc))

        return self.model, self.losses, self.accuracies
