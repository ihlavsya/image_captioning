"""class that performs model training with checkpoints saving"""
import os
import time
import copy

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelTrainer():
    """class that performs model training with checkpoints saving"""

    def __init__(self, encoder, decoder, optimizer, scheduler,
                 loss_function, dataloaders, device,
                 path=None):
        """Set parameters for training"""
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.encoder = encoder
        self.decoder = decoder
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = {"train": len(dataloaders["train"].dataset),
                              "val": len(dataloaders["val"].dataset)}
        self.device = device
        self.path = path

    def save_checkpoint(self, loss, model, checkpoint_path):
        """save model per checkpoint"""
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "loss": loss.item(),
        }
        torch.save(checkpoint_dict, checkpoint_path)

    def train(self, epochs=10):
        """perform update weights for two models, track statistics, save progress"""
        since = time.time()
        best_encoder_wts = copy.deepcopy(self.encoder.state_dict())
        best_decoder_wts = copy.deepcopy(self.decoder.state_dict())
        best_acc = 0.0
        losses = {"val": [], "train": []}
        accuracies = {"val": [], "train": []}
        # you can not to return models
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs - 1))
            print("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.encoder.train()  # Set self.encoder to training mode
                    self.decoder.train()  # Set self.decoder to training mode
                else:
                    self.encoder.eval()   # Set self.encoder to evaluate mode
                    self.decoder.eval()   # Set self.decoder to evaluate mode

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
                    self.decoder.zero_grad()
                    self.encoder.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # Forward, backward and optimize
                        features = self.encoder(images)
                        outputs = self.decoder(features, captions, lengths)
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

                losses[phase].append(epoch_loss)
                accuracies[phase].append(epoch_acc)

                print("{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_encoder_wts = copy.deepcopy(self.encoder.state_dict())
                    best_decoder_wts = copy.deepcopy(self.decoder.state_dict())
                    encoder_checkpoint_path = os.path.join(
                        self.path, "encoder", "%d.pt" % epoch)
                    self.save_checkpoint(
                        loss, self.encoder, encoder_checkpoint_path)
                    decoder_checkpoint_path = os.path.join(
                        self.path, "decoder", "%d.pt" % epoch)
                    self.save_checkpoint(
                        loss, self.decoder, decoder_checkpoint_path)

            print()

        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(best_acc))

        # load best self.encoder weights
        self.encoder.load_state_dict(best_encoder_wts)
        self.decoder.load_state_dict(best_decoder_wts)
        return self.encoder, self.decoder, losses, accuracies
