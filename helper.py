"""module with helper functions that are used in many places in project"""
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from coco_dataset import CocoDataset, collate_fn
from storage import Storage


def get_imgs_dir_filename(data_dir, data_type):
    """get name of images directory"""
    imgs_dir = "{0}/images/{1}".format(data_dir, data_type)
    return imgs_dir


def get_annotations_json_filename(data_dir, data_type, prefix=""):
    """get name of file for annotations to images"""
    captions_filename = "{0}/annotations/{1}captions_{2}.json".format(
        data_dir, prefix, data_type)
    return captions_filename


def get_captions_json_filename(data_dir, data_type, prefix=""):
    """get name of file for annotations to images"""
    captions_filename = "{0}/captions/{1}captions_{2}.json".format(
        data_dir, prefix, data_type)
    return captions_filename


def get_img_filename(data_dir, data_type, base_filename):
    """get name of file for annotations to images"""
    img_filename = os.path.join(data_dir, "images", data_type, base_filename)
    return img_filename


def get_hdf5_filename(data_dir, data_type):
    """get name of file for hdf5 files that store images"""
    base_filename = "{}_images.hdf5".format(data_type)
    filename = os.path.join(data_dir, base_filename)
    return filename


def plot_results(results, title):
    """display results"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["red", "blue"]
    for i, label in enumerate(results.keys()):
        ax.plot(results[label], label=label, color=colors[i])

    plt.legend(loc=2)
    plt.title(title)
    plt.show()
    plt.pause(50)


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


def save_checkpoint(loss_item, model_state_dict, checkpoint_path):
    """save model per checkpoint"""
    checkpoint_dict = {
        "model_state_dict": model_state_dict,
        "loss": loss_item,
    }
    torch.save(checkpoint_dict, checkpoint_path)


def get_checkpoint_path(path, prefix, epoch):
    """get path for checkpoint"""
    checkpoint_filename = "{}.pt".format(epoch)
    checkpoint_path = os.path.join(path, prefix, checkpoint_filename)
    return checkpoint_path


def get_dataloader(data_dir, data_type, wv):
    """get dataloader for specified data_loader/split"""
    images_dir = get_imgs_dir_filename(data_dir, data_type)
    ann_filename = get_annotations_json_filename(data_dir, data_type)
    dataset = CocoDataset(images_dir, ann_filename, wv,
                          transform=Storage.DATA_TRANSFORMS)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=8,
                                              shuffle=True,
                                              collate_fn=collate_fn,
                                              num_workers=2)

    return data_loader


def get_dataloaders(data_dir, data_types, wv):
    """get dataloaders for specified splits/data_types"""
    dataloaders = {}
    for data_type_key, data_type_value in data_types.items():
        dataloader = get_dataloader(data_dir, data_type_value, wv)
        dataloaders[data_type_key] = dataloader

    return dataloaders


def display_formatted_results(results):
    """display results in convenient format"""
    for key in sorted(results):
        rate_r, gamma_r = key
        val_accuracy, train_loss, lr, gamma, stats = results[(rate_r, gamma_r)]
        print("rate_r:{}, gamma_r:{}: val_accuracy:{}, train_loss:{}, lr:{}, gamma:{}".format(
            rate_r, gamma_r, val_accuracy, train_loss, lr, gamma
        ))

        plot_results(stats["losses"], "losses")
        plot_results(stats["accuracies"], "accuracies")
