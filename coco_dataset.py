"""COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
import os
import copy
import torch

import torch.utils.data as data_utils
import nltk
from PIL import Image
from pycocotools.coco import COCO


class CocoDataset(data_utils.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, wv, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = self.coco.getAnnIds()
        self.wv = copy.deepcopy(wv)
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # after simple try to add new tokens to vocabulary
        vocab = self.wv.vocab
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]["caption"]
        img_id = self.coco.anns[ann_id]["image_id"]
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        # image = Image.open(os.path.join(self.root, path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab["<start>"].index)
        for token in tokens:
            if token not in vocab:
                token = "<unk>"
            caption.append(vocab[token].index)
        caption.append(vocab["<end>"].index)
        target = torch.tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)  # i don`t quite get it

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths), dtype=torch.long)

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end].clone().detach()

    return images, targets, lengths
