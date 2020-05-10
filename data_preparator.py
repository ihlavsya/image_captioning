"""module for preparing dataset"""
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import h5py
from matplotlib import pyplot as plt
from skimage import transform, io
from pycocotools.coco import COCO
from time import process_time
from mpi4py import MPI

from storage import Storage
from helper import (get_imgs_dir_filename,
                    get_img_filename,
                    get_annotations_json_filename,
                    get_captions_json_filename,
                    get_hdf5_filename)


def get_edited_json(json_filename, imgs_dir):
    """create new file with annotations from json_file
    that do have corresponding images"""
    edited_anns = []
    edited_images = []
    coco = COCO(json_filename)
    for ann_id in coco.getAnnIds():
        img_id = coco.anns[ann_id]["image_id"]
        img_filename = coco.imgs[img_id]["file_name"]
        img_path = os.path.join(imgs_dir, img_filename)
        if os.path.exists(img_path):
            edited_anns.append(coco.anns[ann_id])
            edited_images.append(coco.imgs[img_id])

    samples = {"imgs": edited_images, "anns": edited_anns}
    edited_json = get_json_data(samples)
    return edited_json


def save_json(json_data, json_filename):
    """saving"""
    with open(json_filename, "w") as outfile:
        json.dump(json_data, outfile)

    print("{0} is written".format(json_filename))


def create_test_data(data_dir, data_types):
    """performs moving samples from several datasets to test dataset"""
    destination_data_type = "test2017"
    destination_dir = get_imgs_dir_filename(data_dir, destination_data_type)
    # create test img dir if does not exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    destination_json_filename = get_annotations_json_filename(
        data_dir, destination_data_type, prefix="cleaned")

    destination_samples = {"imgs": [], "anns": []}
    for data_type in data_types.values():
        json_filename = get_annotations_json_filename(
            data_dir, data_type, prefix="cleaned")
        splitted_samples = get_splitted_samples(json_filename)
        destination_samples["imgs"].extend(
            splitted_samples["destination"]["imgs"])
        destination_samples["anns"].extend(
            splitted_samples["destination"]["anns"])

        json_data = get_json_data(splitted_samples["origin"])
        save_json(json_data, json_filename)

        origin_dir = get_imgs_dir_filename(data_dir, data_type)
        move_images(origin_dir, destination_dir,
                    splitted_samples["destination"]["imgs"])

    json_data = get_json_data(destination_samples)
    save_json(json_data, destination_json_filename)


def get_json_data(samples):
    """function that forms json data"""
    json_data = Storage.BASE_JSON
    json_data["images"] = samples["imgs"]
    json_data["annotations"] = samples["anns"]
    return json_data


def get_splitted_samples(origin_json_filename, factor=0.2):
    """split samples among two datasets"""
    coco = COCO(origin_json_filename)
    origin_img_ids = coco.getImgIds()
    destination_img_mask = np.random.choice(
        [False, True], size=len(origin_img_ids), p=[1-factor, factor])

    new_origin_img_ids = []
    destination_img_ids = []
    for i, value in enumerate(destination_img_mask):
        if value:
            destination_img_ids.append(origin_img_ids[i])
        else:
            new_origin_img_ids.append(origin_img_ids[i])

    origin_samples = get_samples(coco, new_origin_img_ids)
    destination_samples = get_samples(coco, destination_img_ids)
    return {"origin": origin_samples, "destination": destination_samples}


def move_images(origin_dir, destination_dir, imgs):
    """move images from origin dir to destination dir"""
    for img in imgs:
        img_filename = img["file_name"]
        origin_img_path = os.path.join(origin_dir, img_filename)
        destination_img_path = os.path.join(destination_dir, img_filename)
        os.rename(origin_img_path, destination_img_path)

    print("imgs are moved from {0} to {1}".format(origin_dir, destination_dir))


def get_samples(dataset, img_ids):
    """get samples (images and its corresponding annotations)"""
    imgs = []
    anns = []
    for img_id in img_ids:
        img = dataset.imgs[img_id]
        imgs.append(img)

        img_anns = dataset.imgToAnns[img_id]
        anns.extend(img_anns)

    samples = {"imgs": imgs, "anns": anns}
    return samples


def create_cleaned_annotations(data_dir, data_types):
    """perform cleaning and saving of annotations"""
    for data_type in data_types.values():
        old_json_filename = get_annotations_json_filename(data_dir, data_type)
        imgs_dir = get_imgs_dir_filename(data_dir, data_type)
        edited_json = get_edited_json(old_json_filename, imgs_dir)
        new_json_filename = get_annotations_json_filename(
            data_dir, data_type, "cleaned")
        with open(new_json_filename, "w") as file:
            json.dump(edited_json, file)


def get_list_trimmed_annotations(json_filename, captions_per_image):
    """get only some number of captions per image.
    every image has to have equal numbers of captions"""
    coco = COCO(json_filename)
    img_ids = coco.getImgIds()
    anns = []
    imgs = []
    for img_id in img_ids:
        img_anns = coco.imgToAnns[img_id]
        anns.extend(img_anns[:captions_per_image])
        imgs.append(coco.imgs[img_id])

    samples = {"anns": anns, "imgs": imgs}
    return samples


def create_trimmed_annotations(data_dir, data_types, captions_per_image):
    """limit captions per image and remove images from json filename"""
    for data_type in data_types.values():
        json_filename = get_annotations_json_filename(
            data_dir, data_type, prefix="cleaned")
        samples = get_list_trimmed_annotations(
            json_filename, captions_per_image)
        json_data = get_json_data(samples)
        save_json(json_data, json_filename)


def create_only_captions_json(data_dir, data_types):
    """create json that only contains info about captions"""
    for data_type in data_types.values():
        ann_json_filename = get_annotations_json_filename(
            data_dir, data_type, prefix="cleaned")
        coco = COCO(ann_json_filename)
        json_data = {}
        json_data["anns"] = list(coco.anns.values())
        caption_json_filename = get_captions_json_filename(
            data_dir, data_type, prefix="cleaned")
        save_json(json_data, caption_json_filename)


def get_transformed_img(impath):
    """make transformation of image before storing it into file"""
    img = io.imread(impath)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = transform.resize(img, (256, 256))
    # Convert the image to a 0-255 scale.
    img = 255 * img
    # Convert to integer data type pixels.
    img = img.astype(np.uint8)
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255
    return img


def get_transformed_imgs(img_paths):
    images = []
    for impath in img_paths:
        img = get_transformed_img(impath)
        images.append(img)
    print("chunk prepared")
    return images


def create_image_hdf5_datasets_in_parallel_mpi(data_dir, data_types,
                                               captions_per_image):
    """create hdf5 dataset for image directory"""
    comm = MPI.COMM_WORLD
    num_processes = comm.size
    # The process ID (integer 0-3 for 4-process run)
    rank = comm.rank
    for data_type in data_types.values():
        json_filename = get_annotations_json_filename(data_dir,
                                                      data_type,
                                                      prefix="cleaned")
        coco = COCO(json_filename)
        img_ids = coco.getImgIds()

        hdf5_filename = get_hdf5_filename(data_dir, data_type)

        with h5py.File(hdf5_filename,
                       "a",
                       driver='mpio',
                       comm=comm) as hdf5_file:
            # Make a note of the number of captions we are sampling per image
            hdf5_file.attrs["captions_per_image"] = captions_per_image

            # Create dataset inside HDF5 file to store images
            dataset_shape = (len(img_ids), 3, 256, 256)
            images = hdf5_file.create_dataset("images",
                                              shape=dataset_shape,
                                              dtype="uint8")

            for i, img_id in enumerate(img_ids):
                if i % num_processes == rank:
                    img_filename = coco.imgs[img_id]["file_name"]
                    impath = get_img_filename(data_dir,
                                              data_type,
                                              img_filename)
                    img = get_transformed_img(impath)
                    # Save image to HDF5 file
                    images[i] = img
                    if i % 1000 == 0:
                        print("{}-th iteration is written".format(i))

        if rank == 0:
            print("{} is written".format(hdf5_filename))


def create_image_hdf5_datasets(data_dir, data_types,
                               captions_per_image):
    """create hdf5 dataset for image directory"""
    for data_type in data_types.values():
        json_filename = get_annotations_json_filename(
            data_dir, data_type, prefix="cleaned")
        coco = COCO(json_filename)
        img_ids = coco.getImgIds()
        hdf5_filename = get_hdf5_filename(data_dir, data_type)

        with h5py.File(hdf5_filename, "a") as hdf5_file:
            # Make a note of the number of captions we are sampling per image
            hdf5_file.attrs["captions_per_image"] = captions_per_image

            # Create dataset inside HDF5 file to store images
            dataset_shape = (len(img_ids), 3, 256, 256)
            images = hdf5_file.create_dataset("images",
                                              shape=dataset_shape,
                                              dtype="uint8",
                                              compression="gzip",
                                              compression_opts=9)

            for i, img_id in enumerate(img_ids):
                img_filename = coco.imgs[img_id]["file_name"]
                impath = get_img_filename(data_dir, data_type, img_filename)
                img = get_transformed_img(impath)
                # Save image to HDF5 file
                images[i] = img
                if i % 1000 == 0:
                    print("{}-th iteration is written".format(i))

        print("{} is written".format(hdf5_filename))


def get_image_paths(data_dir, data_type):
    """extract paths for images in right order"""
    json_filename = get_annotations_json_filename(
        data_dir, data_type, prefix="cleaned")
    coco = COCO(json_filename)
    img_ids = coco.getImgIds()
    img_paths = []
    for img_id in img_ids:
        img_filename = coco.imgs[img_id]["file_name"]
        impath = get_img_filename(data_dir, data_type, img_filename)
        img_paths.append(impath)
    return img_paths


def create_image_hdf5_datasets_in_parallel(data_dir, data_types,
                                           captions_per_image):
    """create hdf5 dataset for image directory"""
    for data_type in data_types.values():
        img_paths = get_image_paths(data_dir, data_type)
        print(len(img_paths))

        hdf5_filename = get_hdf5_filename(data_dir, data_type)
        dataset_shape = (len(img_paths), 3, 256, 256)

        with h5py.File(hdf5_filename, "a") as hdf5_file:
            # Make a note of the number of captions we are sampling per image
            hdf5_file.attrs["captions_per_image"] = captions_per_image

            # Create dataset inside HDF5 file to store images
            dataset_images = hdf5_file.create_dataset("images",
                                                      shape=dataset_shape,
                                                      dtype="uint8",
                                                      compression="gzip",
                                                      compression_opts=4)

            # and what if they could share same queue
            processes = mp.cpu_count()
            chunk_size = 128
            chunked_img_paths = []
            for start in range(0, len(img_paths), chunk_size):
                end = min(start + chunk_size, len(img_paths))
                chunk_of_image_paths = [img_paths[idx]
                                        for idx in range(start, end)]
                chunked_img_paths.append(chunk_of_image_paths)
            len_of_chunks = len(chunked_img_paths)

            with mp.Pool() as pool:
                chunked_images = pool.imap(
                    get_transformed_imgs, chunked_img_paths, processes)
                for i, chunk_of_images in enumerate(chunked_images):
                    start = i * chunk_size
                    end = start + len(chunk_of_images)
                    dataset_images[start:end] = chunk_of_images
                    print("{} done out of {}".format(i, len_of_chunks))

        print("{} is written".format(hdf5_filename))


def check_dataset(data_dir, data_type, captions_per_image):
    """check dataset for correctness"""
    hdf5_filename = get_hdf5_filename(data_dir, data_type)
    json_filename = get_annotations_json_filename(data_dir, data_type,
                                                  prefix="cleaned")
    coco = COCO(json_filename)
    img_ids = coco.getImgIds()
    print(len(img_ids))
    captions_json_filename = get_captions_json_filename(data_dir,
                                                        data_type,
                                                        prefix="cleaned")

    with open(captions_json_filename) as cap_file:
        captions = json.load(cap_file)["anns"]

    random_indices = np.random.choice(range(len(captions)), 5)
    rand_caps = [captions[idx] for idx in random_indices]
    random_img_indices = [
        idx // captions_per_image for idx in random_indices]
    with h5py.File(hdf5_filename, "r") as hdf5_file:
        images = hdf5_file["images"]
        print(len(images))

        rand_images = [images[idx] for idx in random_img_indices]

    for i, img in enumerate(rand_images):
        img = img.transpose((1, 2, 0))
        imgplot = plt.imshow(img)
        plt.title(rand_caps[i]["caption"])
        plt.show()


def main():
    """perform editing of annotaions for all parts of dataset
    create test dataset"""
    data_dir = "dataset"
    data_types = {"train": "train2017"}
    # create_cleaned_annotations(data_dir, data_types)
    # after this function captions from one image grouped together
    # create_test_data(data_dir, data_types)
    # after this function captions from one image grouped together
    # create_trimmed_annotations(data_dir, data_types, 3)
    # create_only_captions_json(data_dir, data_types)
    t = process_time()
    create_image_hdf5_datasets_in_parallel(data_dir, data_types,
                                           Storage.CAPTIONS_PER_IMAGE)
    elapsed_time = process_time() - t
    print(elapsed_time)
    # check_dataset(data_dir, data_types["test"], Storage.CAPTIONS_PER_IMAGE)
    # hdf5_filename = get_hdf5_filename(data_dir, data_types["train"])
    # with h5py.File(hdf5_filename, "r") as hdf5_file:
    #     imgs = hdf5_file["images"]
    #     imgs = imgs[:10]

    # for i, img in enumerate(imgs):
    #     img = img.transpose((1, 2, 0))
    #     imgplot = plt.imshow(img)
    #     plt.show()


if __name__ == "__main__":
    main()
