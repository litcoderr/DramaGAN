"""
Core Drama Dataset Objects are defined here
"""


import os
import random
import pickle
import pathlib
from tqdm import tqdm
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

from dataset.utils import parse_json
from dataset.preprocess import TextPreprocessor


def custom_collate(batch):
    """
    custom collate function for data_loader.
    Parameters
    ----------
    batch: list
        list of data retrieved from dataset object. length of batch size
    Return
    ------
    batch: tuple
        tuple of two elements(image_batch, text_batch).
        text_batch: list[image_list] -> length: batch_size
            image_list: list[image_vector] -> length: image_seq_length
                image_vector: FloatTensor -> shape: [channel(3), width, height]
        image_batch: list[text_vector] -> length: batch_size
            text_vector: FloatTensor -> shape: [seq, dimension] (description representation vector)
    """
    # list of vector of shape: [text_seq_length, dimension] (description representation vector)
    text_batch = list(map(lambda batch_data: batch_data[0]['description_vector'], batch))

    # list of vector of shape: [1, image_seq_length, channel, width, height]
    image_batch = list(map(lambda batch_data: batch_data[1], batch))

    return text_batch, image_batch


def preprocess_batch(config, text, images):
    """
    Parameters
    ----------
    text: list[text_vector] -> text_vector shape: [text_seq_length, dimension]
    images: list[list[image_vector]] -> image_vector shape: [channel(3), width, height]
    Returns
    -------
    text: FloatTensor shape: [batch_size, image_seq_length, dimension]
    images: FloatTensor shape: [batch_size, image_seq_length, channel(3), width, height]
    """
    text = torch.stack(list(map(lambda text_vector: text_vector.to(config.train_settings.device).mean(0).unsqueeze(0).expand(config.dataset.drama_dataset.n_frame, -1).clone(), text)), dim=0)
    images = torch.stack(list(map(lambda image_list: torch.stack(list(map(lambda image: image.to(config.train_settings.device), image_list)), dim=0), images)), dim=0)
    return text, images


def get_image_dirs(config, datum):
    """
    Get all directory containing images (there can be multiple dirs when data type is 'scene')
    """
    image_root = os.path.join(config.dir, 'AnotherMissOh_images')

    episode, clip, _ = datum['vid'].split('_')
    shots = list(map(lambda shot: str(shot).zfill(4), datum['shot_contained']))
    image_dirs = list(map(lambda shot_str: os.path.join(image_root, episode, clip, shot_str), shots))

    return image_dirs


def filter_data(config, parsed_data):
    """
    Filter invalid data(data with less than config.n_frame number of images)
    """
    filtered_data = []

    for datum in parsed_data:
        # Get Number of Images for this Datum
        n_images = 0
        for image_dir in get_image_dirs(config, datum):
            n_images += len([file_name for file_name in os.listdir(image_dir) if file_name.endswith('.jpg')])

        # If number of images are not enough, filter them out
        if n_images >= config.n_frame:
            filtered_data.append(datum)

    return filtered_data


def init_loader(config, mode, parsed_data):
    """
    Initialize Text Loader and Image Loader. Returns them.
    Parameters
    ----------
    config: object
        Dataset config object
    mode: str
        Dataset mode. Choose from ["train", "test", "val"]
    parsed_data: list<dict>
        Parsed text data parsed from json file
    Return
    ------
    text_loader: data.Dataset
        Text loader object. Returns embedded text by index
    image_loader: data.Dataset
        Image loader object. Returns extracted image feature by index
    """
    # Filter invalid data by config.n_frame and config.frame_selection_method
    filtered_data = filter_data(config, parsed_data)

    # Initialize Text Loader
    text_loader = TextLoader(config, mode=mode, filtered_data=filtered_data)

    # Initialize Image Loader
    image_loader = ImageLoader(config, mode=mode, filtered_data=filtered_data)

    return text_loader, image_loader


class TextLoader(data.Dataset):
    def __init__(self, config, mode, filtered_data):
        self.config = config
        self.mode = mode
        self.filtered_data = filtered_data

        # Initialize Text Preprocessor (for bert embedding)
        self.preprocessor = TextPreprocessor(model_name=self.config.text_embedding_name)
        self.cache_root = os.path.join(self.config.dir, 'Cache', 'text', mode)  # cache root directory

        if self.config.use_cache:
            # Check if cache exists. If not, generate cache and retrieve cache directory
            self.gen_cache(self.filtered_data)
            self.preprocessor.delete()  # Unload model from GPU when finished

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, index):
        data = self.filtered_data[index]

        if self.config.use_cache:
            cache_path = self.get_cache_name(data)
            if not os.path.exists(cache_path):
                self.save_cache(data)

            with open(cache_path, 'rb') as file:
                processed_data = pickle.load(file)
                file.close()
        else:
            processed_data = self.preprocess(data)

        data.update(processed_data)
        return data

    def preprocess(self, datum):
        """
        Preprocess raw text to embedding vector
        Parameters
        ----------
            datum: dict
            Text data dictionary containing QA, Description and more.
        Return
        ------
            preprocessed_datum: dict
            Dictionary containing embedding vector of "description" and QA "answers"
        """
        processed_datum = {
            "description_vector": self.preprocessor.get_embedding(datum['description']).cpu().detach().numpy(),
            "answers_vector": list(map(lambda ans: self.preprocessor.get_embedding(ans).cpu().detach().numpy(), datum['answers']))
        }

        return processed_datum

    def gen_cache(self, filtered_data):
        """
        Generate cache if not exist.
        """
        if not os.path.exists(self.cache_root):  # Check if cache dir exists. If not, preprocess text and store as pickle
            pathlib.Path(self.cache_root).mkdir(parents=True, exist_ok=True)
            for datum in tqdm(filtered_data):
                self.save_cache(datum)

    def save_cache(self, datum):
        pickle_path = self.get_cache_name(datum)
        processed_datum = self.preprocess(datum=datum)

        # Save processed_datum as pickle
        with open(pickle_path, 'wb') as file:
            pickle.dump(processed_datum, file)
            file.close()

    def get_cache_name(self, datum):
        return os.path.join(self.cache_root, '{absolute_id}.pickle'.format(absolute_id=str(datum['absolute_id']).zfill(10)))


class ImageLoader(data.Dataset):
    def __init__(self, config, mode, filtered_data):
        self.config = config
        self.mode = mode
        self.filtered_data = filtered_data

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, index):
        image_dirs = get_image_dirs(self.config, datum=self.filtered_data[index])
        images = []
        for image_dir in image_dirs:
            images += [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir) if image_name.endswith('jpg')]
        images = self.select_images(images, self.config)
        images = list(map(lambda image_name: Image.open(image_name).resize(self.config.img_resize), images))
        return images

    @staticmethod
    def select_images(images, config):
        if config.frame_selection_method == 'random':
            index = [0]
            selected_images = []

            images_len = len(images)
            seq_len = images_len // config.n_frame
            remainder = images_len % config.n_frame

            start_index = 0
            for _ in range(remainder):
                start_index += seq_len + 1
                index.append(start_index)
            for _ in range(config.n_frame-remainder):
                start_index += seq_len
                index.append(start_index)

            for i in range(config.n_frame):
                interval = index[i+1] - index[i]
                temp_index = index[i] + random.randint(0, interval-1)
                selected_images.append(images[temp_index])

        return selected_images


class DramaDataset(data.Dataset):
    """
    Core Drama Dataset
    """
    def __init__(self, config, mode="train"):
        """
        Parameters
        ----------
        config: object
            Global configuration object.
        mode : str
            dataset mode. Choose from ["train", "test", "val"]
        """
        self.train_settings = config.train_settings
        self.config = config.dataset.drama_dataset  # drama dataset configuration info

        # Parse JSON file (containing video info, subtitle, and QA)
        json_file_name = 'AnotherMissOh_QA/AnotherMissOhQA_{mode}_set_subtitle.json'.format(mode=mode)
        json_file_path = os.path.join(self.config.dir, json_file_name)
        json_parsed = parse_json(json_file_path)

        self.text_loader, self.image_loader = init_loader(self.config, mode=mode, parsed_data=json_parsed)

    def __len__(self):
        return len(self.text_loader)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: int
            data index
        Returns
        -------
        text: list
            list of text data(dict)
        images: list
            list of image vector(FloatTensor)
        """
        text = self.text_loader[index]
        images = self.image_loader[index]

        # Text Numpy Vector to Tensor
        text['description_vector'] = torch.FloatTensor(text['description_vector']).squeeze(0)
        text['answers_vector'] = list(map(lambda answer: torch.FloatTensor(answer)
                                          , text['answers_vector']))

        # PIL image to Tensor
        images = list(map(lambda image: transforms.ToTensor()(image)
                          , images))

        return text, images
