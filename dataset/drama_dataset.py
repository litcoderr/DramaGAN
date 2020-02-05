"""
Core Drama Dataset Objects are defined here
"""


import os
import pickle
import pathlib
from tqdm import tqdm
import torch.utils.data as data

from dataset.utils import parse_json
from dataset.preprocess import TextPreprocessor


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

        if self.config.use_cache:
            # Check if cache exists. If not, generate cache and retrieve directory root
            self.text_cache_root = self.gen_cache(self.config, self.mode, self.filtered_data)

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

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
            "description": self.preprocessor.get_embedding(datum['description']).cpu().detach().numpy(),
            "answers": list(map(lambda ans: self.preprocessor.get_embedding(ans).cpu().detach().numpy(), datum['answers']))
        }

        return processed_datum

    def gen_cache(self, config, mode, filtered_data):
        """
        Generate cache if not exist.
        """
        cache_root = os.path.join(config.dir, 'Cache', 'text', mode)  # cache root directory
        if not os.path.exists(cache_root):  # Check if cache dir exists. If not, preprocess text and store as pickle
            pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
            for datum in tqdm(filtered_data):
                pickle_path = os.path.join(cache_root, '{absolute_id}.pickle'.format(absolute_id=str(datum['absolute_id'])))
                processed_datum = self.preprocess(datum=datum)

                # Save processed_datum as pickle
                with open(pickle_path, 'wb') as file:
                    pickle.dump(processed_datum, file)

        return cache_root


class ImageLoader(data.Dataset):
    def __init__(self, config, mode, filtered_data):
        self.config = config
        self.mode = mode
        self.filtered_data = filtered_data

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


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
        self.config = config.dataset.drama_dataset  # drama dataset configuration info

        # Parse JSON file (containing video info, subtitle, and QA)
        json_file_name = 'AnotherMissOh_QA/AnotherMissOhQA_{mode}_set_subtitle.json'.format(mode=mode)
        json_file_path = os.path.join(self.config.dir, json_file_name)
        json_parsed = parse_json(json_file_path)

        self.text_loader, self.image_loader = init_loader(self.config, mode=mode, parsed_data=json_parsed)
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
