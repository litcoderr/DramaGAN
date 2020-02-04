
import os
import torch.utils.data as data

from dataset.utils import parse_json


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
    # TODO Filter Available Data by config.n_frame and config.frame_selection_method
    filtered_data = []

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

        if self.config.use_cache:
            self.qa_cache_root = self.load_cache(self.filtered_data)  # check cache exists and retrieve qa root

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def load_cache(self, filtered_data):
        cache_root = os.path.join(self.config.dir, 'cache/')  # cache root directory
        qa_cache_root = os.path.join(cache_root, 'QA')  # QA text cache directory
        # TODO Check if cache exists. If not, preprocess data.
        return qa_cache_root


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
