
import os
import torch.utils.data as data

from dataset.utils import parse_json


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

        # When using preprocessed text, preprocess and store cache if needed
        self.cache = None
        if self.config.use_cache:
            #TODO make preprocessing
            pass
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
