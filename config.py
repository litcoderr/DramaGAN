import os
from munch import DefaultMunch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

#  Store every settings here
config = {
    "dataset": {
        "drama_dataset": {
            "dir": os.path.join(PROJECT_ROOT, 'dataset/Drama_Dataset'),
            "n_frame": 5,  # number of frames retrieved for each data
            "frame_selection_method": "random",  # frame selection method [choose from: 'random']
            "use_cache": True,  # If True, preprocess text and store as cache, else, process text during runtime
            "text_embedding": "bert_pretrained"  # text embedder [choose from: 'bert_pretrained']
        }
    }
}

config = DefaultMunch.fromDict(config, None)  # for attribute style access of dictionary type  ex) config.dataset ..etc
