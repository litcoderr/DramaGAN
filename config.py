"""
Settings are defined here
"""

import os
from munch import DefaultMunch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

#  Store every settings here
config = {
    "train_settings": {  # Set Training settings here
        "dataset": "drama_dataset",  # dataset name (refer below 'dataset' section)
        "model": "storygan",  # model name (refer below 'model' section)
        "mode": "train",  # Choose from ['train', 'test', 'val']
        "batch_size": 10,  # batch size
        "num_workers": 5,  # number of workers
        "n_epoch": 100,  # number of epochs
        "device": 'cuda'  # Choose from ['cuda', 'cpu']
    },
    "dataset": {
        "drama_dataset": {
            "dir": os.path.join(PROJECT_ROOT, 'dataset/Drama_Dataset'),
            "n_frame": 5,  # number of frames retrieved for each data
            "frame_selection_method": "random",  # frame selection method [choose from: 'random']
            "img_resize": (256, 256),  # image resize

            # Caching Options (Text Embeddings and Image Feature Extraction)
            "use_cache": True,  # If True, preprocess text and store as cache, else, process text during runtime
            "text_embedding_name": "bert_pretrained",  # text embedder [choose from: 'bert_pretrained']
            "image_feature_extraction": "None"  # image feature extraction method [choose from: 'None']
        }
    },
    "model": {
        "storygan": {
            "img_size": 256,
            "text_dim": 768,
            "text_hidden_dim": 75,
            "noise_dim": 75,
            "gen_channel": 192 * 8,
            "latent_img_dim": 15
        }
    }
}

config = DefaultMunch.fromDict(config, None)  # for attribute style access of dictionary type  ex) config.dataset ..etc
