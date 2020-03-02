"""
Settings are defined here
"""

import os
from munch import DefaultMunch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

#  Store every settings here
config = {
    "train_settings": {  # Set Training settings here
        "root": PROJECT_ROOT,
        "dataset": "drama_dataset",  # dataset name (refer below 'dataset' section)
        "model": "storygan",  # model name (refer below 'model' section)
        "mode": "train",  # Choose from ['train', 'test', 'val']
        "batch_size": 10,  # batch size
        "num_workers": 0,  # number of workers; 0 recommended to prevent shared memory error
        "n_epoch": 100,  # number of epochs
        "device": 'cuda',  # Choose from ['cuda', 'cpu']
        "d_lr": 0.0002,  # Discriminator Learning Rate
        "g_lr": 0.0002,  # Generator Learning Rate
        "g_n_step": 2,  # Number of generator step per iteration
        "vid_loss_ratio": 1.0,  # video loss ratio
        "log_cycle": 10,  # train loss logging cycle
        "save_cycle": 500,  # saving model cycle
        "load_pretrained": {"epoch": 9, "iteration": 499},  # loading pre-trained # None if train from beginning
    },
    "inference_settings": {
        "root": os.path.join(PROJECT_ROOT, 'result'),  # where results are stored
        "pretrained": {"epoch": 58, "iteration": 999},
        "mode": "train",  # dataset mode
        "dataset_index": [10, 11, 12, 13]  # index to inference
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
            "dis_channel": 96,
            "latent_img_dim": 15,
            "label_num": 2
        }
    }
}

config = DefaultMunch.fromDict(config, None)  # for attribute style access of dictionary type  ex) config.dataset ..etc
