from trainer.trainer import Trainer
from models import StoryGan
from dataset import DramaDataset
from dataset.drama_dataset import custom_collate, preprocess_batch

import torch


class StoryGanTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config=config)
        self.dataset = DramaDataset(config=self.config, mode=self.config.train_settings.mode)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.train_settings.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.config.train_settings.num_workers,
            collate_fn=custom_collate
        )

        self.model = StoryGan().to(self.config.train_settings.device)

    def train(self):
        for epoch in range(self.config.train_settings.n_epoch):
            for text, images in self.data_loader:
                """
                Parameters
                ----------
                text: list[text_vector] -> text_vector shape: [text_seq_length, dimension]
                images: list[list[image_vector]] -> image_vector shape: [channel(3), width, height]
                """
                # preprocess text and image -> sent to desired device
                text, images = preprocess_batch(self.config, text, images)
                # now text shape: [batch_size, image_seq_length, dimension]
                # now images shape: [batch_size, image_seq_length, channel(3), width, height]
                pass
