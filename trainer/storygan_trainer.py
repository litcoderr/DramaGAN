from trainer.trainer import Trainer
from models import StoryGan
from dataset import DramaDataset
from dataset.drama_dataset import custom_collate, preprocess_batch

import torch
import random


class StoryGanTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config=config)
        self.dataset = DramaDataset(config=self.config, mode=self.config.train_settings.mode)
        self.dataset_size = len(self.dataset)
        self.video_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.train_settings.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.config.train_settings.num_workers,
            collate_fn=custom_collate
        )
        self.image_batch_iterator = iter(torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.train_settings.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.config.train_settings.num_workers,
            collate_fn=custom_collate
        ))

        self.model = StoryGan(self.config).to(self.config.train_settings.device)

    def train(self):
        for epoch in range(self.config.train_settings.n_epoch):
            for video_description, video in self.video_loader:
                """
                Parameters
                ----------
                text: list[text_vector] -> text_vector shape: [text_seq_length, dimension]
                images: list[list[image_vector]] -> image_vector shape: [channel(3), width, height]
                """
                ############# INPUT DATA PRE-PROCESSING ############
                # 1-1. preprocess video_description and video vector -> send to desired device
                # video_description shape: [batch_size, image_seq_length, text_dim(768)]
                # video shape: [batch_size, image_seq_length, channel(3), img_size(256), img_size(256)]
                # will be used for image sequence generation
                video_description, video = preprocess_batch(self.config, video_description, video)
                video_context = video_description.clone()[:, 0, :]  # [batch_size, text_dim(768)]

                # 1-2. get batch of single description and corresponding image
                # will be used  for single image generation
                # image_description shape: [batch_size, text_dim(768)]
                # image shape: [batch_size, channel(3), img_size(256), img_size(256)]
                image_description, image = self.sample_image_batch()
                image_context = image_description.clone()  # [batch_size, text_dim(768)]

                ############# FEED FORWARD ############
                # 2-1. Generate Single Image
                generated_image = self.model.sample_images(desc=image_description, context=image_context)
                generated_video = self.model.sample_video(desc=video_description, context=video_context)
                pass

    def sample_image_batch(self):
        """
        Call and get batch of single image and corresponding description
        :return:
            description: torch.FloatTensor shape: [batch_size, text_dimension]
            image: torch.FloatTensor shape: [batch_size, channel(3), img_size, img_size]
        """
        random_index = list(map(lambda _: random.randint(0, self.config.dataset.drama_dataset.n_frame-1), range(self.config.train_settings.batch_size)))
        # make index vector for torch.gather
        desc_gather_index = torch.LongTensor(random_index).view(-1, 1, 1).expand(
            self.config.train_settings.batch_size, self.config.dataset.drama_dataset.n_frame, self.config.model.storygan.text_dim)\
            .to(self.config.train_settings.device)

        images_gather_index = torch.LongTensor(random_index).view(-1, 1, 1, 1, 1).expand(
            self.config.train_settings.batch_size, self.config.dataset.drama_dataset.n_frame, 3, self.config.model.storygan.img_size, self.config.model.storygan.img_size)\
            .to(self.config.train_settings.device)

        # get batch
        # descriptions [batch_size, seq_length, text_dim]
        # images [batch_size, seq_length, channel(3), img_size, img_size]
        descriptions, images = next(self.image_batch_iterator)
        descriptions, images = preprocess_batch(self.config, descriptions, images)

        # sample by random index
        descriptions = torch.gather(descriptions, dim=1, index=desc_gather_index)[:, 0, :]
        images = torch.gather(images, dim=1, index=images_gather_index)[:, 0, :, :, :]

        return descriptions, images
