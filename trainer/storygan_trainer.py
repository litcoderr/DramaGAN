from trainer.trainer_interface import Trainer
from trainer.utils import session_key_gen

from models import StoryGan, D_STY, D_IMG
from models.storygan.layers import compute_discriminator_loss, compute_generator_loss, KL_loss
from dataset import DramaDataset
from dataset.drama_dataset import custom_collate, preprocess_batch

import torch
import torch.optim as optim
import random
import json
import os


class ModelController:
    """
    Saves results
    """
    def __init__(self, config):
        self.config = config
        self.session_name = session_key_gen(self.config)
        self.save_root = os.path.join(self.config.train_settings.root, "pretrained", self.session_name)

    def gen_file_path(self, epoch, iteration):
        path = os.path.join(self.save_root, "{}_{}".format(epoch, iteration))
        model_path = os.path.join(path, "{}_{}_gen.pth".format(epoch, iteration))
        img_d_path = os.path.join(path, "{}_{}_imgd.pth".format(epoch, iteration))
        vid_d_path = os.path.join(path, "{}_{}_vidd.pth".format(epoch, iteration))
        losses_path = os.path.join(path, "{}_{}_losses.json".format(epoch, iteration))
        return path, model_path, img_d_path, vid_d_path, losses_path

    def load_model(self):
        before_epoch = 0
        before_step = -1
        model = StoryGan(self.config).to(self.config.train_settings.device)
        img_d = D_IMG(config=self.config).to(self.config.train_settings.device)
        vid_d = D_STY(config=self.config).to(self.config.train_settings.device)

        # load pretrained if needed and update epoch and step index
        if self.config.train_settings.load_pretrained is not None:
            desired_epoch = self.config.train_settings.load_pretrained["epoch"]
            desired_iteration = self.config.train_settings.load_pretrained["iteration"]
            path, model_path, img_d_path, vid_d_path, _ = self.gen_file_path(epoch=desired_epoch, iteration=desired_iteration)
            if os.path.exists(path):
                model.load_state_dict(torch.load(model_path))
                img_d.load_state_dict(torch.load(img_d_path))
                vid_d.load_state_dict(torch.load(vid_d_path))

                before_epoch = desired_epoch
                before_step = desired_iteration
                print('loaded pretrained from epoch: {} step: {}'.format(before_epoch, before_step))
            else:
                print('pretrained epoch: {} step: {} does not exist\nNothing loaded'.format(desired_epoch, desired_iteration))
        else:
            print('Nothing loaded')

        return (model, img_d, vid_d), before_epoch, before_step

    def save_model(self, models, epoch, iteration, losses):
        gen, img_d, vid_d = models
        path, gen_path, img_d_path, vid_d_path, losses_path = self.gen_file_path(epoch, iteration)

        # Make Directory
        os.makedirs(path, exist_ok=True)

        # Save state dict
        torch.save(gen.state_dict(), gen_path)
        torch.save(img_d.state_dict(), img_d_path)
        torch.save(vid_d.state_dict(), vid_d_path)

        # Save losses in json
        with open(losses_path, 'w') as losses_file:
            json.dump(losses, losses_file)

        return path


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
        self.video_batch_iterator = iter(self.video_loader)
        self.image_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.train_settings.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.config.train_settings.num_workers,
            collate_fn=custom_collate
        )
        self.image_batch_iterator = iter(self.image_loader)
        self.n_step = len(self.video_loader)

        # Models
        self.model_controller = ModelController(self.config)
        (self.model, self.img_d, self.vid_d), self.before_epoch, self.before_step = self.model_controller.load_model()

        # Optimizers
        self.img_d_optim = optim.Adam(self.img_d.parameters(), lr=self.config.train_settings.d_lr, betas=(0.5, 0.999))
        self.vid_d_optim = optim.Adam(self.vid_d.parameters(), lr=self.config.train_settings.d_lr, betas=(0.5, 0.999))

        g_param = []
        for p in self.model.parameters():
            if p.requires_grad:
                g_param.append(p)
        self.g_optim = optim.Adam(g_param, lr=self.config.train_settings.g_lr, betas=(0.5, 0.999))

    def train(self):
        image_real_labels = torch.FloatTensor(self.config.train_settings.batch_size).fill_(1).to(self.config.train_settings.device)
        image_fake_labels = torch.FloatTensor(self.config.train_settings.batch_size).fill_(0).to(self.config.train_settings.device)
        video_real_labels = torch.FloatTensor(self.config.train_settings.batch_size).fill_(1).to(self.config.train_settings.device)
        video_fake_labels = torch.FloatTensor(self.config.train_settings.batch_size).fill_(0).to(self.config.train_settings.device)

        print('start traing @ epoch: {} step: {}'.format(self.before_epoch, self.before_step+1))
        step_index = self.before_step
        for epoch in range(self.before_epoch, self.config.train_settings.n_epoch):
            while True:
                step_index += 1
                if step_index == self.n_step:
                    step_index = -1
                    break
                try:
                    video_description, video = next(self.video_batch_iterator)
                except StopIteration:
                    self.video_batch_iterator = iter(self.video_loader)
                    video_description, video = next(self.video_batch_iterator)

                """
                Parameters
                ----------
                text: list[text_vector] -> text_vector shape: [text_seq_length, dimension]
                images: list[list[image_vector]] -> image_vector shape: [channel(3), width, height]
                """
                ############# INPUT DATA PRE-PROCESSING ############
                # 1-1. preprocess video_description and video vector -> send to desired device
                # video_description shape: [batch_size, image_seq_length, text_dim(768)]
                # video shape: [batch_size, channel(3), image_seq_length, img_size(256), img_size(256)]
                # will be used for image sequence generation
                video_description, video = preprocess_batch(self.config, video_description, video)
                video_context = video_description.clone()[:, 0, :]  # [batch_size, text_dim(768)]
                video = video.permute(0, 2, 1, 3, 4)

                # 1-2. get batch of single description and corresponding image
                # will be used  for single image generation
                # image_description shape: [batch_size, text_dim(768)]
                # image shape: [batch_size, channel(3), img_size(256), img_size(256)]
                image_description, image = self.sample_image_batch()
                image_context = image_description.clone()  # [batch_size, text_dim(768)]


                ############# UPDATE DISCRIMINATOR ############
                # 2-1. Generate Single Image
                generated_image, img_desc_mu, img_desc_logvar = self.model.sample_images(desc=image_description,
                                                                                         context=image_context)
                generated_video, vid_c_mu, vid_c_logvar, vid_desc_mu, vid_desc_logvar = self.model.sample_video(
                    desc=video_description, context=video_context)

                # 2-2. Update Discriminator
                self.img_d.zero_grad()
                self.vid_d.zero_grad()

                img_err_d, im_err_d_real, im_err_d_wrong, im_err_fake, acc_d = compute_discriminator_loss(
                    self.img_d, image, generated_image, image_real_labels, image_fake_labels, None, img_desc_mu)

                vid_err_d, vid_err_d_real, vid_err_d_wrong, vid_err_fake, _ = compute_discriminator_loss(
                     self.vid_d, video, generated_video, video_real_labels, video_fake_labels, None, vid_c_mu)

                img_err_d.backward()
                vid_err_d.backward()

                self.img_d_optim.step()
                self.vid_d_optim.step()

                ############# UPDATE GENERATOR ############
                # Perform twice for generator and discriminator balance
                for _ in range(self.config.train_settings.g_n_step):
                    self.model.zero_grad()

                    generated_image, img_desc_mu, img_desc_logvar = self.model.sample_images(desc=image_description,
                                                                                             context=image_context)
                    generated_video, vid_c_mu, vid_c_logvar, vid_desc_mu, vid_desc_logvar = self.model.sample_video(
                        desc=video_description, context=video_context)

                    # compute generator loss
                    img_err_g, acc_g = compute_generator_loss(
                        self.img_d, generated_image, image_real_labels, None, img_desc_mu)
                    vid_err_g, _ = compute_generator_loss(
                        self.vid_d, generated_video, video_real_labels, None, vid_c_mu)

                    # compute kl divergence loss
                    # ....... no use since not using vae 하지만 원본 코드에는 이렇게 해서 잘 학습됐다..;;
                    img_kl_loss = KL_loss(img_desc_mu, img_desc_logvar)
                    vid_kl_loss = KL_loss(vid_c_mu, vid_c_logvar)
                    kl_loss = img_kl_loss + self.config.train_settings.vid_loss_ratio * vid_kl_loss

                    # fusion with generator loss and kl loss
                    g_loss_total = img_err_g + self.config.train_settings.vid_loss_ratio * vid_err_g + kl_loss

                    # update
                    g_loss_total.backward()
                    self.g_optim.step()

                if (step_index+1) % self.config.train_settings.log_cycle == 0:
                    # log training loss
                    print("epoch:[{current_epoch}/{total_epoch}] step:[{current_step}/{total_step}]img_d: {img_err_d} vid_d: {vid_err_d} g: {g_loss_total}".format(
                        current_epoch=epoch+1,
                        total_epoch=self.config.train_settings.n_epoch,
                        current_step=step_index+1,
                        total_step=self.n_step,
                        img_err_d=img_err_d.data,
                        vid_err_d=vid_err_d.data,
                        g_loss_total=g_loss_total.data
                    ))

                if (step_index+1) % self.config.train_settings.save_cycle == 0:
                    losses = {
                        "img_err_d": float(img_err_d.data),
                        "vid_err_d": float(vid_err_d.data),
                        "g_loss_total": float(g_loss_total.data)
                    }
                    # save model
                    save_path = self.model_controller.save_model(models=(self.model, self.img_d, self.vid_d), epoch=epoch, iteration=step_index, losses=losses)
                    print('saved model @: {}'.format(save_path))

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
        try:
            descriptions, images = next(self.image_batch_iterator)
        except StopIteration:
            self.image_batch_iterator = iter(self.image_loader)
            descriptions, images = next(self.image_batch_iterator)

        descriptions, images = preprocess_batch(self.config, descriptions, images)

        # sample by random index
        descriptions = torch.gather(descriptions, dim=1, index=desc_gather_index)[:, 0, :]
        images = torch.gather(images, dim=1, index=images_gather_index)[:, 0, :, :, :]

        return descriptions, images
