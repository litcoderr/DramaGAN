import torch
import torch.nn as nn

from models.storygan.layers import DynamicFilterLayer, upSampleBlock


class StoryGan(nn.Module):
    def __init__(self, config):
        super(StoryGan, self).__init__()
        ### PARAMETERS CONFIG ###
        self.dataset_config = config.dataset.drama_dataset
        self.model_config = config.model.storygan
        self.train_config = config.train_settings

        ### INITIALIZE MODULE ###

        # Encode text input
        self.fc_text_encode = nn.Linear(self.model_config.text_dim, self.model_config.text_hidden_dim)
        # desc-context encoding GRU
        self.desc_context_gru = nn.GRUCell(self.model_config.text_hidden_dim, self.model_config.text_hidden_dim)
        self.desc_sample_gru = nn.GRUCell(self.model_config.text_hidden_dim + self.model_config.noise_dim,
                                          self.model_config.text_hidden_dim)

        # Encoded Description Sample and Context Sample Fusion
        self.zdc_sample_fc = nn.Sequential(
            nn.Linear(self.model_config.text_hidden_dim * 2, int(self.model_config.gen_channel * 4 * 2)),
            nn.BatchNorm1d(int(self.model_config.gen_channel * 4 * 2)),
            nn.ReLU(True)
        )

        # latent image generator from desc_sample
        self.latent_image_generator = nn.Sequential(
            nn.Linear(self.model_config.text_hidden_dim, self.model_config.latent_img_dim ** 2),
            nn.BatchNorm1d(self.model_config.latent_img_dim ** 2)
        )

        # filter generator from z_desc_context
        self.filter_generator = nn.Sequential(
            nn.Linear(self.model_config.text_hidden_dim, self.model_config.latent_img_dim ** 2, bias=False),
            nn.BatchNorm1d(self.model_config.latent_img_dim ** 2)
        )

        # Dynamic Filter Layer
        filter_size = self.model_config.latent_img_dim
        self.dynamic_filter = DynamicFilterLayer((filter_size, filter_size, 1),
                                                 pad=(filter_size//2, filter_size//2), grouping=False)

        # downsampler
        self.downsample = nn.Sequential(
            nn.Conv2d(1, self.model_config.gen_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_config.gen_channel),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.model_config.gen_channel, self.model_config.gen_channel//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.model_config.gen_channel//2),
            nn.LeakyReLU(0.2, inplace=True),
            )

        latent_dim = self.model_config.gen_channel
        self.upblock1 = upSampleBlock(in_channel=latent_dim, out_channel=latent_dim//2)
        self.upblock2 = upSampleBlock(in_channel=latent_dim//2, out_channel=latent_dim//4)
        self.upblock3 = upSampleBlock(in_channel=latent_dim//4, out_channel=latent_dim//8)
        self.upblock4 = upSampleBlock(in_channel=latent_dim//8, out_channel=latent_dim//16)
        self.upblock5 = upSampleBlock(in_channel=latent_dim//16, out_channel=latent_dim//32)
        self.upblock6 = upSampleBlock(in_channel=latent_dim//32, out_channel=latent_dim//64)
        self.conv_block = nn.Sequential(
            nn.Conv2d(latent_dim//64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def desc_noise_filter(self, original_desc):
        batch_size = original_desc.shape[0]
        noise = torch.FloatTensor(batch_size, self.model_config.noise_dim).normal_(0, 1).to(self.train_config.device)

        # output shape: [batch_size, text_hidden_dim + noise_dim (150)]
        return torch.cat((original_desc, noise), dim=1)

    def desc_context_rnn(self, desc, context):
        """
        Fusion Description and Context through GRU
        :param desc: torch.FloatTensor shape: [batch_size, dim] or [batch_size, seq_length, dim]
        :param context: torch.FloatTensor shape: [batch_size, dim]
        :return:
            rnn_output: torch.FloatTensor shape: [batch_size, seq_length, dim]
        """
        video_len = 1 if len(desc.shape) == 2 else self.dataset_config.n_frame
        h_t = [context]
        if len(desc.shape) == 2:  # reshape to [batch_size, 1, dim]
            desc = desc.unsqueeze(dim=1)

        # Forward through GRU Cell
        for frame_index in range(video_len):
            h_t.append(self.desc_context_gru(desc[:, frame_index, :], h_t[-1]))

        # h_t shape: [batch_size * seq_length, text_hidden_dimension(75)]
        h_t = torch.stack(h_t[1:], dim=1).view(-1, self.model_config.text_hidden_dim)

        return h_t

    def desc_sample_rnn(self, desc_sample, seq_length=None):
        video_len = seq_length if seq_length is not None else self.dataset_config.n_frame
        if video_len > 1:
            h_t = [desc_sample[:, 0, :]]
        else:
            h_t = [desc_sample]

        # forward through GRU
        for frame_index in range(video_len):
            if video_len == 1:
                desc_sample_noise = self.desc_noise_filter(desc_sample)
            else:
                desc_sample_noise = self.desc_noise_filter(desc_sample[:, frame_index, :])
            h_t.append(self.desc_sample_gru(desc_sample_noise, h_t[-1]))

        # h_t shape: [batch_size * seq_length, text_hidden_dimension(75)]
        h_t = torch.stack(h_t[1:], dim=1).view(-1, self.model_config.text_hidden_dim)

        return h_t

    def sample_video(self, desc, context):
        """
        :param desc: Torch.FloatTensor shape: [batch_size, seq_length, text_dim(768)]
        :param context: Torch.FloatTensor shape: [batch_size, text_dim(768)]
        :return: Torch.FloatTensor shape: [batch_size, 3, seq_length,img_size(256), 256]
        """
        # 1. Encode description and context dim(768) -> dim(75)
        desc = self.fc_text_encode(desc)
        context = self.fc_text_encode(context)

        # 2-1. Encode description and context through GRU
        # shape: [batch_size * seq_length, text_hidden_dim(75)]
        z_desc_context = self.desc_context_rnn(desc, context)

        # 2-2. Sample desc and context using VAE
        # first reshape context and desc
        context = context.repeat(1, self.dataset_config.n_frame).view(
            context.shape[0]*self.dataset_config.n_frame, context.shape[-1])
        temp_desc = desc.view(-1, desc.shape[-1])

        # sample
        # TODO Need to test VAE
        desc_sample, desc_mu, desc_logvar = temp_desc, temp_desc, temp_desc
        context_sample, context_mu, context_logvar = context, context, context

        # 2-3. Encode only desc through GRU
        # shape: [batch_size * seq_length, text_hidden_dim(75)]
        z_desc_sample = self.desc_sample_rnn(desc)

        # 3-1. z_desc_sample and context_sample fusion
        # will be used for up-sampling to image
        # result shape: [batch_size * seq_length, 768, 4, 4]
        zdc_sample = torch.cat((z_desc_sample, context_sample), dim=1)
        zdc_sample = self.zdc_sample_fc(zdc_sample)
        zdc_sample = zdc_sample.view(-1, self.model_config.gen_channel // 2, 4, 4)

        # 3-2. Generate small image from desc_sample and filter using z_desc_context

        # 3-2-1. Generate small image from desc_sample
        # shape: [batch_size * seq_length, 1, latent_img_size(15), latent_img_size(15)]
        desc_sample_image = self.latent_image_generator(desc_sample)
        desc_sample_image = desc_sample_image.view(-1, 1,
                                                   self.model_config.latent_img_dim, self.model_config.latent_img_dim)

        # 3-2-2. Generate and apply Dynamic Filter
        # shape: [batch_size * seq_length, 768, 4, 4]
        z_desc_context_filter = self.filter_generator(z_desc_context)
        z_desc_context_filter = z_desc_context_filter.view(-1, 1,
                                                           self.model_config.latent_img_dim,
                                                           self.model_config.latent_img_dim)
        desc_context_image = self.dynamic_filter(
            [desc_sample_image, z_desc_context_filter])  # [batch_size, 1, latent_img_dim(15), 15]
        desc_context_image = self.downsample(desc_context_image)

        # 4. Fusion 3-1 and 3-2 and UpSample
        # result image shape: [batch_size, 3, seq_length, 256, 256]
        z_desc_context_all = torch.cat((zdc_sample, desc_context_image), dim=1)  # [batch_size, gen_channel(1536), 4, 4]
        img_hidden = self.upblock1(z_desc_context_all)
        img_hidden = self.upblock2(img_hidden)
        img_hidden = self.upblock3(img_hidden)
        img_hidden = self.upblock4(img_hidden)
        img_hidden = self.upblock5(img_hidden)
        img_hidden = self.upblock6(img_hidden)
        result_video = self.conv_block(img_hidden)  # [batch_size*seq_length, 3, 256, 256]

        result_video = result_video.view(result_video.shape[0]//self.dataset_config.n_frame,
                                         self.dataset_config.n_frame,
                                         3, self.model_config.img_size, self.model_config.img_size)
        result_video = result_video.permute(0, 2, 1, 3, 4)
        return result_video

    def sample_images(self, desc, context):
        """
        :param desc: Torch.FloatTensor shape: [batch_size, text_dim(768)]
            Description vector representation of image
        :param context: Torch.FloatTensor shape: [batch_size, text_dim(768)]
            Contextual vector representation (mean vector of text vectors of each scene)
        :return:
            generated_image: Torch.FloatTensor shape: [batch_size, channel(3), img_size(256), img_size(256)]
        """

        # 1. Encode description and context dim(768) -> dim(75)
        desc = self.fc_text_encode(desc)
        context = self.fc_text_encode(context)

        # 2. Sample desc and context by VAE
        # TODO Test out VAE -> not implemented in original code
        desc_sample, desc_mu, desc_logvar = desc, desc, desc
        context_sample, context_mu, context_logvar = context, context, context

        # 3-1. Encode description and context through GRU
        # shape: [batch_size * seq_length, text_hidden_dim(75)]
        z_desc_context = self.desc_context_rnn(desc, context)

        # 3-2. Encode only desc_sample through GRU
        # shape: [batch_size * seq_length, text_hidden_dim(75)]
        z_desc_sample = self.desc_sample_rnn(desc_sample, 1)

        # 4-1. z_desc_sample and context_sample fusion
        # will be used for up-sampling to image
        # result shape: [batch_size, 768, 4, 4]
        zdc_sample = torch.cat((z_desc_sample, context_sample), dim=1)
        zdc_sample = self.zdc_sample_fc(zdc_sample)
        zdc_sample = zdc_sample.view(-1, self.model_config.gen_channel // 2, 4, 4)

        # 4-2. Generate small image from desc_sample and filter using z_desc_context

        # 4-2-1. Generate small image from desc_sample
        # shape: [batch_size, 1, latent_img_size(15), latent_img_size(15)]
        desc_sample_image = self.latent_image_generator(desc_sample)
        desc_sample_image = desc_sample_image.view(-1, 1,
                                                   self.model_config.latent_img_dim, self.model_config.latent_img_dim)

        # 4-2-2. Generate and apply Dynamic Filter
        # shape: [batch_size, 768, 4, 4]
        z_desc_context_filter = self.filter_generator(z_desc_context)
        z_desc_context_filter = z_desc_context_filter.view(-1, 1,
                                                           self.model_config.latent_img_dim,
                                                           self.model_config.latent_img_dim)
        desc_context_image = self.dynamic_filter([desc_sample_image, z_desc_context_filter])  # [batch_size, 1, latent_img_dim(15), 15]
        desc_context_image = self.downsample(desc_context_image)

        # 5. Fusion 4-1 and 4-2 and UpSample
        # result image shape: [batch_size, 3, 256, 256]
        z_desc_context_all = torch.cat((zdc_sample, desc_context_image), dim=1)  # [batch_size, gen_channel(1536), 4, 4]
        img_hidden = self.upblock1(z_desc_context_all)
        img_hidden = self.upblock2(img_hidden)
        img_hidden = self.upblock3(img_hidden)
        img_hidden = self.upblock4(img_hidden)
        img_hidden = self.upblock5(img_hidden)
        img_hidden = self.upblock6(img_hidden)
        result_image = self.conv_block(img_hidden)

        return result_image
