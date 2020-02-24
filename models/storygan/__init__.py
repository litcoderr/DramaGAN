
import torch
import torch.nn as nn


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
        self.desc_sample_gru = nn.GRUCell(self.model_config.text_hidden_dim + self.model_config.noise_dim, self.model_config.text_hidden_dim)

        # Encoded Description Sample and Context Sample Fusion
        self.zdc_sample_fc = nn.Sequential(
            nn.Linear(self.model_config.text_hidden_dim * 2, int(self.model_config.gen_channel * 4 * 2)),
            nn.BatchNorm1d(int(self.model_config.gen_channel * 4 * 2)),
            nn.ReLU(True)
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

    def sample_images(self, desc, context):
        """
        :param desc: Torch.FloatTensor shape: [batch_size, text_dim(768)]
            Description vector representation of image
        :param context: Torch.FloatTensor shape: [batch_size, text_dim(768)]
            Contextual vector representation (mean vector of text vectors of each scene)
        :return:
            generated_image: Torch.FloatTensor shape: [batch_size, channel(3), img_size(256), img_size(256)]
        """
        generated_image = None

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

        return generated_image

