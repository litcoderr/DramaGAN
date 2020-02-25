import torch.nn as nn
import torch
from models.storygan.layers import conv3x3

# 코드 그대로 따옴
class D_GET_LOGITS(nn.Module):
    """
    Discriminator building block (used for getting logits)
    """
    def __init__(self, ndf, nef, video_len = 1, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        self.video_len = video_len
        if bcondition:
            self.conv1 = nn.Sequential(
                conv3x3(ndf * 8 * video_len, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
            self.conv2 = nn.Sequential(
                nn.Conv2d(ndf * 8, ndf*4, kernel_size=4, stride=4),
                nn.Sigmoid()
                )
            self.conv3 = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
            )
            self.convc = nn.Sequential(
                conv3x3(self.ef_dim, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(ndf * 8, ndf*4, kernel_size=4, stride=4),
                nn.Sigmoid())
            self.conv3 = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
            )
        # if video_len > 1:
        #     self.storynet = nn.GRUCell(self.ef_dim, self.ef_dim)

    def forward(self, features):
        # conditioning output
        if type(features) is tuple:
            h_code, c_code = features
        else:
            h_code = features
            c_code = None

        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 16, 16)
            c_code = self.convc(c_code)
            h_code = self.conv1(h_code)
            h_c_code = h_code * c_code
        else:
            h_c_code = h_code
        output = self.conv2(h_c_code)
        output = self.conv3(output)
        return output.view(-1)


class D_IMG(nn.Module):
    """
    Image Discriminator
    """
    def __init__(self, config, use_categories=False):
        super(D_IMG, self).__init__()
        self.config = config.model.storygan

        self.df_dim = self.config.dis_channel
        self.ef_dim = self.config.text_hidden_dim
        self.label_num = self.config.label_num
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None
        self.get_cond_logits = D_GET_LOGITS(ndf, nef, 1)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


class D_STY(nn.Module):
    """
    Story Discriminator
    """
    def __init__(self, config):
        super(D_STY, self).__init__()
        self.dataset_config = config.dataset.drama_dataset
        self.config = config.model.storygan

        self.df_dim = self.config.dis_channel
        self.ef_dim = self.config.text_hidden_dim
        self.text_dim = self.config.text_hidden_dim
        self.label_num = self.config.label_num
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef, self.dataset_config.n_frame)
        self.get_uncond_logits = None
        self.cate_classify = None

    def forward(self, story):
        N, C, video_len, W, H = story.shape
        story = story.permute(0,2,1,3,4)
        story = story.contiguous().view(-1, C,W,H)
        story_embedding = torch.squeeze(self.encode_img(story))
        _, C1, W1, H1 = story_embedding.shape
        #story_embedding = story_embedding.view(N,video_len, C1, W1, H1)
        #story_embedding = story_embedding.mean(1).squeeze()
        story_embedding = story_embedding.permute(2,3,0,1)
        story_embedding = story_embedding.view( W1, H1, N,video_len * C1)
        story_embedding = story_embedding.permute(2,3,0,1)
        return story_embedding