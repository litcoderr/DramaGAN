import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


# 원본 코드 거의 그대로 따옴
# catelogits 이라는 부분은 categorical loss 를 계산하는 부분. drama dataset에서는 불필요
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DynamicFilterLayer(nn.Module):  # MergeLayer
    def __init__(self, filter_size, stride=(1, 1), pad=(0, 0), flip_filters=False, grouping=False):
        super(DynamicFilterLayer, self).__init__()
        self.filter_size = filter_size  # tuple 3
        self.stride = stride  # tuple 2
        self.pad = pad  # tuple 2
        self.flip_filters = flip_filters
        self.grouping = grouping

    def get_output_shape_for(self, input_shapes):
        if self.grouping:
            shape = (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2], input_shapes[0][3])
        else:
            shape = (input_shapes[0][0], 1, input_shapes[0][2], input_shapes[0][3])
        return shape

    def forward(self, _input, **kwargs):
        # def get_output_for(self, _input, **kwargs):
        image = _input[0]
        filters = _input[1]

        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        filter_size = self.filter_size

        if self.grouping:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)),
                                               (np.prod(filter_size), filter_size[0], filter_size[1]))
            filter_localexpand = filter_localexpand_np.float()

            outputs = []

            for i in range(3):
                input_localexpand = F.Conv2d(image[:, [i], :, :], kerns=filter_localexpand,
                                             subsample=self.stride, border_mode=border_mode, conv_mod=conv_mode)
                output = torch.sum(input_localexpand * filters[i], dim=1, keepdim=True)
                outputs.append(output)

            output = torch.cat(outputs, dim=1)

        else:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size)),
                                               (np.prod(filter_size), filter_size[2], filter_size[0], filter_size[1]))
            filter_localexpand = torch.from_numpy(filter_localexpand_np.astype('float32')).cuda()
            # filter_localexpand [filtersize(15)*filtersize, 1, filtersize, filtersize] -> every [1, filtersize, filtersize] only one element is 1. else 0.
            input_localexpand = F.conv2d(image, filter_localexpand,
                                         padding=self.pad)  # [batch_size, filtersize(15)*filtersize, width(15), height]
            output = torch.sum(input_localexpand * filters, dim=1, keepdim=True)  # [batch_size, width(15), height(15)]

        return output


def upSampleBlock(in_channel, out_channel):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(True))
    return block


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels, real_catelabels,
                               conditions):
    ratio = 1.0
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = netD(real_imgs)
    fake_features = netD(fake)
    # real pairs
    inputs = (real_features, cond)
    real_logits = netD.get_cond_logits(inputs)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], cond[1:])
    wrong_logits = netD.get_cond_logits(inputs)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = netD.get_cond_logits(inputs)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = netD.get_cond_logits(real_features)
        fake_logits = netD.get_cond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    acc = 0
    if netD.cate_classify is not None:
        cate_logits = netD.cate_classify(real_features)
        cate_logits = cate_logits.squeeze()
        errD = errD + ratio * cate_criterion(cate_logits, real_catelabels)
        acc = accuracy_score(real_catelabels.cpu().data.numpy().astype('int32'),
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD, errD_real.data, errD_wrong.data, errD_fake.data, acc


def compute_generator_loss(netD, fake_imgs, real_labels, fake_catelabels, conditions):
    ratio = 0.4
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    cond = conditions.detach()
    fake_features = netD(fake_imgs)
    # fake pairs
    inputs = (fake_features, cond)
    fake_logits = netD.get_cond_logits(inputs)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = netD.get_cond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    acc = 0
    # If using category calssification
    if netD.cate_classify is not None:
        cate_logits = netD.cate_classify(fake_features)
        cate_logits = cate_logits.squeeze()
        errD_fake = errD_fake + ratio * cate_criterion(cate_logits, fake_catelabels)
        acc = accuracy_score(fake_catelabels.cpu().data.numpy().astype('int32'),
            (cate_logits.cpu().data.numpy() > 0.5).astype('int32'))
    return errD_fake, acc
