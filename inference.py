from config import config
from trainer.utils import session_key_gen

from trainer.storygan_trainer import ModelController as StoryGanController
from dataset import DramaDataset
from dataset.drama_dataset import custom_collate, preprocess_batch

import os
import numpy as np
from torchvision import transforms
from PIL import Image
import shutil

inference_dict = {
    'storygan_drama_dataset': (StoryGanController, DramaDataset)
}


def gen_file_name(root, mode, dataset_index, frame_index):
    path = os.path.join(root, '{}_{}'.format(mode, dataset_index))
    fake_img_path = os.path.join(path, '{}_{}_{}_fake.jpg'.format(mode, dataset_index, frame_index))
    real_img_path = os.path.join(path, '{}_{}_{}_real.jpg'.format(mode, dataset_index, frame_index))
    return path, fake_img_path, real_img_path


if __name__ == '__main__':
    mode = config.inference_settings.mode
    model_controller, dataset = inference_dict[session_key_gen(config)]
    model_controller = model_controller(config=config)
    (model, _, _), _, _ = model_controller.load_model(pretrained_dict=config.inference_settings.pretrained)
    dataset = dataset(config=config, mode=mode)

    # setup folder for inference if not exist
    root = config.inference_settings.root
    if not os.path.exists(root):
        os.makedirs(root)

    for dataset_index in config.inference_settings.dataset_index:
        print('inference mode: {} dataset_index: {}'.format(mode, dataset_index))
        if dataset_index >= len(dataset):
            print('index {} out of bound of length {}'.format(dataset_index, len(dataset)))
            break
        text, images = dataset[dataset_index]
        text_batch, images_batch = custom_collate([(text, images)])  # [batch_size, seq_length(5), text_dim(768)]
        text_batch, images_batch = preprocess_batch(config, text_batch, images_batch)
        text_context = text_batch.clone()[:, 0, :]  # [batch_size, text_dim(768)]

        # generate video
        video, _, _, _, _ = model.sample_video(desc=text_batch, context=text_context)
        video = video[0].permute(1, 0, 2, 3)  # [seq_length, 3, width, height]

        path, _, _ = gen_file_name(root, mode, dataset_index, 0)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        else:
            shutil.rmtree(path)

        for frame_index in range(video.shape[0]):
            _, fake_image_path, real_image_path = gen_file_name(root, mode, dataset_index, frame_index)
            fake_image = video[frame_index]

            original_img = transforms.ToPILImage()(images[frame_index]).convert("RGB")
            original_img.save(real_image_path)

            fake_image = transforms.ToPILImage()(fake_image.cpu()).convert("RGB")
            fake_image.save(fake_image_path)
