from trainer.storygan_trainer import StoryGanTrainer

trainer_dict = {
    'storygan_drama_dataset': StoryGanTrainer
}


def init_trainer(config):
    key = '{model_name}_{dataset_name}'.format(model_name=config.train_settings.model,
                                               dataset_name=config.train_settings.dataset)
    return trainer_dict[key](config=config)
