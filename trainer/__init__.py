from trainer.storygan_trainer import StoryGanTrainer
from trainer.utils import session_key_gen

trainer_dict = {
    'storygan_drama_dataset': StoryGanTrainer
}


def init_trainer(config):
    return trainer_dict[session_key_gen(config)](config=config)
