from trainer import init_trainer
from config import config

# To resolve too many files open error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# get trainer based on config
trainer = init_trainer(config)
if __name__ == '__main__':
    trainer.train()
