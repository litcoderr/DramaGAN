from trainer import init_trainer
from config import config

trainer = init_trainer(config)
if __name__ == '__main__':
    trainer.train()
