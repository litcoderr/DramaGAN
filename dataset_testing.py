from config import config
from dataset import DramaDataset

if __name__ == '__main__':
    mode = "train"
    dataset = DramaDataset(config=config, mode=mode)
    pass
