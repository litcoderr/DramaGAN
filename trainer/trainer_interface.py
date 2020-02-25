
class Trainer(object):
    """
    Trainer Interface
    """
    def __init__(self, config):
        self.config = config

    def train(self):
        raise NotImplementedError
