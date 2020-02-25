
def session_key_gen(config):
    key = '{model_name}_{dataset_name}'.format(model_name=config.train_settings.model,
                                               dataset_name=config.train_settings.dataset)
    return key
