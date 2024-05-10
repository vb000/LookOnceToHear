import os
import importlib
import json

import wandb

def import_attr(import_path):
    module, attr = import_path.rsplit('.', 1)
    return getattr(importlib.import_module(module), attr)

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def get_wandb_id(run_dir=None):
    """
    Get the wandb id of the run_dir, otherwise generate a new one and
    save it to run_dir/wandb_id.txt.
    """
    if run_dir is None or not os.path.exists(os.path.join(run_dir, 'wandb_id.txt')):
        id = wandb.util.generate_id()
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'wandb_id.txt'), 'w') as f:
            f.write(id)
    with open(os.path.join(run_dir, 'wandb_id.txt')) as f:
        return f.read().strip()
