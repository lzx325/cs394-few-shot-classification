import torch
import numpy as np
from typing import Dict
from tqdm import tqdm

DATA_CONFIG = {
    'root_dir': './data',
    'target_img_size': 64,
}


METHODS_CONFIGS = {
    'unpretrained_baseline': {
        'name': 'unpretrained_baseline'
    },
    'pretrained_baseline': {
        'name': 'pretrained_baseline'
    },
    'protonet': {
        'name': 'protonet',
        # TODO(protonet): your ProtNet hyperparams (if there are any)
    },
    'maml': {
        'name': 'maml',
        # TODO(maml): your maml hyperparams
<<<<<<< HEAD
        'num_inner_steps':5,
=======
        'inner_loop_lr': 'TODO',
        'num_inner_steps': 'TODO',
        'ft_optim_kwargs': 'TODO',
        # ... other MAML hyperparameters?
>>>>>>> 64d4539f27c55bb8fa7b92415490061811888f91
    },
}

TRAINING_CONFIG = {
    'unpretrained_baseline': {
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 0,
        'optim_kwargs': {'lr': 0.001},
    },
    'pretrained_baseline': {
<<<<<<< HEAD
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 100,
        'optim_kwargs': {'lr': 0.001},
    },
    'protonet': {
        # TODO(protonet): your pretrained_baseline hyperparams
        'batch_size': 20,
        'num_train_steps_per_episode': 50,
        'num_train_episodes': 200,
        'optim_kwargs': {'lr': 1e-3},
    },
    'maml': {
        # TODO(maml): your pretrained_baseline hyperparams
        'batch_size':20,
        'inner_lr':0.1,
        'num_train_episodes':10000,
        'optim_kwargs':{'lr':1e-3}
=======
        # TODO(pretrained_baseline): your pretrained_baseline hyperparams
        'batch_size': 'TODO',
        'num_train_steps_per_episode': 'TODO',
        'num_train_episodes': 'TODO',
        'optim_kwargs': 'TODO',
        # ... other Pretrained Baseline hyperparameters?
    },
    'protonet': {
        # TODO(protonet): your ProtoNet hyperparams
        'batch_size': 'TODO',
        'num_train_steps_per_episode': 'TODO',
        'num_train_episodes': 'TODO',
        'optim_kwargs': 'TODO',
        # ... other ProtoNet hyperparameters?
    },
    'maml': {
        # TODO(maml): your MAML hyperparams
        'batch_size': 'TODO',
        'num_train_steps_per_episode': 'TODO',
        'num_train_episodes': 'TODO',
        'optim_kwargs': 'TODO',
        # ... other MAML hyperparameters?
>>>>>>> 64d4539f27c55bb8fa7b92415490061811888f91
    },
}

COMMON_TRAINING_CONFIG = {
    'random_seed': 42,
    'num_shots': 9, # TODO: this is what you may vary to have different K values
    'num_classes_per_task': 5,
}


def construct_config(method_name: str, **overwrite_kwargs) -> Dict:
    default_config = {
        'data': DATA_CONFIG,
        'model': METHODS_CONFIGS[method_name],
        'training': {**COMMON_TRAINING_CONFIG, **TRAINING_CONFIG[method_name]},
        'device': ('cuda' if torch.cuda.is_available() else 'cpu')
    }
    final_config = {**default_config, **overwrite_kwargs}

    return final_config
