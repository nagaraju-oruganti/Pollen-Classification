import random, os
import numpy as np
import torch
    
### SEED EVERYTHING
def seed_everything(seed: int = 42):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    print(f'set seed to {seed}')
    
class Config:
    
    # Seed
    seed = 3407
    seed_everything(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # repos
    data_dir = '../inputs/data'
    models_dir = '../models'
    
    # Image preprocess
    centered_zero = False       # True: rescales to -1, 1 
    height = 224
    width = 224
    
    # Augmentation
    aug_threshold = 1.1         # no augmentation
    max_rotation_angle = 30
    
    ## Train parameters
    train_batch_size = 16
    valid_batch_size = 32
    iters_to_accumlate = 1
    learning_rate = 4e-5
    num_epochs = 100
    
    ## Model params
    in_channels = 20        # frames in the image
    
    ## Run params
    sample_run = False
    save_epoch_wait = 1
    early_stop_count = 10
    save_checkpoint = True