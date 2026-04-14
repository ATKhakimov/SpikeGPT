########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import logging
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
from src.spikingjelly.clock_driven import functional
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

### Step 1: set training data ##########################################################################

datafile_train = "data/ru_train_full.npy"
os.environ['VOCAB_SIZE'] = '50258'  # rugpt3large fast tokenizer vocab

### Step 2: set model size #############################################################################

ctx_len = 1024
n_layer = 12
n_embd  = 512  # n_layer=12, n_embd=512 → ~100M params with vocab=50257

# 'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
model_type = 'RWKV'

### Step 3: set batch size #############################################################################

# A100 80GB: 100M model fits comfortably at batch 32
# batch_size must be divisible by B_GROUP_FORWARD and B_GROUP_BACKWARD in model.py
batch_size = 32

### Step 4: set learning rate, training mini-epochs ####################################################

lr_init = 6e-4
lr_final = 1e-5
# each mini-epoch = epoch_length_fixed random samples of length ctx_len
n_epoch = 350  # 350 × 10.24M = ~3.58B tokens
# 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_frequency = 10
epoch_save_path = 'checkpoints/spikegpt-ru-'

epoch_length_fixed = 10000

# Resume from last checkpoint if available
resume_from = None
import glob
checkpoint_files = sorted(glob.glob('checkpoints/spikegpt-ru-*.pth'),
                          key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('-')[-1]))
if checkpoint_files:
    resume_from = checkpoint_files[-1]
    print(f'Found latest checkpoint: {resume_from}')

########################################################################################################

import src.utils
src.utils.set_seed(42)

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

grad_norm_clip = 1.0
warmup_tokens = 0

betas = (0.9, 0.99)
eps = 4e-9

num_workers = 0

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile_train)
data_full = np.load(datafile_train, mmap_mode='r')
val_split = int(len(data_full) * 0.99)  # last 1% → validation
train_dataset = Dataset(data_full[:val_split], ctx_len, epoch_length_fixed)
valid_dataset = Dataset(data_full[val_split:], ctx_len, epoch_length_fixed=500)
########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

    # # load a trained model. remember to change random seed
#     m2 = torch.load('medium/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
#     model.load_state_dict(m2)
    test_dataset = None
    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, tconf, resume_from=resume_from)

    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
