from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel 
from datasets.custom import CustomAlignedDataset
from torch.utils.data import DataLoader
import sampling_method as sm
from ema import EMA
from latent_space import LDM

import numpy as np
import torch
import yaml
from utils import dict2namespace
from tqdm.autonotebook import tqdm
import os

device = 'cuda'
# batch_size = 8
ckpt_path = './results/BBDM_SDE_ema/best_ckpt.pth'
ckpt = torch.load(ckpt_path)

use_ema = True


# config
f = open('cf.yaml', 'r')
dict_config = yaml.load(f, Loader=yaml.FullLoader)
nconfig = dict2namespace(dict_config)

max_pixel_pet = float(nconfig.data.dataset_config.max_pixel_ori)
batch_size = nconfig.data.train.batch_size

# ldm
latent_encoder = LDM(nconfig.model.VQGAN.params)
latent_encoder = latent_encoder.to(device)

# dataset
test_dataset = CustomAlignedDataset(nconfig.data.dataset_config, 'test')

# score func model
score_model = UNetModel(**vars(nconfig.model.BB.params.UNetParams))
  
# load pretrained model
score_model.load_state_dict(ckpt['best_model']) 
score_model = score_model.to(device)


if use_ema:
    emA = EMA(0.999)
    emA.shadow = ckpt['ema']
    emA.reset_device(score_model)
    emA.apply_shadow(score_model)

# sampling with Numer SDE Solvers
num_steps = 500#@param {'type':'integer'}

# tmp = ckpt_path.split('/')[:-1]
# result_path = os.path.join('/'.join(tmp), str(num_steps))
result_path = '/workdir/ssd2/nguyent_petct/tiennh/BBDM_SDE/results/ema_snr0.64'
os.makedirs(result_path, exist_ok=True)


def marginal_prob_std(t):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
  Args:
    t: A vector of time steps.
  Returns:
    The standard deviation.
  """
  t = torch.tensor(t, device=device)
  # return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))
  std = torch.sqrt(2. * (t - t ** 2))
  
  return std 

def diffusion_coeff(t):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  diff = torch.ones_like(t, device=device)
  diff = diff * torch.tensor(2.0).sqrt()
  # return torch.tensor(sigma**t, device=device)
  return diff 

def drift_coeff(t): 
  # return batch of 1 / (1 - t)
  drift = torch.ones_like(t, device=device)
  # drift = drift / (1 - t + 1e-3)
  # drift = drift / (1 - t)
  drift = drift * torch.tensor(1.0)
  return drift



test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=8,
                                     drop_last=False)

sampler = sm.pc_sampler



for batch in tqdm(test_loader):
  (x, x_name), (y, y_name) = batch
  y = y.to(device)
  y_latent = latent_encoder.encode(y)
  samples = sampler(y_latent, score_model, marginal_prob_std, drift_coeff, diffusion_coeff, batch_size=y.shape[0], num_steps=num_steps, snr=0.64 ,device=device)
  samples_1 = latent_encoder.decode(samples)
  # samples_1.clamp(0.0, 1.0)
  for i in range(batch_size): 
    image = samples_1[i]
    sample_path = os.path.join(result_path, x_name[i])
    image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.mul_(max_pixel_pet).add_(0.2).clamp_(0, max_pixel_pet).permute(1, 2, 0).to('cpu').numpy()
    file_name = x_name[i]
    print(sample_path)
    np.save(sample_path, image)
