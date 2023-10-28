from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel 
from datasets.custom import CustomAlignedDataset
from torch.utils.data import DataLoader
import sampling_method as sm

import torch
import yaml
from utils import dict2namespace
from tqdm.autonotebook import tqdm

device = 'cuda'
batch_size = 8
f = open('cf.yaml', 'r')
weight_path = './best_ckpt_86.pth'

# config
dict_config = yaml.load(f, Loader=yaml.FullLoader)
nconfig = dict2namespace(dict_config)

# dataset
test_dataset = CustomAlignedDataset(nconfig.data.dataset_config, 'test')

# score func model
score_model = UNetModel(**vars(nconfig.model.BB.params.UNetParams))
    
    # load pretrained model
score_model.load_state_dict(torch.load(weight_path)) 
score_model = score_model.to(device)

# sampling with Numer SDE Solvers
num_steps =  500#@param {'type':'integer'}


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
  drift = drift / (1 - t)
  return drift



sampler = sm.Euler_Maruyama_sampler
test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     num_workers=8,
                                     drop_last=True)

from ldm import LDM
latent_encoder = LDM(nconfig.model.VQGAN.params)
latent_encoder = latent_encoder.to(device)

for batch in tqdm(test_loader):
  (x, x_name), (y, y_name) = batch
   
  
  y = y.to(device)
  y_latent = latent_encoder.encode(y)
  samples = sampler(y_latent, score_model, marginal_prob_std, drift_coeff, diffusion_coeff, batch_size=y.shape[0], num_steps=num_steps, device=device)
  samples = latent_encoder.decode(samples)
  break