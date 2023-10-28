
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel 
import torch
from datasets.custom import CustomAlignedDataset
import yaml
from utils import dict2namespace
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import faulthandler

faulthandler.enable()

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

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


def diffusion_coeff( t):
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

# sigma =  25.0#@param {'type':'number'}
# marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
# diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def loss_fn(model, x_0, y,  marginal_prob_std ,eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x_0.shape[0], device=x_0.device) * (1. - eps) + eps
  z = torch.randn_like(x_0)
  std = marginal_prob_std(random_t)
  one_minus_t = 1. - random_t
  perturbed_x = x_0 * one_minus_t[:, None, None, None] + y * random_t[:, None, None, None] + z * std[:, None, None, None] 
  score = model(perturbed_x, random_t, marginal_prob_std)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


f = open('cf.yaml', 'r')
dict_config = yaml.load(f, Loader=yaml.FullLoader)
nconfig = dict2namespace(dict_config)
res_folder = 'results/BBDM_SDE_v1/'
os.makedirs(res_folder, exist_ok=True)


train_dataset = CustomAlignedDataset(nconfig.data.dataset_config)
val_dataset = CustomAlignedDataset(nconfig.data.dataset_config, 'val')
test_dataset = CustomAlignedDataset(nconfig.data.dataset_config, 'test')

## size of a mini-batch

batch_size =  8 #@param {'type':'integer'}

train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            drop_last=True)
val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    num_workers=8,
                                    drop_last=True)
test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     num_workers=8,
                                     drop_last=True)




# score_model = torch.nn.DataParallel(UNetModel(**vars(nconfig.model.BB.params.UNetParams)))
score_model = UNetModel(**vars(nconfig.model.BB.params.UNetParams))

# score_model = ScoreNet(marginal_prob_std=marginal_prob_std)
score_model = score_model.to(device)

n_epochs =   200#@param {'type':'integer'}

## learning rate
lr=1e-4 #@param {'type':'number'}

from ldm import LDM
latent_encoder = LDM(nconfig.model.VQGAN.params)
latent_encoder = latent_encoder.to(device)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = range(n_epochs)

best_loss = 100000.0

for epoch in tqdm_epoch:
  print()
  print('Epoch: {}'.format(epoch))
  avg_loss = 0.
  num_items = 0

  for batch in tqdm(train_loader):
    (x, x_name), (y, y_name) = batch
    score_model.train()

    x, y = x.to(device), y.to(device)
    x_latent = latent_encoder.encode(x)
    y_latent = latent_encoder.encode(y)

    loss = loss_fn(score_model, x_latent, y_latent, marginal_prob_std)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  print('Average Loss: {:5f}'.format(avg_loss / num_items))
  # write the loss to a file
  with open(res_folder +'/train_loss.txt', 'a') as f:
    f.write('epoch {}:'.format(epoch) + str(avg_loss / num_items))
    f.write('\n')
  # Evaluate the model on the validation set.
  score_model.eval()
  avg_loss = 0.
  num_items = 0

  for batch in tqdm(val_loader):
    with torch.no_grad():
      (x, x_name), (y, y_name) = batch
      x, y = x.to(device), y.to(device)
      x_latent = latent_encoder.encode(x)
      y_latent = latent_encoder.encode(y)
      loss = loss_fn(score_model, x_latent, y_latent, marginal_prob_std)
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
  # Print the averaged validation loss.
  print('Validation Loss: {:5f}'.format(avg_loss / num_items))
  # Save the model if the validation loss is the best we've seen so far.
  loss = avg_loss / num_items 
  with open( res_folder + '/val_loss.txt', 'a') as f:
    f.write('epoch {}:'.format(epoch) + str(loss))
    f.write('\n')
  if loss < best_loss:
    best_loss = loss
    torch.save(score_model.state_dict(), res_folder + '/best_ckpt_{}.pth'.format(epoch))

  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), res_folder + '/ckpt.pth')


