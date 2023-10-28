from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel 
import torch
from datasets.custom import CustomAlignedDataset
from ema import EMA
import yaml
from utils import dict2namespace, remove_file
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import faulthandler

faulthandler.enable()
batch_size =  8 #@param {'type':'integer'}
device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
accumulate_grad_batches = 4
use_scheduler = True
n_epochs =   300#@param {'type':'integer'}
use_ema = True 
resume_ckpt = True
ckpt_path = 'results/BBDM_SDE_ema/best_ckpt.pth'
## learning rate
lr=1e-4 #@param {'type':'number'}

f = open('cf.yaml', 'r')
dict_config = yaml.load(f, Loader=yaml.FullLoader)
nconfig = dict2namespace(dict_config)
res_folder = 'results/BBDM_SDE_ema/'
os.makedirs(res_folder, exist_ok=True)


# score_model = torch.nn.DataParallel(UNetModel(**vars(nconfig.model.BB.params.UNetParams)))
score_model = UNetModel(**vars(nconfig.model.BB.params.UNetParams))

# score_model = ScoreNet(marginal_prob_std=marginal_prob_std)
score_model = score_model.to(device)

# initialize EMA
if use_ema:
  emA = EMA(nconfig.model.EMA.ema_decay)
  update_ema_interval = nconfig.model.EMA.update_ema_interval
  start_ema_step = nconfig.model.EMA.start_ema_step
  emA.register(score_model)

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

train_dataset = CustomAlignedDataset(nconfig.data.dataset_config)
val_dataset = CustomAlignedDataset(nconfig.data.dataset_config, 'val')
test_dataset = CustomAlignedDataset(nconfig.data.dataset_config, 'test')

## size of a mini-batch

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








from ldm import LDM
latent_encoder = LDM(nconfig.model.VQGAN.params)
latent_encoder = latent_encoder.to(device)

optimizer = Adam(score_model.parameters(), lr=lr)

if use_scheduler: 
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, threshold_mode='rel', **vars(nconfig.model.BB.lr_scheduler))

tqdm_epoch = range(n_epochs)

best_loss = 100000.0

for epoch in tqdm_epoch:
  print()
  print('Epoch: {}'.format(epoch))
  avg_loss = 0.
  num_items = 0
  global_step = 0 
  for batch in tqdm(train_loader):
    global_step += 1
    (x, x_name), (y, y_name) = batch
    score_model.train()

    x, y = x.to(device), y.to(device)
    x_latent = latent_encoder.encode(x)
    y_latent = latent_encoder.encode(y)

    loss = loss_fn(score_model, x_latent, y_latent, marginal_prob_std)
    
    loss.backward()
    if global_step % accumulate_grad_batches == 0:
      optimizer.step()
      optimizer.zero_grad()
      if scheduler is not None:
        scheduler.step(loss)

    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]

    if use_ema and global_step % (update_ema_interval * accumulate_grad_batches) == 0:
      with_decay = False if global_step < start_ema_step else True
      emA.update(score_model, with_decay=with_decay)
  

  # Print the averaged training loss so far.  
  print('Average Loss: {:5f}'.format(avg_loss / num_items))
  # write the loss to a file
  with open(res_folder +'/train_loss.txt', 'a') as f:
    f.write('epoch {}:'.format(epoch) + str(avg_loss / num_items))
    f.write('\n')
  
  # Evaluate the model on the validation set.
  avg_loss = 0.
  num_items = 0

  # TODO: use EMA
  if use_ema:
    emA.apply_shadow(score_model)

  score_model.eval()
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
    ckpt = {}
    ckpt['best_model'] = score_model.state_dict()
    ckpt['epoch'] = epoch
    ckpt['optimizer'] = optimizer.state_dict()
    ckpt['scheduler'] = scheduler.state_dict() if scheduler is not None else None
    if use_ema:
      ckpt['ema'] = emA.shadow
    remove_file(res_folder + '/best_ckpt.pth')
    torch.save(ckpt, res_folder + '/best_ckpt.pth')

  if use_ema:
    emA.restore(score_model)
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), res_folder + '/last_model.pth')


