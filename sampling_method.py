
import torch
from tqdm.autonotebook import tqdm
import numpy as np
from utils import to_flattened_numpy, from_flattened_numpy

# num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler( y, 
                        score_model,
                        marginal_prob_std,
                           drift_coeff,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=500,
                           device='cuda',
                           eps=1e-3):

  t = torch.ones(batch_size, device=device)
#   init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
#     * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1.-eps, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = y.clone()
  with torch.no_grad():
    for time_step in tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      f = drift_coeff(batch_time_step)[:, None, None, None] * (y-x)

      # print('time step {} ----------------------------------------------'.format(time_step))
      # print(x)
      
      mean_x = x + ( (g**2)[:, None, None, None] - f )* score_model(x, batch_time_step,marginal_prob_std) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
  # Do not include any noise in the last sampling step.
  return mean_x



#the Predictor-Corrector sampler (double click to expand or collapse)
# signal_to_noise_ratio = 0.16 #@param {'type':'number'}
## The number of sampling steps.
# num_steps =  500#@param {'type':'integer'}
def pc_sampler(y, score_model, 
               marginal_prob_std,
               drift_coeff,
               diffusion_coeff,
               batch_size=64,
               num_steps=500,
               snr=0.16,
               device='cuda',
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps.
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

  Returns:
    Samples.
  """
  # t = torch.ones(batch_size, device=device)
  init_x = y.clone()
  time_steps = np.linspace(1-eps, eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      
      # print('time step {} ----------------------------------------------'.format(time_step))
      # print(x)

      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step, marginal_prob_std)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      f = drift_coeff(batch_time_step)[:, None, None, None] * (y-x)
      score = score_model(x, batch_time_step, marginal_prob_std)
      x_mean = x +( (g**2)[:, None, None, None] - f ) * score * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

      # check nan values
      if torch.isnan(x).any():
        print('x is nan---------------------------')
        print(' time_step: ', time_step)
        print(' torch.sqrt(g**2 * step_size)', torch.sqrt(g**2 * step_size))
        print(' score: ', score)
        print(' (g**2)[:, None, None, None] - f: ', (g**2)[:, None, None, None] - f)
        print(' langevin_step_size: ', langevin_step_size)
        return x_mean

    # The last step does not include any noise
    return x_mean
  


  #@title Define the ODE sampler (double click to expand or collapse)

from scipy import integrate

## The error tolerance for the black-box ODE solver
# error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(y, score_model,
                marginal_prob_std,
                drift_coeff,
                diffusion_coeff,
                batch_size=64,
                atol=1e-5,
                rtol=1e-5,
                device='cuda',
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    # init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    #   * marginal_prob_std(t)[:, None, None, None]
    init_x = y.clone()
  else:
    init_x = z

  shape = init_x.shape

  def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift = drift_coeff[:, None, None, None] * (y-x)
        score = score_model(x, t, marginal_prob_std)
        drift = drift - diffusion_coeff[:, None, None, None] ** 2 * score * 0.5
        # Set the diffusion function to zero for ODEs.
        diffusion = 0
        return drift, diffusion

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    
    x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
    vec_t = torch.ones(shape[0], device=x.device) * t
    drift, diffusion = sde(x, vec_t)
    return  to_flattened_numpy(drift)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x
