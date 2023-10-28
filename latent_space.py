import torch
import torch.nn as nn
from model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LDM(nn.Module):
    def __init__(self, vqgan_config):
        super().__init__()
        self.vqgan = VQModel(**vars(vqgan_config)).eval()
        # self.vqgan = self.vqgan.eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {vqgan_config.ckpt_path}")

    # def forward(self, x, x_cond, context=None):
    #     with torch.no_grad():
    #         x_latent = self.encode(x, cond=False)
    #         x_cond_latent = self.encode(x_cond, cond=True)
    #     return x_latent, x_cond_latent
    
    @torch.no_grad()
    def encode(self, x):
        model = self.vqgan 
        x_latent = model.encoder(x)
        x_latent = model.quant_conv(x_latent)
        
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent):
        model = self.vqgan
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out