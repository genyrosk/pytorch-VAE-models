from .data_mnist import load_mnist
from .dense_model import VAE_Simple
from .conv_model import VAE_Conv
from .upsampled_model import VAE_Upsampled
from .super_resolution_model import VAE_SuperResolution

models = {
    'simple': VAE_Simple,
    'conv': VAE_Conv,
    'upsampled': VAE_Upsampled,
    'super_resolution': VAE_SuperResolution
}
