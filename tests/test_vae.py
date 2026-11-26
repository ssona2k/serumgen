import torch

from serumgen.vae import VAE, vae_loss


def test_forward_shapes():
    model = VAE(input_dim=50, latent_dim=8)
    x = torch.rand(4, 50)
    recon, mu, logvar = model(x)
    assert recon.shape == (4, 50)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)


def test_decode_range():
    model = VAE(input_dim=50, latent_dim=8)
    out = model.decode(torch.randn(2, 8))
    assert out.min() >= 0.0 and out.max() <= 1.0  # Sigmoid output


def test_loss_is_scalar():
    model = VAE(input_dim=50, latent_dim=8)
    x = torch.rand(4, 50)
    recon, mu, logvar = model(x)
    loss, r, k = vae_loss(recon, x, mu, logvar, beta=0.5)
    assert loss.dim() == 0
