import numpy as np
import torch
import baseVAE
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

from typing import List, TypeVar

Tensor = TypeVar("torch.tensor")
Array = TypeVar("numpy.ndarray")


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.0000001)
        m.bias.data.fill_(0.0)


class AR_VAE(baseVAE.BaseVAE, LightningModule):
    def __init__(
        self,
        X: Array,  # shape (T x B x N) (batch size, sequence length, dimensionality)
        latent_dim: int,  # size of the latent dimension
        kld_weight: float = 1,  # weight for KL loss in loss
        lr: float = 0.005,  # learning rate
        weight_decay: float = 0,  # weight decay
        hidden_dims: List = None,  # list of hidden dimensions for encoder
        **kwargs
    ) -> None:
        super(AR_VAE, self).__init__()

        self.X = X
        self.in_channels = X.shape[1]
        self.latent_dim = latent_dim

        self.lr = lr
        self.kld_weight = kld_weight
        self.weight_decay = weight_decay

        self.curr_device = None
        self.hold_graph = False

        self.hparams = {"lr": self.lr}

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 16]

        # Build
        in_channels = self.in_channels
        for h_dim in hidden_dims:
            net = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=h_dim), nn.ReLU()
            )
            net.apply(init_weights)
            modules.append(net)
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # can we take the final layer as input from user?
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=hidden_dims[-1], out_features=self.in_channels)
        )

    def encode(self, X: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param X: (Tensor) timeseries of Tensors, shape = [T x B x P], B is the batch size, T is number of time steps,
            N is number of features
        :return: (Tensor) [B x T-lag x L] of latent codes for each timestep (mu_t, sigma_t)
        """
        T = X.shape[0]
        mu = []
        log_var = []

        input = torch.cat([X])
        result = self.encoder(input)
        mu.append(self.fc_mu(result))
        log_var.append(self.fc_var(result))

        mu = torch.stack(mu)
        log_var = torch.stack(log_var)
        return [mu, log_var]

    def reparameterize(self, pi: List[Tensor], eps) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0, 1).
        :param mu: (Tensor) Mean of the latent Gaussian [T-1 x L]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [T-1 x L]
        :return: (Tensor) [T-1 x L]

        UPDATE THIS AND LOOP OVER ALL TIME STEPS
        """
        mu, logvar = pi
        std = torch.exp(0.5 * logvar)
        return eps * std + mu

    def make_peak(self, T: int, B: int, T0: int = 190, dT: int = 365) -> Tensor:
        peaks = torch.zeros((T, B, 1), requires_grad=False)
        while T0 < T:
            peaks[T0, :, :] = 10
            T0 = T0 + dT
        return peaks

    def make_trend(self, T: int, B: int, slope: float = 0.005,) -> Tensor:
        a = torch.linspace(0, slope * T, T)
        b = torch.zeros(B, 1)
        return (b + a).T.reshape(T, B, 1)

    def make_seasonality(self, T: int, B: int, period: int = 365,) -> Tensor:
        a = torch.linspace(0, T, T)
        season = torch.sin(a * torch.acos(torch.zeros(1)).item() * 4 / period)
        b = torch.zeros(B, 1)
        return (b + season).T.reshape(T, B, 1)

    def decode(self, Z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param Z: (Tensor) [T-1 x B x L]
        :return S: (Tensor) [T x B x P]
        """
        result = self.decoder_input(Z)
        W = self.decoder(result)
        S = self.final_layer(W)
        return S

    def forward(self, X: Tensor, **kwargs) -> List[Tensor]:
        pi = self.encode(X)
        mu, logvar = pi
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        Z = self.reparameterize(pi, eps)
        mu, log_var = pi
        return [self.decode(Z), X, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the tVAE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs[
            "M_N"
        ]  # Account for the minibatch samples from the dataset, currently, M_N is always set to 1
        kld_weight = kld_weight * self.kld_weight

        recons = torch.squeeze(recons)
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            torch.mean(
                0.5 * torch.sum(mu ** 2 + log_var.exp() - 1 - log_var, dim=2), dim=1
            )
        )
        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD_Loss": kld_loss,
            "KLD_weight": kld_weight,
        }

    def sample(self, num_samples: int,) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = (torch.rand(1, num_samples, self.latent_dim) * 4) - 2
        samples = self.decode(z)
        samples = samples.squeeze(axis=0)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [T x P]
        :return: (Tensor) [T x P]
        """
        return self.forward(x)

    def configure_optimizers(self):
        self.optims = []
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.optims.append(optimizer)
        return self.optims

    def training_step(self, batch, batch_idx):
        X = batch
        self.curr_device = X.device
        results = self.forward(X)
        train_loss = self.loss_function(*results, M_N=1, optimizer_idx=0, batch_idx=0)
        return train_loss

    def train_dataloader(self):
        T = len(self.X)
        return DataLoader(self.X, batch_size=T, shuffle=False, drop_last=False)

    def validation_step(self, batch, optimizer_idx=0):
        X = batch
        self.curr_device = X.device
        results = self.forward(X)
        val_loss = self.loss_function(*results, M_N=1, optimizer_idx=optimizer_idx)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        kld_loss = torch.stack([x["KLD_Loss"] for x in outputs]).mean()
        recons_loss = torch.stack([x["Reconstruction_Loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "avg_val_loss": avg_loss,
            "kld_loss": kld_loss,
            "reconstruction_loss": recons_loss,
        }
        print(tensorboard_logs)
        return {
            "val_loss": avg_loss,
            "kld_loss": kld_loss,
            "reconstruction_loss": recons_loss,
            "log": tensorboard_logs,
        }

    def val_dataloader(self):
        T = len(self.X)
        self.sample_dataloader = DataLoader(
            self.X, batch_size=T, shuffle=False, drop_last=False
        )
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def marginal(self, D, sample=100):
        z = (torch.rand(1, sample, self.latent_dim) * 4) - 2
        s = self.decode(z)
        Xhat = s.squeeze(axis=0)
        mu_Xhat = Xhat.mean(axis=0)
        cov_inv_Xhat = torch.tensor(
            np.linalg.inv(np.cov(Xhat.T.detach().numpy()))
        ).float()
        loglike = -0.5 * torch.matmul(
            (D - mu_Xhat).T, torch.matmul(cov_inv_Xhat, (D - mu_Xhat))
        )
        return loglike

    def marginal_log_likelihood(self, X, samples=100):
        B = X.shape[0]
        loglike = []
        for i in range(B):
            loglike.append(self.marginal(X[i, :], sample=samples))
        return loglike
