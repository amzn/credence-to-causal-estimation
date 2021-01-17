import numpy as np
import torch
import baseVAE
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

from typing import List, TypeVar

Tensor = TypeVar('torch.tensor')
Array = TypeVar('numpy.ndarray')


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.0000000001)
        m.bias.data.fill_(0.0)


class normalize(nn.Module):
    def __init__(self, target_idx=0):
        super().__init__()
        self.target_idx = target_idx

    def forward(self, input):
        try:
            shape = input.shape
            output = torch.zeros(shape, requires_grad=True)
            m = torch.mean(input[:, :, 0], axis=0)
            s = torch.mean(input[:, :, 0], axis=0)
            for i in range(shape[1]):
                output[:, i, :] = (input[:, i, :] - m[i])/s[i]
            return output
        except:
            return input


class AR_VAE(baseVAE.BaseVAE, LightningModule):

    def __init__(self,
                 X: Array,  # shape (T x B x N) (batch size, sequence length, dimensionality)
                 lag: int,  # Autoregressive lag
                 latent_dim: int,  # size of the latent dimension
                 normalize_idx: int = 0,  # normalize_index
                 lr: float = 0.005,  # learning rate
                 weight_decay: float = 0,  # weight decay
                 hidden_dims: List = None,  # list of hidden dimensions for encoder
                 **kwargs) -> None:
        super(AR_VAE, self).__init__()

        self.X = X
        self.lag = lag
        self.in_channels = X.shape[1]
        self.latent_dim = latent_dim

        self.lr = lr
        self.weight_decay = weight_decay

        self.curr_device = None
        self.hold_graph = False

        self.hparams = {'lr': self.lr}

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 16]
        # Normalize
        # net = nn.Sequential(normalize(normalize_idx))
        # modules.append(net)

        # Build
        in_channels = self.in_channels
        for h_dim in hidden_dims:
            net = nn.Sequential(nn.Linear(in_features=in_channels, out_features=h_dim), nn.ReLU())
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
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # can we take the final layer as input from user?
        self.final_layer = nn.Sequential(nn.Linear(in_features=hidden_dims[-1],
                                                   out_features=self.in_channels))

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
        #print(input.size())
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

    def make_peak(self, T: int, B: int,
                  T0: int = 190,
                  dT: int = 365) -> Tensor:
        peaks = torch.zeros((T, B, 1), requires_grad=False)
        while(T0 < T):
            peaks[T0, :, :] = 10
            T0 = T0+dT
        return peaks

    def make_trend(self, T: int, B: int,
                   slope: float = 0.005,
                   ) -> Tensor:
        a = torch.linspace(0, slope*T, T)
        b = torch.zeros(B, 1)
        return (b+a).T.reshape(T, B, 1)

    def make_seasonality(self, T: int, B: int, period: int = 365,) -> Tensor:
        a = torch.linspace(0, T, T)
        season = torch.sin(a*torch.acos(torch.zeros(1)).item() * 4/period)
        b = torch.zeros(B, 1)
        return (b+season).T.reshape(T, B, 1)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param Z: (Tensor) [T-1 x B x L]
        :param S_init: (Tensor) [lag x B x P]
        :return S: (Tensor) [T x B x P]
        UPDATE THIS AND LOOP OVER ALL TIME STEPS
        """
        T = 100
        B = 100
        

        peaks = self.make_peak(T, B)
        trend = self.make_trend(T, B, 0.005)
        seasonality_y = self.make_seasonality(T, B, 365)
        seasonality_m = self.make_seasonality(T, B, 30)
        w1 = torch.cat((peaks, trend, seasonality_y, seasonality_m), axis=2)


            
        result = self.decoder_input(z)
        w = self.decoder(result)
        # concatenating w with addition trends (seasonality+peaks+linear)
        #w = torch.cat((w, w1[t, :, :]), axis=1)
        # concatenating lag terms
        s = self.final_layer(w)
        #S = torch.cat([S, s.reshape([1, s.shape[0], s.shape[1]])])
        return s

    def forward(self, X: Tensor, **kwargs) -> List[Tensor]:
        pi = self.encode(X)
        mu, logvar = pi
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        Z = self.reparameterize(pi, eps)
        mu, log_var = pi
        return [self.decode(Z), X, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
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

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        #print(recons.size())
        #print(input.size())
        recons_loss = F.mse_loss(recons, input)
        #print(mu.size())
        kld_loss = torch.mean(torch.mean(0.5 * torch.sum(mu ** 2 + log_var.exp()-1 - log_var, dim=1), dim=0))
        
        loss = recons_loss + kld_weight * kld_loss
        #loss = recons_loss + 0.001 * kld_loss
        #loss = recons_loss

        
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
        
        
    def sample(self,
               T: int,
               num_samples: int,
               ) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        S_init = torch.randn(self.lag, num_samples, self.X.shape[2])
        z = torch.randn(T-1, num_samples, self.latent_dim)
        samples = self.decode(z, S_init)
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        kld_loss = torch.stack([x['KLD'] for x in outputs]).mean()
        recons_loss = torch.stack([x['Reconstruction_Loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss, 'kld_loss': kld_loss, 'Reconstruction_Loss': recons_loss}
        print(tensorboard_logs)
        # self.sample_images()
        return {'val_loss': avg_loss, 'kld_loss': kld_loss, 'Reconstruction_Loss': recons_loss, 'log': tensorboard_logs}


    def val_dataloader(self):
        T = len(self.X)
        self.sample_dataloader = DataLoader(self.X, batch_size=T, shuffle=False, drop_last=False)
        self.num_val_imgs = len(self.sample_dataloader)
        return self.sample_dataloader

    def marginal(self, D, sample=100):
        T = D.shape[0]
        X = D.reshape((T, 1, -1))
        pi = self.encode(X)
        mu, logvar = pi

        B = X.shape[1]
        peaks = self.make_peak(T, B)
        trend = self.make_trend(T, B, 0.005)
        seasonality_y = self.make_seasonality(T, B, 365)
        seasonality_m = self.make_seasonality(T, B, 30)
        w1 = torch.cat((peaks, trend, seasonality_y, seasonality_m), axis=2)
        loglike = []
        for t in range(self.lag, T):
            Xhat = []
            for i in range(0, sample):
                z = torch.randn(B, self.latent_dim)
                result = self.decoder_input(z)
                w = self.decoder(result)
                # concatenating w with addition trends (seasonality+peaks+linear)
                #w = torch.cat((w, w1[t, :, :]), axis=1)
                # concatenating lag terms
                w = torch.cat([w] + [X[i, :, :] for i in range(t-self.lag, t)], dim=1)
                s = self.final_layer(w)
                Xhat.append(s)
            Xhat = torch.stack(Xhat)[:, 0, :]
            mu_Xhat = Xhat.mean(axis=0)
            cov_inv_Xhat = torch.tensor(np.linalg.inv(np.cov(Xhat.T.detach().numpy()))).float()
            loglike_t = -0.5*torch.matmul((D[t, :] - mu_Xhat).T, torch.matmul(cov_inv_Xhat, (D[t, :] - mu_Xhat)))
            loglike.append(loglike_t)
        return torch.sum(torch.tensor(loglike).float())

    def marginal_log_likelihood(self, X, samples=100):
        B = X.shape[1]
        loglike = []
        for i in range(B):
            loglike.append(self.marginal(X[:, i, :], sample=samples))
        return loglike
