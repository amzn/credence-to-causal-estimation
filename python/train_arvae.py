import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import t_VAE
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

<<<<<<< HEAD
=======
from typing import Optional
>>>>>>> 588be1c43e29e166a4aaaf71ed881cbdb197d412

sns.set()


def train(data, hyper_params, input_checkpoint_path=None, output_checkpoint_path='ar_vae.ckpt'):
    # TRAINING FUNCTION
    max_epochs = hyper_params['epochs']
    lag = hyper_params['lag']
    latent_dim = hyper_params['latent_dim']
    hidden_dims = hyper_params['hidden_dims']
    kld_weight = hyper_params['kld_weight']
    
    vae_model = t_VAE.AR_VAE(lag=lag,
                             latent_dim=latent_dim,
                             X=torch.tensor(data).float(),
                             hidden_dims=hidden_dims,
                             kld_weight=kld_weight,
                             ).float()
    if input_checkpoint_path is not None:
        vae_model = t_VAE.AR_VAE.load_from_checkpoint(input_checkpoint_path,
                                                      lag=lag,
                                                      latent_dim=latent_dim,
                                                      X=torch.tensor(data).float(),
                                                      hidden_dims=hidden_dims,
                                                      kld_weight=kld_weight,
                                                      ).float()

    print('Loss Before Training')
    res = vae_model.forward(torch.tensor(data).float())
    print(vae_model.loss_function(*res, M_N=1))

    # Logger
    tt_logger = TestTubeLogger(
        save_dir=os.getcwd(),
        name='t_VAE_log',
        debug=False,
        create_git_tag=False)

    # Trainer
    runner = Trainer(max_epochs=max_epochs,
                     logger=tt_logger,
                     log_every_n_steps=50,
                     limit_train_batches=2.,
                     limit_val_batches=3.,
                     num_sanity_val_steps=100,
                     checkpoint_callback=False
                     )

    runner.fit(vae_model)

    runner.save_checkpoint(output_checkpoint_path)

    print('Loss After Training')
    res = vae_model.forward(torch.tensor(data).float())
    print(vae_model.loss_function(*res, M_N=1))

    return vae_model, runner


# Function for Fetching the Interpretable Transformation Map
def fetch_ITM(vae_model):
    weights = None
    bias = None
    i = 0
    for p in vae_model.final_layer.parameters():
        if i == 0:
            weights = p
        else:
            bias = p
        i += 1
    return weights[:, :], bias[:]


# Function to encode Interventions
def intervene_raw(target_idx, feature_idx, bias, intervention, checkpoint_path, hyper_params, data):
    lag = hyper_params['lag']
    latent_dim = hyper_params['latent_dim']
    hidden_dims = hyper_params['hidden_dims']
    vae_model_intv = t_VAE.AR_VAE.load_from_checkpoint(checkpoint_path,
                                                       lag=lag,
                                                       latent_dim=latent_dim,
                                                       hidden_dims=hidden_dims,
                                                       X=torch.tensor(data).float(),
                                                       ).float()
    w_i, b_i = fetch_ITM(vae_model_intv)
    print(w_i.shape)
    print(w_i[target_idx, :][:, feature_idx])
    if bias:
        intv = intervention((w_i[target_idx, :][:, feature_idx], b_i[target_idx]))
    else:
        intv = intervention(w_i[target_idx, :][:, feature_idx])
    if bias:
        b_i[target_idx] = intv[1]
        for i in range(len(target_idx)):
            for j in range(len(feature_idx)):
                w_i[target_idx[i], feature_idx[j]] = intv[0][i, j]
        print((w_i[target_idx, :][:, feature_idx], b_i[target_idx]))
    else:
        for i in range(len(target_idx)):
            for j in range(len(feature_idx)):
                w_i[target_idx[i], feature_idx[j]] = intv[0][i, j]
        print(w_i[target_idx, :][:, feature_idx])
    return vae_model_intv


# Function to generate example samples
def generate_example_sample(data, vae_model, post_intervention_vae_model=None, T0=None, eps=None):
    # DATA -> PI -> Z -> W -> S(prime) ###
    pi = vae_model.encode(torch.tensor(data).float())  # calculate vector in sampling space; pi = (mean, log variance)
    #print(pi[0][0])
    if eps is None:
        mu, logvar = pi
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
    Z = vae_model.reparameterize(pi, eps)  # drawing sample from the sampling space
    #print(Z.shape)
    S_init = torch.tensor(data[0:vae_model.lag, :, :]).float()  # initializing lag terms

    T = Z.shape[0]  # length of timeseries
    B = Z.shape[1]  # number of bundles
    S = S_init

    # Hardcoding few dimensions in latent space
    peaks = vae_model.make_peak(T, B)
    trend = vae_model.make_trend(T, B, 0.005)
    seasonality_y = vae_model.make_seasonality(T, B, 365)
    seasonality_m = vae_model.make_seasonality(T, B, 30)
    w1 = torch.cat((peaks, trend, seasonality_y, seasonality_m), axis=2)

    # LOOPING OVER EACH STEP
    for t in range(T):
        # z = torch.cat([Z[t]] + [S[-i] for i in range(vae_model.lag)], dim=1)
        z = Z[t]
        result = vae_model.decoder_input(z)  # calculating input to decoder
        #print(result.shape)
        w = vae_model.decoder(result)  # calculating latent vector
        #print(w.shape)
        # concat latent vector with addition trends (seasonality+peaks+linear)
        w_appended = torch.cat((w, w1[t, :, :]), axis=1)
        w_lag = torch.cat([w] + [S[-i] for i in range(vae_model.lag)], dim=1)
        #print(w_lag.shape)
        # POST INTERVENTION PART
        if((post_intervention_vae_model is not None) and (T0 is not None)):
            if t >= (T0-vae_model.lag):
                if t == (T0-vae_model.lag):
                    Sprime = S.clone()  # copy the pre-intervention model
                sprime = post_intervention_vae_model.final_layer(w_lag)  # calculate the outcome
                # collect the outcome
                Sprime = torch.cat([Sprime, sprime.reshape([1, sprime.shape[0], sprime.shape[1]])])

        # PRE INTERVENTION PART/ COUNTERFACTUAL
        s = vae_model.final_layer(w_lag)
        S = torch.cat([S, s.reshape([1, s.shape[0], s.shape[1]])])

        # COLLECTING LATENT VECTORS
        if t == 0:
            W = w.reshape([1, w.shape[0], w.shape[1]])
        else:
            W = torch.cat([W, w.reshape([1, w.shape[0], w.shape[1]])])
    #print(W.shape)
    if((post_intervention_vae_model is not None) and (T0 is not None)):
        return S, Sprime, T0, pi, Z, W
    return S, pi, Z, W


# Function to plot data+samples
def plot(data, vae_model, post_intervention_vae_model=None, T0=None, out_folder=None):
    sns.set(font_scale=1.25)

    if((post_intervention_vae_model is not None) and (T0 is not None)):
        S, Sprime, T0, pi, Z, W = generate_example_sample(data, vae_model,
                                                          post_intervention_vae_model=post_intervention_vae_model,
                                                          T0=T0)
    else:
        S, pi, Z, W = generate_example_sample(data, vae_model)
    mu, logvar = pi
    T = mu.shape[0]
    for i in range(data.shape[1]):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))

        ax[0, 0].plot(np.arange(vae_model.lag + 5, T+vae_model.lag), (S).detach().numpy()[vae_model.lag + 5:, i, :], alpha=0.5)
        ax[0, 0].set_ylabel('Normalized (Outcome)')
        ax[0, 0].set_xlabel('Time Steps')
        ax[0, 0].title.set_text('Simulated X')

        ax[0, 1].plot(np.arange(vae_model.lag + 5, T+vae_model.lag), data[vae_model.lag + 5:, i, :])
        ax[0, 1].set_ylabel('Normalized (Outcome)')
        ax[0, 1].set_xlabel('Time Steps')
        ax[0, 1].title.set_text('Observed X')

        ax[1, 0].plot(np.arange(vae_model.lag, vae_model.lag+T), (mu).detach().numpy()[:, i, :])
        ax[1, 0].set_ylabel('Latent Value')
        ax[1, 0].set_xlabel('Time Steps')
        ax[1, 0].title.set_text('Mean Z')

        ax[1, 1].plot(np.arange(vae_model.lag, vae_model.lag+T), (logvar).detach().numpy()[:, i, :])
        ax[1, 1].set_ylabel('Latent Value')
        ax[1, 1].set_xlabel('Time Steps')
        ax[1, 1].title.set_text('Log-Std Z')

        ax[2, 0].plot(np.arange(0, T+vae_model.lag),
                      (S).detach().numpy()[:, i, :]-(S).detach().numpy()[:, i, :], alpha=0.5)
        ax[2, 0].set_ylabel('Normalized (Outcome)')
        ax[2, 0].set_xlabel('Time Steps')
        ax[2, 0].title.set_text('Difference: X(Intervened) - X')

        ax[2, 1].plot(np.arange(vae_model.lag, T+vae_model.lag), (W).detach().numpy()[:, i, :])
        ax[2, 1].set_ylabel('Outcome (Penultimate Layer)')
        ax[2, 1].set_xlabel('Time Steps')
        ax[2, 1].title.set_text('Simulated W')

        if post_intervention_vae_model is not None:
            if T0 is not None:
                ax[0, 0].set_prop_cycle(None)
                ax[0, 0].plot(np.arange(0, T+post_intervention_vae_model.lag),
                              (Sprime).detach().numpy()[:, i, :])
                ax[0, 0].axvline(x=T0)
                ax[0, 0].set_prop_cycle(None)
                ax[2, 0].plot(np.arange(0, T+post_intervention_vae_model.lag),
                              (Sprime).detach().numpy()[:, i, :]-(S).detach().numpy()[:, i, :])
                ax[2, 0].axvline(x=T0)

        if out_folder is not None:
<<<<<<< HEAD
            fig.savefig(out_folder+'example_sample_%d.png' % (i))
=======
            fig.savefig(out_folder + "example_sample_%d.png" % (i))


# Exclude Bundles with small targets
data = np.delete(data, np.argwhere(median_targets[:, :2] < 200)[:, 0], 1)
mean_targets = np.delete(
    mean_targets, np.argwhere(median_targets[:, :2] < 200)[:, 0], 0
)


class DataProcesser:
    """
    """

    def __init__(self, data: np.ndarray, targets: int, adjust: Optional[float] = 10):
        B = data.shape[1]  # total number of bundles
        T = data.shape[0]  # length timeseries
        N = data.shape[2]  # total number of donors + targets

        self.B = B
        self.T = T
        self.N = N
        self.targets = targets
        self.adjust = adjust

        def prepare_input(self, data: np.ndarray):
            data_norm0 = self.normalized_data(data=data)

            # remove outlier
            data_norm0[data_norm0 > 5] = 5
            data_norm0[data_norm0 < -5] = -5

            # convert 3D to 2D
            data_norm, std_t, std_d = convert_to_2d(self, data_norm0)

            # append mean and std
            mean_targets = np.mean(data[:, :, :], axis=0)

            processed_data_norm = np.append(
                np.append(
                    np.append(data_norm, np.log(std_t[:, None]) / adjust, axis=1),
                    np.log(std_d[:, None]) / adjust,
                    axis=1,
                ),
                np.log(mean_targets) / adjust,
                axis=1,
            )
            return processed_data_norm

        def normalized_data(self, data: np.ndarray):
            N = self.N
            targets = self.targets

            # Separetly compute standard deviation for largest target and largest donor
            std_t = np.std(data[:, :, targets - 1], axis=0)
            std_d = np.std(data[:, :, N - 1], axis=0)

            data_norm0t = (
                data[:, :, :targets] - np.mean(data[:, :, :targets], axis=0)
            ) / std_t[None, :, None]
            data_norm0d = (
                data[:, :, targets:] - np.mean(data[:, :, targets:], axis=0)
            ) / std_d[None, :, None]
            data_norm0 = np.append(data_norm0t, data_norm0d, axis=2)

            print(f"Shape: {data_norm0.shape}")
            print(f"\nMean of raw data: {np.mean(data):.3f}")
            print(f"Std. dev of raw data: {np.std(data):.3f}")
            print(f"\nMean of normalized data: {np.mean(data_norm0):.3f}")
            print(f"Std. dev of normalized data: {np.std(data_norm0):.3f}")
            return data_norm0, std_t, std_d

        def convert_to_2d(self, data_input):
            B = self.B
            N = self.N
            T = self.T

            data_rescale = np.zeros((B, N * T))
            for i in range(0, B):
                for j in range(0, N):
                    data_rescale[i, T * j : T * (j + 1)] = data_input[:, i, j]
            return data_rescale

>>>>>>> 588be1c43e29e166a4aaaf71ed881cbdb197d412
