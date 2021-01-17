import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

import t_VAE
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

sns.set()


def train(data, hyper_params, input_checkpoint_path=None, output_checkpoint_path='ar_vae.ckpt', M_N=1):
    # TRAINING FUNCTION
    max_epochs = hyper_params['epochs']
    lag = hyper_params['lag']
    latent_dim = hyper_params['latent_dim']
    hidden_dims = hyper_params['hidden_dims']
    if 'if_normalize' in hyper_params:
        if_normalize = hyper_params['if_normalize']
    else:
        if_normalize = True #Default is True

    vae_model = t_VAE.AR_VAE(lag=lag,
                             latent_dim=latent_dim,
                             X=torch.tensor(data).float(),
                             hidden_dims=hidden_dims,
                             if_normalize=if_normalize,
                             ).float()
    if input_checkpoint_path is not None:
        vae_model = t_VAE.AR_VAE.load_from_checkpoint(input_checkpoint_path,
                                                      lag=lag,
                                                      latent_dim=latent_dim,
                                                      X=torch.tensor(data).float(),
                                                      hidden_dims=hidden_dims,
                                                      if_normalize=if_normalize,
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
#     runner = Trainer(max_epochs=max_epochs,
#                      logger=tt_logger,
#                      log_save_interval=50,
#                      train_percent_check=1.,
#                      val_percent_check=1.,
#                      num_sanity_val_steps=100,
#                      early_stop_callback=False)
    runner = Trainer(max_epochs=max_epochs,
                     logger=tt_logger,
                     log_every_n_steps=50,
                     limit_train_batches=1.,
                     limit_val_batches=1.,
                     num_sanity_val_steps=100,
                     checkpoint_callback=False
                     )

    runner.fit(vae_model)

    runner.save_checkpoint(output_checkpoint_path)

    print('Loss After Training')
    res = vae_model.forward(torch.tensor(data).float())
    print(vae_model.loss_function(*res, M_N=M_N))

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
    if eps is None:
        mu, logvar = pi
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
    Z = vae_model.reparameterize(pi, eps)  # drawing sample from the sampling space

    result = vae_model.decoder_input(Z)
    W = vae_model.decoder(result)
    S = vae_model.final_layer(W)

    return S, pi, Z, W


def reshape(s,T,B,N):
    s2 = np.zeros((T,B,N))
    if torch.is_tensor(s):
        s = s.detach().numpy()
    for i in range(0,B):
        for j in range(0,N):
            try:
                s2[:,i,j] = s[i,T*j:T*(j+1)] 
            except:
                s2[:,i,j] = s[0,i,T*j:T*(j+1)]
    return s2


# Function to plot data+samples
def plot(data, vae_model, N, post_intervention_vae_model=None, T0=None, out_folder=None):
    sns.set(font_scale=1.25)

    if((post_intervention_vae_model is not None) and (T0 is not None)):
        S, Sprime, T0, pi, Z, W = generate_example_sample(data, vae_model,
                                                          post_intervention_vae_model=post_intervention_vae_model,
                                                          T0=T0)
    else:
        S, pi, Z, W = generate_example_sample(data, vae_model)

    B = data.shape[0]
    T = int(data.shape[1]/N)
    S = reshape(s=S,T=T,B=B,N=N)
    data = reshape(s=data,T=T,B=B,N=N)

    mu, logvar = pi
    for i in range(data.shape[1]):
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(35, 20))

        ax[0, 0].plot(np.arange(0, T), (S)[:, i, :], alpha=0.5)
        ax[0, 0].set_ylabel('Normalized (Outcome)')
        ax[0, 0].set_xlabel('Time Steps')
        ax[0, 0].title.set_text('Simulated X')

        ax[0, 1].plot(np.arange(0, T), data[:, i, :])
        ax[0, 1].set_ylabel('Normalized (Outcome)')
        ax[0, 1].set_xlabel('Time Steps')
        ax[0, 1].title.set_text('Observed X')

        # ax[1, 0].plot(np.arange(0, mu.shape[0]), (mu).detach().numpy()[:, i, :])
        ax[1, 0].hist((mu).detach().numpy()[0, i, :])
        ax[1, 0].set_ylabel('Latent Value')
        ax[1, 0].set_xlabel('Value')
        ax[1, 0].title.set_text('Mean Z')

        # ax[1, 1].plot(np.arange(0,logvar.shape[0]), (logvar).detach().numpy()[:, i, :])
        ax[1, 1].hist((logvar).detach().numpy()[0, i, :])
        ax[1, 1].set_ylabel('Latent Value')
        ax[1, 1].set_xlabel('Value')
        ax[1, 1].title.set_text('Log-Std Z')

        ax[2, 0].plot(np.arange(0, T),
                      (S)[:, i, :]-(S)[:, i, :], alpha=0.5)
        ax[2, 0].set_ylabel('Normalized (Outcome)')
        ax[2, 0].set_xlabel('Time Steps')
        ax[2, 0].title.set_text('Difference: X(Intervened) - X')

        # ax[2, 1].plot(np.arange(0,W.shape[0]), (W).detach().numpy()[:, i, :])
        ax[2, 1].hist((W).detach().numpy()[0, i, :])
        ax[2, 1].set_ylabel('Outcome (Penultimate Layer)')
        ax[2, 1].set_xlabel('Value')
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
            fig.savefig(out_folder+'example_sample_%d.png' % (i))
