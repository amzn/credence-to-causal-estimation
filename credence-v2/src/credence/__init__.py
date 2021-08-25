import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import tqdm
import pytorch_lightning.callbacks.progress as pb

import autoencoder


class Credence:
    def __init__(
        self,
        data,  # dataframe
        post_treatment_var,  # list of post treatment variables
        treatment_var,  # list of treatment variable(s)
        categorical_var,  # list of variables which are categorical
        numerical_var,  # list of variables which are numerical
        var_bounds={},  # dictionary of bounds if certain variable is bounded
    ):
        self.data_raw = data
        self.Ynames = post_treatment_var
        self.Tnames = treatment_var

        self.categorical_var = categorical_var
        self.numerical_var = numerical_var

        self.var_bounds = var_bounds

        # preprocess data
        self.data_processed = self.preprocess(
            self.data_raw,
            self.Ynames,
            self.Tnames,
            self.categorical_var,
            self.numerical_var,
        )

        self.Xnames = [
            x for x in self.data_processed.columns if x not in self.Ynames + self.Tnames
        ]

    # train generator
    def fit(
        self,
        latent_dim=2,
        hidden_dim=[16],
        batch_size=10,
        treatment_effect_fn=lambda x: 0,
        selection_bias_fn=lambda x, t: 0,
        effect_rigidity=1e20,
        bias_rigidity=1e20,
        kld_rigidity=0.1,
        max_epochs=100,
    ):

        # generator for T
        self.m_treat = autoencoder.conVAE(
            df=self.data_processed,
            Xnames=[],
            Ynames=self.Tnames,
            cat_cols=self.categorical_var,
            var_bounds=self.var_bounds,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            kld_rigidity=kld_rigidity,
        )  # .to('cuda:0')
        bar = pb.ProgressBar()
        self.trainer_treat = pl.Trainer(
            gpus=1,
            precision=16,
            limit_train_batches=0.5,
            max_epochs=max_epochs,
            callbacks=[bar],
        )
        self.trainer_treat.fit(
            self.m_treat, self.m_treat.train_loader, self.m_treat.val_loader
        )

        # generator for X | T
        self.m_pre = autoencoder.conVAE(
            df=self.data_processed,
            Xnames=self.Tnames,
            Ynames=self.Xnames,
            cat_cols=self.categorical_var,
            var_bounds=self.var_bounds,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            kld_rigidity=kld_rigidity,
        )  # .to('cuda:0')

        bar = pb.ProgressBar()
        self.trainer_pre = pl.Trainer(
            gpus=1,
            precision=16,
            limit_train_batches=0.5,
            max_epochs=max_epochs,
            callbacks=[bar],
        )
        self.trainer_pre.fit(self.m_pre, self.m_pre.train_loader, self.m_pre.val_loader)

        # generator for Y(1),Y(0) | X, T
        self.m_post = autoencoder.conVAE(
            df=self.data_processed,
            Xnames=self.Xnames + self.Tnames,
            Ynames=self.Ynames,
            cat_cols=self.categorical_var,
            var_bounds=self.var_bounds,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            potential_outcome=True,
            treatment_cols=self.Tnames,
            treatment_effect_fn=treatment_effect_fn,
            selection_bias_fn=selection_bias_fn,
            effect_rigidity=effect_rigidity,
            bias_rigidity=bias_rigidity,
            kld_rigidity=kld_rigidity,
        )  # .to('cuda:0')

        bar = pb.ProgressBar()
        self.trainer_post = pl.Trainer(
            gpus=1,
            precision=16,
            limit_train_batches=0.5,
            max_epochs=max_epochs,
            callbacks=[bar],
        )
        self.trainer_post.fit(
            self.m_post, self.m_post.train_loader, self.m_post.val_loader
        )
        
        # returning trained generators
        return [self.m_treat, self.m_pre, self.m_post]
    
    # sample from generator
    def sample(self, num_samples=1000):
        # initializing latent variables from standard normal distribution
        pi_treat = (
            torch.zeros((num_samples, self.m_treat.latent_dim)),
            torch.zeros((num_samples, self.m_treat.latent_dim)),
        )
        pi_pre = (
            torch.zeros((num_samples, self.m_pre.latent_dim)),
            torch.zeros((num_samples, self.m_pre.latent_dim)),
        )
        pi_post = (
            torch.zeros((num_samples, self.m_post.latent_dim)),
            torch.zeros((num_samples, self.m_post.latent_dim)),
        )
        
        # sample from conVAE
        Tgen = self.m_treat.sample(pi=pi_treat, x=torch.empty(size=(num_samples, 0)))
        Xgen = self.m_pre.sample(pi=pi_pre, x=Tgen)
        Ygen = self.m_post.sample(pi=pi_post, x=torch.cat((Xgen, Tgen), 1))
        
        # wrapping in a dataframe
        df = pd.DataFrame(Xgen.detach().numpy(), columns=self.Xnames)
        df_T = pd.DataFrame(Tgen.detach().numpy(), columns=self.Tnames)
        df_Y = pd.DataFrame(Ygen.detach().numpy())
        df = df.join(df_T).join(df_Y)

        return df

    def preprocess(
        self, df, post_treatment_var, treatment_var, categorical_var, numerical_var
    ): # this function preprocesses the categorical variables from objects to numerics 

        # codifying categorical variables
        df_cat = (df[categorical_var]).astype("category")
        for col in categorical_var:
            df_cat[col] = df_cat[col].cat.codes

        # codifying numeric variables
        df_num = df[numerical_var]

        # joining columns
        df_ = df_cat.join(df_num)

        return df_
