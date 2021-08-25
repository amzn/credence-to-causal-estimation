import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

# conVAE trains a generator which can generate Y | X
# the object takes in the input the list of the column names of Y and columns names of X

class conVAE(pl.LightningModule):
    def __init__(
        self,
        df,
        Ynames,  # random variable to be generated / inputted
        Xnames=[],  # random variable to condition on
        cat_cols=[],  # categorical columns
        var_bounds={},  # variable lower and upper bounds
        latent_dim=2,  # latent space dimensions
        hidden_dim=[16],  # perceptrons in each layer
        batch_size=10,  # batch size
        potential_outcome=False,  # indicator if the generated sample has both potential outcomes
        treatment_cols=["T"],  # treatment indicator column
        treatment_effect_fn=lambda x: 0,  # treatment effect function defined by the user
        selection_bias_fn=lambda x, t: 0,  # selection bias function defined by the user
        effect_rigidity=1e20,  # strength of treatment effect constraint
        bias_rigidity=1e20,  # strength of selection bias constraint
        kld_rigidity=0.1,  # strength of KL divergence loss
    ):
        super().__init__()
        
        # initializing internal variables/objects
        self.in_dim = len(Ynames)  # input dimension
        self.con_dim = len(Xnames)  # dimension of variable to condition on
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cat_cols = cat_cols
        self.encoder_dims = [self.in_dim] + self.hidden_dim
        self.decoder_dims = [self.latent_dim + self.con_dim] + self.hidden_dim[::-1]
        self.batch_size = batch_size
        self.potential_outcome = potential_outcome
        self.kld_rigidity = kld_rigidity
        
        # indices of the categorical variable in Y
        self.cat_col_idx = [i for i in range(self.in_dim) if Ynames[i] in cat_cols]
        
        # if output has all potential outcomes then add corresponding categorical variable indices
        if self.potential_outcome:
            self.cat_col_idx += [
                self.in_dim + i for i in range(self.in_dim) if Ynames[i] in cat_cols
            ]
        
        # store the variable bound per index
        self.var_bounds = {
            Ynames.index(k): v for k, v in var_bounds.items() if k in Ynames
        }
        
        # if generating potential outcomes then add constraints for user defined treatment effects and selection bias function
        if self.potential_outcome:
            self.T_col = [i for i in range(self.con_dim) if Xnames[i] in treatment_cols]
            self.X_col = [
                i for i in range(self.con_dim) if Xnames[i] not in treatment_cols
            ]

            self.alpha = effect_rigidity
            self.f = treatment_effect_fn

            self.beta = bias_rigidity
            self.g = selection_bias_fn
        
        # loading the input data to tensor dataset from dataframe
        self.data = TensorDataset(
            torch.from_numpy(df[Xnames].values.astype(float)).float(),
            torch.from_numpy(df[Ynames].values.astype(float)).float(),
        )
        
        # splitting the data into training and validation
        training_split = 0.8
        self.train_data, self.val_data = random_split(
            self.data,
            [ int(df.shape[0] * training_split), (df.shape[0]) - int((df.shape[0] * training_split)) ],
        )
        
        # training and validation dataset loader
        self.train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True
        )

        # Encoder layers
        self.encoder_module = []
        for layer in range(len(hidden_dim)):
            self.encoder_module.append(
                nn.Sequential(
                    nn.Linear(self.encoder_dims[layer], self.encoder_dims[layer + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*self.encoder_module)
        
        # embedding layers
        self.en_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.en_logvar = nn.Linear(self.hidden_dim[-1], self.latent_dim)

        # Decoder layers
        self.decoder_module = []
        for layer in range(len(hidden_dim)):
            self.decoder_module.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[layer], self.decoder_dims[layer + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*self.decoder_module)
        if self.potential_outcome:
            self.decode_Y = nn.Linear(self.hidden_dim[0], 2 * self.in_dim)
        else:
            self.decode_Y = nn.Linear(self.hidden_dim[0], self.in_dim)

    def forward(self, y):
        # embedding into latent space
        l = self.encoder(y)
        
        # mean and logvariance of the projected point in latent space
        mu = self.en_mu(l)
        logvar = self.en_logvar(l)
        
        return (mu, logvar)

    def sample(self, pi, x):
        # position in latent space
        mu, logvar = pi 
        
        # generating a standard normal random variable
        e = torch.randn(mu.shape).to(self.device)
        std = torch.exp(0.5 * logvar)
        
        #reparameterization
        z = mu + std * e
        
        # conditioning on x
        z_ = torch.cat((z, x), 1)
        
        # projecting out to Y space
        w = self.decoder(z_)
        y_hat = self.decode_Y(w)
        
        # binarizing discrete vars
        y_hat[:, self.cat_col_idx] = (y_hat[:, self.cat_col_idx] > 0.5).float()
        
        return y_hat

    def loss_fn(self, yhat, y, mu, logvar, **kwargs):
        if self.potential_outcome:
            # potential outcome specific arguments
            T = kwargs["T"]
            X = kwargs["X"]
            yhat_prime = kwargs["y_prime"]

            # observed potential outcome
            yhat_obs = T * yhat[:, self.in_dim :] + (1 - T) * yhat[:, : self.in_dim]

            # reconstruction loss
            recons = F.mse_loss(yhat_obs, y)

            # treatment effect constraint
            constraint_effect = torch.sum(
                torch.square(yhat[:, 1] - yhat[:, 0] - self.f(X))
            )

            # selection bias constraint
            constraint_bias = torch.sum(
                torch.square(
                    T * (yhat[:, 0] - yhat_prime[:, 0])
                    + (1 - T) * (yhat_prime[:, 1] - yhat[:, 1])
                    - self.g(X, T)
                )
            )

            # adding to the loss calculation
            recons = (
                recons + self.alpha * constraint_effect + self.beta * constraint_bias
            )
        else:
            # reconstruction loss
            recons = F.mse_loss(yhat, y)
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        recons = recons + self.kld_rigidity * kld

        # variable bounds
        var_bounds_constraint = torch.tensor(0).float().to(self.device)
        for k, v in self.var_bounds.items():
            if "lower" in v:
                lower_lim = v["lower"]
                var_bounds_constraint += torch.max((yhat[:, k] < lower_lim).float()).to(
                    self.device
                )
                if self.potential_outcome:
                    var_bounds_constraint += torch.max(
                        (yhat[:, k + self.in_dim] < lower_lim).float()
                    ).to(self.device)

            if "upper" in v:
                upper_lim = v["upper"]
                var_bounds_constraint += torch.max((yhat[:, k] > upper_lim).float()).to(
                    self.device
                )
                if self.potential_outcome:
                    var_bounds_constraint += torch.max(
                        (yhat[:, k + self.in_dim] > upper_lim).float()
                    ).to(self.device)

        recons = (
            recons + torch.tensor(float("1e20")).to(self.device) * var_bounds_constraint
        )
        return recons

    def configure_optimizers(self):
        # initializing optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # accessing training data
        x, y = train_batch
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        
        # projecting in latent space
        l = self.encoder(y)
        mu = self.en_mu(l)
        logvar = self.en_logvar(l)
        
        # reparameterization
        e = torch.randn(mu.shape).to(self.device)
        std = torch.exp(0.5 * logvar)
        z = mu + std * e
        
        # conditioning on x and projecting z out to y space
        z_ = torch.cat((z, x), 1)
        w = self.decoder(z_)
        y_hat = self.decode_Y(w)
        
        # if potential outcome, generate E[ Y(z) | X, z] and E[ Y(z) | X, 1-z]
        if self.potential_outcome:
            x_prime = x.clone()
            x_prime[:, self.T_col] = 1 - x_prime[:, self.T_col]
            z_prime_ = torch.cat((z, x_prime), 1)
            w_prime = self.decoder(z_prime_)
            y_hat_prime = self.decode_Y(w_prime)
            loss = self.loss_fn(
                y_hat,
                y,
                mu,
                logvar,
                X=x[:, self.X_col],
                T=x[:, self.T_col],
                y_prime=y_hat_prime,
            )

        else:
            loss = self.loss_fn(y_hat, y, mu, logvar) 
            
        # returning training loss and log it
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # function identical to training_step but with validation data loader
        x, y = val_batch
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        l = self.encoder(y)
        mu = self.en_mu(l)
        logvar = self.en_logvar(l)

        e = torch.randn(mu.shape).to(self.device)
        std = torch.exp(0.5 * logvar)
        z = mu + std * e
        z_ = torch.cat((z, x), 1)
        w = self.decoder(z_)
        y_hat = self.decode_Y(w)

        if self.potential_outcome:
            x_prime = x.clone()
            x_prime[:, self.T_col] = 1 - x_prime[:, self.T_col]
            z_prime_ = torch.cat((z, x_prime), 1)
            w_prime = self.decoder(z_prime_)
            y_hat_prime = self.decode_Y(w_prime)
            loss = self.loss_fn(
                y_hat,
                y,
                mu,
                logvar,
                X=x[:, self.X_col],
                T=x[:, self.T_col],
                y_prime=y_hat_prime,
            )

        else:
            loss = self.loss_fn(y_hat, y, mu, logvar)  # F.mse_loss(y_hat, y)

        self.log("val_loss", loss)

    def fit_model(self, gpus=1, precision=16, limit_train_batches=0.5):
        trainer = pl.Trainer(
            gpus=gpus, precision=precision, limit_train_batches=limit_train_batches
        )
        trainer.fit(self, self.train_loader, self.val_loader)
        return trainer
