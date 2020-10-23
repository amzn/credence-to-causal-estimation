import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model as lm
import torch
from train_arvae import generate_example_sample

from functools import reduce

sns.set()


get_leaves = lambda hierarchy: hierarchy if type(hierarchy) == list else \
    reduce(lambda x, y: x+y, map(get_leaves, hierarchy.values()), [])


def choose_donors(feat_df, target_df, k):
    model = lm.ElasticNet()
    model = model.fit(X=feat_df, y=target_df)
    top_k_donor_idx = model.coef_.argsort()[-k:][::-1]

    return feat_df.iloc[:, top_k_donor_idx]


def verify_causal_model(vae_model, post_intervention_vae_model, T0, input_df, target, donors, method='REMI',
                        number_samples=100):

    # #------------------------------------------------------------------------##
    # traversing hierarchy to get list leaves
    target_leaves = target
    donor_leaves = donors
    # getting just leaf nodes because the data simulator is trained to generate leaves
    target_df = input_df[input_df.columns.intersection(target_leaves)]
    donor_df = input_df[input_df.columns.intersection(donor_leaves)]
    # choosing top K donor leaves
    donor_df_k = choose_donors(donor_df, target_df.iloc[:, 0]+target_df.iloc[:, 1], k=len(donor_df.columns))
    # setting up input to neural network
    X = pd.concat([target_df, donor_df_k], axis=1, join="inner")
    leaves_k = list(X.columns)
    X = X.to_numpy()
    X = torch.tensor((X.reshape(X.shape[0], 1, X.shape[1]) - np.mean(X))/np.std(X)).float()
    df_X = pd.DataFrame(X[:, 0, :].detach().numpy(), index=input_df.index, columns=leaves_k)
    df_X.plot(figsize=(20, 8), title='Input Data for AR-VAE').legend(bbox_to_anchor=(0.975, 0.50))

    # RUNNING VAE MODEL TO GENERATE SAMPLES
    S, Sprime, T0, pi, Z, W = generate_example_sample(X, vae_model,
                                                      post_intervention_vae_model=post_intervention_vae_model,
                                                      T0=T0)
    # Getting the parameter around which we want to sample
    # Sampling
    SamplesPrime = []
    Samples = []
    TrueCumulativeTE = []
    for i in range(number_samples):
        S, Sprime, T0, pi, Z, W = generate_example_sample(X, vae_model,
                                                          post_intervention_vae_model=post_intervention_vae_model,
                                                          T0=T0)
        mu, logvar = pi
        df_S_prime = pd.DataFrame(Sprime[:, 0, :].detach().numpy(), index=input_df.index, columns=leaves_k)
        df_S_prime['t'] = pd.to_datetime(list(input_df.index))

        df_S = pd.DataFrame(S[:, 0, :].detach().numpy(), index=input_df.index, columns=leaves_k)
        df_S['t'] = pd.to_datetime(list(input_df.index))

        SamplesPrime.append(df_S_prime)
        Samples.append(df_S)
        diff_df_S = df_S_prime - df_S
        diff_df_title = '(True) Causal Effect on Simulated Sample %d' % (i)
        diff_df_S.drop(columns=['t']).plot(figsize=(20, 8), title=diff_df_title).legend(bbox_to_anchor=(0.975, 0.50))
        TrueCumulativeTE.append(diff_df_S.sum(axis=0))
    # RUNNING CAUSAL ESTIMATION METHOD (Currently only coded for REMI)
    if method == 'REMI':
        from remi.bayesian_linear_forecaster import BayesianLinearForecaster
        EstCumulativeTE = []
        for df_S in Samples:
            print(df_S.columns)
            t0 = df_S_prime.index[T0]
            forecaster = BayesianLinearForecaster(df_S_prime, t0, hierarchy=None)
            forecaster.sample(2*365, use_advi=True, n_init=15000)

            # import pdb
            # pdb.set_trace()

            d = forecaster.fit_summary(only_aggregated_summary=False)
            d1 = d['total_true_value'] - d['total_predicted_value']
            EstCumulativeTE.append(d1)
            # Add some metric of comparison here!
            forecaster.plot_fit()

        return TrueCumulativeTE, EstCumulativeTE
