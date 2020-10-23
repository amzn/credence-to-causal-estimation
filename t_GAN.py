#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 20:12:40 2020

@author: harspari
"""

import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (seq_len, batch_size, in_dim)
    Output: sequence of shape (seq_len, batch_size, out_dim)
    """

    def __init__(self, in_dim, out_dim, n_layers=2, hidden_dim=32):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element.
    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs


def train(dataset_path, batchSize, L=2, lr=0.0005, manualSeed=None, outf='outf', imf='imf',
          delta_condition=False, epochs=100, alternate=True, delta_lambda=10, tensorboard_image_every=10,
          checkpoint_every=5):

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default="btp", help='dataset to use (only btp for now)')
    # parser.add_argument('--dataset_path', required=True, help='path to dataset')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    # parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    # parser.add_argument('--nz', type=int, default=100, help='dimensionality of the latent vector z')
    # parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    # parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    # parser.add_argument('--cuda', action='store_true', help='enables cuda')
    # parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    # parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    # parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
    # parser.add_argument('--imf', default='images', help='folder to save images')
    # parser.add_argument('--manualSeed', type=int, help='manual seed')
    # parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
    # parser.add_argument('--run_tag', default='', help='tags for the current run')
    # parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints')
    # parser.add_argument('--tensorboard_image_every', default=5, help='interval for displaying images on tensorboard')
    # parser.add_argument('--delta_condition', action='store_true', help='whether to use the mse loss for deltas')
    # parser.add_argument('--delta_lambda', type=int, default=10, help='weight for the delta condition')
    # parser.add_argument('--alternate', action='store_true', help='whether to alternate between adversarial and mse'
    #   'loss in generator')
    # parser.add_argument('--dis_type', default='cnn', choices=['cnn','lstm'], help='architecture to be used for  '
    #   'discriminator')
    # parser.add_argument('--gen_type', default='lstm', choices=['cnn','lstm'], help='architecture to be used for '
    #   'generator')
    # opt = parser.parse_args()

    # Create writer for tensorboard
    date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    run_name = 'tGAN'+str(date)
    log_dir_name = os.path.join(os.getcwd(), run_name)
    writer = SummaryWriter(log_dir_name)
    # writer.add_text('Options', str(opt), 0)
    # print(opt)

    try:
        os.makedirs(outf)
    except OSError:
        pass
    try:
        os.makedirs(imf)
    except OSError:
        pass

    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True

    dataset = torch.from_numpy(pd.read_csv(dataset_path, index_col=0).to_numpy()).float()
    dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], 1))
    print(dataset.shape)

    # assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nz = int(L)
    # Retrieve the sequence length as first dimension of a sequence in the dataset
    seq_len = dataset[0].size(0)
    # An additional input is needed for the delta
    in_dim = L + 1 if delta_condition else L

    netD = LSTMDiscriminator(in_dim=1, hidden_dim=32).to(device)
    netG = LSTMGenerator(in_dim=in_dim, out_dim=1, hidden_dim=32).to(device)

    assert netG
    assert netD

    # if opt.netG != '':
    #     netG.load_state_dict(torch.load(opt.netG))
    # if opt.netD != '':
    #     netD.load_state_dict(torch.load(opt.netD))

    print("|Discriminator Architecture|\n", netD)
    print("|Generator Architecture|\n", netG)

    criterion = nn.BCELoss().to(device)
    delta_criterion = nn.MSELoss().to(device)

    # Generate fixed noise to be used for visualization
    fixed_noise = torch.randn(batchSize, seq_len, nz, device=device)

    if delta_condition:
        # Sample both deltas and noise for visualization
        deltas = dataset.sample_deltas(batchSize).unsqueeze(2).repeat(1, seq_len, 1)
        fixed_noise = torch.cat((fixed_noise, deltas), dim=2)

    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr)
    optimizerG = optim.Adam(netG.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            niter = epoch * len(dataloader) + i

            # Save just first batch of real data for displaying
            if i == 0:
                real_display = data.cpu()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with real data
            netD.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            label = torch.full((batch_size, seq_len, 1), real_label, device=device)

            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake data
            noise = torch.randn(batch_size, seq_len, nz, device=device)
            if delta_condition:
                # Sample a delta for each batch and concatenate to the noise for each timestep
                deltas = dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
                noise = torch.cat((noise, deltas), dim=2)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Visualize discriminator gradients
            for name, param in netD.named_parameters():
                writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            if delta_condition:
                # If option is passed, alternate between the losses instead of using their sum
                if alternate:
                    optimizerG.step()
                    netG.zero_grad()
                noise = torch.randn(batch_size, seq_len, nz, device=device)
                deltas = dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
                noise = torch.cat((noise, deltas), dim=2)
                # Generate sequence given noise w/ deltas and deltas
                out_seqs = netG(noise)
                delta_loss = delta_lambda * delta_criterion(out_seqs[:, -1] - out_seqs[:, 0], deltas[:, 0])
                delta_loss.backward()

            optimizerG.step()

            # Visualize generator gradients
            for name, param in netG.named_parameters():
                writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)

            ###########################
            # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
            ###########################

            # Report metrics
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
            if delta_condition:
                writer.add_scalar('MSE of deltas of generated sequences', delta_loss.item(), niter)
                print(' DeltaMSE: %.4f' % (delta_loss.item()/delta_lambda), end='')
            print()
            writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
            writer.add_scalar('GeneratorLoss', errG.item(), niter)
            writer.add_scalar('D of X', D_x, niter)
            writer.add_scalar('D of G of z', D_G_z1, niter)

        # End of the epoch #####
        real_plot = time_series_to_plot(dataset.denormalize(real_display))
        if (epoch % tensorboard_image_every == 0) or (epoch == (epochs - 1)):
            writer.add_image("Real", real_plot, epoch)

        fake = netG(fixed_noise)
        fake_plot = time_series_to_plot(dataset.denormalize(fake))
        torchvision.utils.save_image(fake_plot, os.path.join(imf, 'tGAN_epoch'+str(epoch)+'.jpg'))
        if (epoch % tensorboard_image_every == 0) or (epoch == (epochs - 1)):
            writer.add_image("Fake", fake_plot, epoch)

        # Checkpoint
        if (epoch % checkpoint_every == 0) or (epoch == (epochs - 1)):
            torch.save(netG, '%s/%s_netG_epoch_%d.pth' % (outf, 'tGAN', epoch))
            torch.save(netD, '%s/%s_netD_epoch_%d.pth' % (outf, 'tGAN', epoch))
    return netG, netD


def time_series_to_plot(time_series_batch, dpi=35, feature_idx=0, n_images_per_row=4, titles=None):
    """Convert a batch of time series to a tensor with a grid of their plots

    Args:
        time_series_batch (Tensor): (batch_size, seq_len, dim) tensor of time series
        dpi (int): dpi of a single image
        feature_idx (int): index of the feature that goes in the plots (the first one by default)
        n_images_per_row (int): number of images per row in the plot
        titles (list of strings): list of titles for the plots
    Output:
        single (channels, width, height)-shaped tensor representing an image
    """
    # Iterates over the time series
    images = []
    for i, series in enumerate(time_series_batch.detach()):
        fig = plt.figure(dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
        if titles:
            ax.set_title(titles[i])
        ax.plot(series[:, feature_idx].numpy())  # plots a single feature of the time series
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(data)
        plt.close(fig)

    # Swap channel
    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    # Make grid
    grid_image = vutils.make_grid(images.detach(), nrow=n_images_per_row)
    return grid_image


def tensor_to_string_list(tensor):
    """Convert a tensor to a list of strings representing its value"""
    scalar_list = tensor.squeeze().numpy().tolist()
    return ["%.5f" % scalar for scalar in scalar_list]

# class DatasetGenerator:
#     def __init__(self, generator, seq_len=96, noise_dim=100, dataset=None):
#         """Class for fake dataset generation
#         Args:
#             generator (pytorch module): trained generator to use
#             seq_len (int): length of the sequences to be generated
#             noise_dim (int): input noise dimension for gan generator
#             dataset (Dataset): dataset providing normalize and denormalize functions for deltas and series
#               (by default, don't normalize)
#         """
#         self.generator = generator
#         self.seq_len = seq_len
#         self.noise_dim = noise_dim
#         self.dataset = dataset

#     def generate_dataset(self, outfile=None, batch_size=4, delta_list=None, size=1000):
#         """Method for generating a dataset
#         Args:
#             outfile (string): name of the npy file to save the dataset. If None, it is simply returned as pytorch
#               tensor
#             batch_size (int): batch size for generation
#             seq_len (int): sequence length of the sequences to be generated
#             delta_list (list): list of deltas to be used in the case of conditional generation
#             size (int): number of time series to generate if delta_list is present, this parameter is ignored
#         """
#         #If conditional generation is required, then input for generator must contain deltas
#         if delta_list:
#             noise = torch.randn(len(delta_list), self.seq_len, self.noise_dim)
#             deltas = torch.FloatTensor(delta_list).view(-1, 1, 1).repeat(1, self.seq_len, 1)
#             if self.dataset:
#                 #Deltas are provided in original range, normalization required
#                 deltas = self.dataset.normalize_deltas(deltas)
#             noise = torch.cat((noise, deltas), dim=2)
#         else:
#             noise = torch.randn(size, self.seq_len, self.noise_dim)

#         out_list = []
#         for batch in noise.split(batch_size):
#             out_list.append(self.generator(batch))
#         out_tensor = torch.cat(out_list, dim=0)

#         #Puts generated sequences in original range
#         if self.dataset:
#             out_tensor = self.dataset.denormalize(out_tensor)

#         if outfile:
#             np.save(outfile, out_tensor.detach().numpy())
#         else:
#             return out_tensor
