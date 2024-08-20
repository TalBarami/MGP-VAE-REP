import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import argparse
import os
from itertools import cycle
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from logger import init_logger
from vid_process import resize_cropped, resize_keepAR, resize_mnist
from flags import *
from networks import Encoder, Decoder
from utils import weights_init, mse_loss, plot_image, plot_training_images
from covariance_fns import *
from flags import *
from setup_priors import *
from dataloader import *

logger = init_logger(ENCODER_SAVE, osp.join('logs', DATASET))


def KL_loss_L1_without_mean(sigma_p_inv, sigma_q, mu_q, det_p, det_q):
    # sigma_p_inv: (d, nlen, nlen), det_p: (d)
    # sigma_q: (batch_size, d, nlen, nlen), mu_q: (batch_size, d, nlen)

    l1 = torch.einsum('kij,mkji->mk', sigma_p_inv, sigma_q)      # tr(sigma_p_inv sigma_q)
    l2 = torch.einsum('mki,mki->mk', mu_q, torch.einsum('kij,mkj->mki', sigma_p_inv, mu_q))      # <mu_q, sigma_p_inv, mu_q>
    loss = torch.sum(l1 + l2 + torch.log(det_p) - torch.log(det_q), dim=1) # KL divergence b/w two Gaussian distri
    return loss

def KL_loss_L1(sigma_p_inv, sigma_q, mu_q, mu_p, det_p, det_q):
    # sigma_p_inv: (n_dim, n_frames, n_frames), det_p: (d)
    # sigma_q: (batch_size, n_dim, n_frames, n_frames), mu_q: (batch_size, d, nlen)

    l1 = torch.einsum('kij,mkji->mk', sigma_p_inv, sigma_q)      # tr(sigma_p_inv sigma_q)
    l2 = torch.einsum('mki,mki->mk', mu_p - mu_q, torch.einsum('kij,mkj->mki', sigma_p_inv, mu_p - mu_q))      # <mu_q, sigma_p_inv, mu_q>
    loss = torch.sum(l1 + l2 + torch.log(det_p) - torch.log(det_q), dim=1)
    return loss

if (__name__ == '__main__'):
    logger.info('Starting train.py')
    logger.info(f'Parameters: {FLAGS}')

    # model definition
    encoder = Encoder()
    encoder.apply(weights_init)

    decoder = Decoder()
    decoder.apply(weights_init)

    # load saved models if load_saved flag is true
    if LOAD_SAVED:
        encoder.load_state_dict(torch.load(os.path.join('outputs', 'checkpoints', DATASET, ENCODER_SAVE)))
        decoder.load_state_dict(torch.load(os.path.join('outputs', 'checkpoints', DATASET, DECODER_SAVE)))

    # loss definition
    mse_loss = nn.MSELoss(reduction='sum')

    # add option to run on gpu
    if (CUDA):
        encoder.cuda()
        decoder.cuda()
        mse_loss.cuda()

    # optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = LR, betas=(BETA1, BETA2))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if torch.cuda.is_available() and not CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # creating directories
    if not os.path.exists(osp.join('outputs', 'checkpoints', DATASET)):
            os.makedirs(osp.join('outputs', 'checkpoints', DATASET))

    if not os.path.exists(osp.join('outputs', 'reconstructed_images', DATASET)):
        os.makedirs(osp.join('outputs', 'reconstructed_images', DATASET))

    if not os.path.exists(osp.join('outputs', 'style_transfer_training', DATASET)):
        os.makedirs(osp.join('outputs', 'style_transfer_training', DATASET))

    dataset = load_dataset()
    loader = cycle(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True))

    sigma_p_inv, det_p = setup_pz(NUM_FEA, FEA_DIM, FEA)

    encoder_test = Encoder()
    encoder_test.apply(weights_init)

    decoder_test = Decoder()
    decoder_test.apply(weights_init)

    encoder_test.eval()
    decoder_test.eval()

    if CUDA:
        encoder_test.cuda()
        decoder_test.cuda()

    lowest_loss = float('inf')

    for epoch in range(START_EPOCH, END_EPOCH):
        epoch_losses = {'img': 0, 'kl': 0}
        # epoch_loss = 0
        # kl_annealing_factor = min(1, epoch / 100)
        kl_annealing_factor = 1 * BETA
        for iteration in range(len(dataset)//BATCH_SIZE):
            X_in = next(loader).float().cuda()
            optimizer.zero_grad()

            X1, KL1, muL1, det_q1 = encoder(X_in)
            dec = decoder(X1)
            img_loss = mse_loss(X_in, dec)
            epoch_losses['img'] += img_loss.item()

            sigma_q1 = torch.einsum('ijkl,ijlm->ijkm', KL1, torch.einsum('ijkl->ijlk', KL1))

            mul1_transpose = torch.transpose(muL1, dim0=1, dim1=2)
            if ZERO_MEAN_FEA:
                mu_p_transpose = get_prior_mean(FEA_MEAN_S, FEA_MEAN_E)
                kl_loss1 = KL_loss_L1(sigma_p_inv, sigma_q1, mul1_transpose, mu_p_transpose, det_p, det_q1)
            else:
                kl_loss1 = KL_loss_L1_without_mean(sigma_p_inv, sigma_q1, mul1_transpose, det_p, det_q1)

            # calculate KL divergence
            kl_loss = torch.mean(kl_loss1)
            kl_loss = kl_loss * kl_annealing_factor
            total_loss = img_loss + kl_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
            optimizer.step()

            # display losses
            # if (iteration % 200 == 0 and iteration != 0):
                # print('Epoch : ', epoch, ', Iteration : ', iteration, ", Total Loss : ", total_loss.item(),', Image Loss : ', torch.mean(img_loss).item(), ', KL Div. Loss : ', torch.mean(kl_loss).item())
                # print(f'Epoch: {epoch}, Iteration: {iteration}, Total loss: {total_loss.item()}, Image loss: {torch.mean(img_loss).item()}, KL Div. loss: {torch.mean(kl_loss).item()}({kl_annealing_factor})')
                # epoch_loss += total_loss.item()
            epoch_losses['kl'] += kl_loss1.sum().item()
        epoch_losses['img'] = epoch_losses['img'] / len(dataset)
        epoch_losses['kl'] = epoch_losses['kl'] / len(dataset)
        epoch_losses['total'] = epoch_losses['img'] + epoch_losses['kl']
        logger.info(f'Epoch: {epoch}, Total loss: {epoch_losses["total"]}, Image loss: {epoch_losses["img"]}, KL Div. loss: {epoch_losses["kl"]}(AF: {kl_annealing_factor})')
        scheduler.step()

        # retrieving another batch to reconstruct for saving reconstructed images
        X_in = next(loader).float().cuda()
        original_sample = X_in.cpu()[0, :, :, :, :]
        with torch.no_grad():
            enc, KL1, muL1, det_q1 = encoder(X_in)
            dec = decoder(enc)
            decoded_sample = dec.detach().cpu()[0, :, :, :, :]

        save_image(original_sample, OUTPUT_PATH + f'/{epoch}_{ENCODER_SAVE}_org.png', nrow=NUM_FRAMES, normalize=True)
        save_image(decoded_sample, OUTPUT_PATH + f'/{epoch}_{ENCODER_SAVE}_rec.png', nrow=NUM_FRAMES, normalize=True)

        if epoch_losses['total'] < lowest_loss:
            lowest_loss = epoch_losses['total']
            torch.save(encoder.state_dict(), os.path.join('outputs', 'checkpoints', DATASET, ENCODER_SAVE))
            torch.save(decoder.state_dict(), os.path.join('outputs', 'checkpoints', DATASET, DECODER_SAVE))
            logger.info('Model Saved! Epoch loss at {}'.format(lowest_loss))

            encoder_test.load_state_dict(torch.load(os.path.join('outputs', 'checkpoints', DATASET, ENCODER_SAVE)))
            decoder_test.load_state_dict(torch.load(os.path.join('outputs', 'checkpoints', DATASET, DECODER_SAVE)))

            video1 = next(loader).float().cuda()[0].unsqueeze(0)
            video2 = next(loader).float().cuda()[0].unsqueeze(0)

            with torch.no_grad():
                X1_v1, KL1_v1, muL1_v1, det_q1_v1 = encoder_test(video1, BATCH_SIZE=1)
                dec_v1 = decoder_test(X1_v1, BATCH_SIZE=1)

                X1_v2, KL1_v2, muL1_v2, det_q1_v2 = encoder_test(video2, BATCH_SIZE=1)
                dec_v2 = decoder_test(X1_v2, BATCH_SIZE=1)

            plot_training_images(video1, video2, dec_v1, dec_v2, X1_v1, X1_v2, epoch, decoder_test)

