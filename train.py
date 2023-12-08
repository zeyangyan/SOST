from SOST import *
import numpy as np
import time
import sys
# ----------
#  Training
# ----------

valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = Variable(batch["A"].type(Tensor))
        X2 = Variable(batch["B"].type(Tensor))


        # 提取与logg有关的通量
        # X1_index = read_syn_line_index(X1)
        # X2_index = read_obs_line_index(X2)

        # print(X1.shape, 'X1.shape') [4, 1, 4800]

        # Sampled style codes
        style_1 = Variable(torch.randn(X1.size(0), opt.style_dim, 1).type(Tensor))
        style_2 = Variable(torch.randn(X1.size(0), opt.style_dim, 1).type(Tensor))


        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)
        # print(c_code_1.shape,s_code_1.shape,'cscode')  [1,256,1197] [1,8,1]
        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # 提取与logg有关的通量
        # X11_index = read_syn_line_index(X11)
        # X22_index = read_obs_line_index(X22)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # print(X12)
        # 提取与logg有关的通量
        # X21_index = read_syn_line_index(X21)
        # X12_index = read_obs_line_index(X12)

        # print(X21.shape,'X21.shape') [1,1,4800]
        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        # X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        # X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0
        X121 = Dec1(c_code_12, s_code_1)
        X212 = Dec2(c_code_21, s_code_2)



        # 提取与logg有关的通量
        # X121_index = read_syn_line_index(X121)
        # X212_index = read_obs_line_index(X212)

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)  # syn
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)  # obs
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        # loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        # loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1)
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2)



        # loss_x12_x1 = criterion_recon(X12, X1)

        # _wavelength = np.linspace(3900, 8699, num=4800)

        # ------------cos loss--------------
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # empty_tensor = torch.empty((0,)).to(device)
        # for i1, j1 in zip(X12,X1):
        #     i1 = i1.detach().cpu().numpy().reshape(-1)
        #     j1 = j1.detach().cpu().numpy().reshape(-1)
        #     # print(_wavelength.shape)
        #     input1 = torch.from_numpy(i1).to(device)
        #     x12_norm = torch.reshape(input1, (1, -1))
        #     input2 = torch.from_numpy(j1).to(device)
        #     x1_norm = torch.reshape(input2, (1, -1))
        #     corr_coef = torch.nn.functional.cosine_similarity(x12_norm, x1_norm, dim=1).to(device)
        #     corr_coef = (1 - corr_coef)
        #     empty_tensor = torch.cat((empty_tensor, corr_coef)).to(device)
        # loss_x12_x1 = lambda_X12X1 * (sum(empty_tensor))



        # _wavelength = np.linspace(3900, 8699, num=4800)
        #
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # empty_tensor = torch.empty((0,)).to(device)
        # for i1, j1 in zip(X11,X1):
        #     i1 = i1.detach().cpu().numpy().reshape(-1)
        #     j1 = j1.detach().cpu().numpy().reshape(-1)
        #     # print(_wavelength.shape)
        #     X121_norm = normalize_spectrum_spline(wave=_wavelength, flux=i1)
        #     X1_norm = normalize_spectrum_spline(wave=_wavelength, flux=j1)
        #     result = np.vstack((X121_norm, X1_norm))
        #     result = torch.from_numpy(result).to(device)
        #     corr_coef = torch.corrcoef(result)[0, 1].to(device)
        #     corr_coef = (1-corr_coef).unsqueeze(0)
        #     empty_tensor = torch.cat((empty_tensor, corr_coef)).to(device)
        # loss_X121_X1 = lambda_X12X1 * sum(empty_tensor)
        #
        # _wavelength = np.linspace(3900, 8699, num=4800)
        #
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # empty_tensor = torch.empty((0,)).to(device)
        # for i2, j2 in zip(X22,X2):
        #     i2 = i2.detach().cpu().numpy().reshape(-1)
        #     j2 = j2.detach().cpu().numpy().reshape(-1)
        #     # print(_wavelength.shape)
        #     X212_norm = normalize_spectrum_spline(wave=_wavelength, flux=i2)
        #     X2_norm = normalize_spectrum_spline(wave=_wavelength, flux=j2)
        #     result = np.vstack((X212_norm, X2_norm))
        #     result = torch.from_numpy(result).to(device)
        #     corr_coef = torch.corrcoef(result)[0, 1].to(device)
        #     corr_coef = (1-corr_coef).unsqueeze(0)
        #     empty_tensor = torch.cat((empty_tensor, corr_coef)).to(device)
        # loss_X212_X2 = lambda_X12X1 * sum(empty_tensor)


        # _wavelength = np.linspace(3900, 8699, num=4800)
        # x12_norm = normalize_spectrum_spline(wave = _wavelength, flux = X12)
        # x1_norm = normalize_spectrum_spline(wave = _wavelength, flux = X1)
        # loss_x12_x1 = criterion_recon(x12_norm, x1_norm)



        #与logg有关的loss
        # loss_GAN_index_1 = lambda_gan * criterion_recon(X21_index, X2_index)
        # loss_GAN_index_2 = lambda_gan * criterion_recon(X12_index, X1_index)
        # loss_ID_index_1 = lambda_id * criterion_recon(X1_index, X11_index)
        # loss_ID_index_2 = lambda_id * criterion_recon(X2_index, X22_index)
        # loss_cyc_index_1 = lambda_cyc * criterion_recon(X121_index, X1_index)
        # loss_cyc_index_2 = lambda_cyc * criterion_recon(X212_index, X2_index)

        # print(loss_GAN_index_1,'loss_GAN_index_1')
        # # print(loss_GAN_1, 'loss_GAN_1')
        # # print(loss_ID_1, 'loss_ID_1')
        # # print(loss_s_1, 'loss_s_1')
        # # print(loss_cyc_1, 'loss_cyc_1')
        # print(loss_cyc_index_1, 'loss_cyc_index_1')
        # print(loss_ID_index_1, 'loss_ID_index_1')
        # # print(loss_c_1, 'loss_c_1')


        # Total loss
        loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
            # + loss_X121_X1
            # + loss_X212_X2
            # + loss_x12_x1
            # + loss_GAN_index_1
            # + loss_GAN_index_2
            # + loss_ID_index_1
            # + loss_ID_index_2
            # + loss_cyc_index_1
            # + loss_cyc_index_2
        )

        distance_loss = (
            loss_ID_1
            + loss_ID_2
            + loss_cyc_1
            + loss_cyc_2
        )

        latent_loss = (
            loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
        )

        gan_loss = (
                loss_GAN_1
                + loss_GAN_2
        )
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        # batches_done = epoch * len(dataloader) + i
        # batches_left = opt.n_epochs * len(dataloader) - batches_done
        #
        # time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        # prev_time = time.time()

        # Print log
        # sys.stdout.write(
        #     "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [distance loss: %f] [latent loss: %f] [gan loss: %f] [loss_X121_X1 loss: %f] [X212_X2 loss: %f] "
        #     % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), distance_loss.item(), latent_loss.item(), gan_loss.item(), loss_X121_X1.item(), loss_X212_X2.item())
        # )
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [distance loss: %f] [latent loss: %f] [gan loss: %f]   "
            % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), distance_loss.item(), latent_loss.item(), gan_loss.item())
        )
        # sys.stdout.write(
        #     "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [distance loss: %f] [latent loss: %f] [gan loss: %f]  ETA: %s"
        #     % (epoch, opt.n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), distance_loss.item(), latent_loss.item(), gan_loss.item() , time_left)
        # )

        # losses_cp['zsh_synth_val'].append(zsh_synth_rec_score.data.item())
        # losses_cp['zsh_obs_val'].append(zsh_obs_rec_score.data.item())
        # losses_cp['zsh_val'].append(zsh_score.data.item())
        # if model.use_split:
        #     losses_cp['zsp_val'].append(zsp_score.data.item())
        # losses_cp['x_synthobs_val'].append(x_synthobs_score.data.item())
        # losses_cp['x_obssynth_val'].append(x_obssynth_score.data.item())


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Enc1.state_dict(), "saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Dec1.state_dict(), "saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Enc2.state_dict(), "saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Dec2.state_dict(), "saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D1.state_dict(), "saved_models/%s/D1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D2.state_dict(), "saved_models/%s/D2_%d.pth" % (opt.dataset_name, epoch))