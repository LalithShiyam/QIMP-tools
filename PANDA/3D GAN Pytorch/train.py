import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
from generators import *
from discriminators import *
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from init import Options
from utils.utils import *
from logger import *


# -----  Loading the init options -----
opt = Options().parse()
min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))

if opt.gpu_ids != '-1':
    num_gpus = len(opt.gpu_ids.split(','))
else:
    num_gpus = 0
print('number of GPU:', num_gpus)
# -------------------------------------



# -----  Loading the list of data -----
train_list = create_list(opt.data_path)
val_list = create_list(opt.val_path)

for i in range(opt.increase_factor_data):  # augment the data list for training

    train_list.extend(train_list)
    val_list.extend(val_list)

print('Number of training patches per epoch:', len(train_list))
print('Number of validation patches per epoch:', len(val_list))
# -------------------------------------




# -----  Transformation and Augmentation process for the data  -----
trainTransforms = [
            NiftiDataset.Resample(opt.new_resolution, opt.resample),
            NiftiDataset.Augmentation(),
            NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
            NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
            ]

train_set = NifitDataSet(train_list, direction=opt.direction, transforms=trainTransforms, train=True)    # define the dataset and loader
train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)  # Here are then fed to the network with a defined batch size
# -------------------------------------





# -----  Creating the Generator and discriminator -----
generator = build_netG(opt)
discriminator = build_netD(opt)
check_dir(opt.checkpoints_dir)

# -----  Pretrain the Generator and discriminator -----
if opt.resume:
    generator.load_state_dict(new_state_dict(opt.generatorWeights))
    discriminator.load_state_dict(new_state_dict(opt.discriminatorWeights))
    print('Generator and discriminator Weights are loaded')
else:
    pretrainW = './checkpoints/g_pre-train.pth'
    if os.path.exists(pretrainW):
        generator.load_state_dict(new_state_dict(pretrainW))
        print('Pre-Trained G Weight is loaded')
# -------------------------------------





criterionMSE = nn.MSELoss()  # nn.MSELoss()
criterionGAN = GANLoss()
criterion_pixelwise = nn.L1Loss()
# -----  Use Single GPU or Multiple GPUs -----
if (opt.gpu_ids != -1) & torch.cuda.is_available():
    use_gpu = True
    generator.cuda()
    discriminator.cuda()
    criterionGAN.cuda()
    criterion_pixelwise.cuda()

    if num_gpus > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

optim_generator = optim.Adam(generator.parameters(), betas=(0.5,0.999), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), betas=(0.5,0.999), lr=opt.discriminatorLR)
net_g_scheduler = get_scheduler(optim_generator, opt)
net_d_scheduler = get_scheduler(optim_discriminator, opt)
# -------------------------------------



# -----  Training Cycle -----
print('Start training :) ')
epoch_count = opt.epoch_count
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for batch_idx, (data, label) in enumerate(train_loader):

        real_a = data
        real_b = label

        if use_gpu:                              # forward
            real_b = real_b.cuda()
            fake_b = generator(real_a.cuda())    # generate fake data
            real_a = real_a.cuda()
        else:
            fake_b = generator(real_a)

        ######################
        # (1) Update D network
        ######################
        optim_discriminator.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = discriminator.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)

        # Combined D loss
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        mean_discriminator_loss += discriminator_loss
        discriminator_loss.backward()
        optim_discriminator.step()

        ######################
        # (2) Update G network
        ######################

        optim_generator.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = discriminator.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterion_pixelwise(fake_b, real_b) * opt.lamb

        costrain = Cor_CoeLoss(fake_b, real_b) * opt.lamb

        generator_total_loss = loss_g_gan + loss_g_l1 + costrain

        mean_generator_total_loss += generator_total_loss
        generator_total_loss.backward()
        optim_generator.step()


        ######### Status and display #########
        sys.stdout.write(
            '\r [%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f' % (
                epoch_count, (opt.niter + opt.niter_decay + 1), batch_idx, len(train_loader),
                discriminator_loss, generator_total_loss))

    update_learning_rate(net_g_scheduler, optim_generator)
    update_learning_rate(net_d_scheduler, optim_discriminator)

    ##### Logger ######

    valTransforms = [
        NiftiDataset.Resample(opt.new_resolution, opt.resample),
        NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
        NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    ]

    val_set = NifitDataSet(val_list, direction=opt.direction, transforms=valTransforms, test=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    plot_generated_batch(val_list=val_list, model=generator, resample=opt.resample, resolution=opt.new_resolution,
                         patch_size_x=opt.patch_size[0], patch_size_y=opt.patch_size[1],
                         patch_size_z=opt.patch_size[2], stride_inplane=opt.stride_inplane,
                         stride_layer=opt.stride_layer, batch_size=1,
                         epoch=epoch_count)

    # test
    avg_psnr = 0
    for batch in val_loader:
        input, target = batch[0].cuda(), batch[1].cuda()

        prediction = generator(input)
        mse = criterionMSE(prediction, target)
        from math import log10
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr

    epoch_count += 1

    if epoch % opt.save_fre == 0:
        # Do checkpointing
        torch.save(generator.state_dict(), '%s/g_epoch_{}.pth'.format(epoch) % opt.checkpoints_dir)
        torch.save(discriminator.state_dict(), '%s/d_epoch_{}.pth'.format(epoch) % opt.checkpoints_dir)

    sys.stdout.write(
        '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss:%.4f Avg. PSNR:%.4f \n' % (
            epoch_count-1, (opt.niter + opt.niter_decay + 1), batch_idx, len(train_loader),
            mean_discriminator_loss / len(train_loader),
            mean_generator_total_loss / len(train_loader),
            avg_psnr / len(val_loader)))
