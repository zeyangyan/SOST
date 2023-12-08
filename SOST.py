import argparse
import itertools
from torch.utils.data import DataLoader
from models import *
from datasets import *
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=260, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=261, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="synth2obs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=1, help="size of height")
parser.add_argument("--img_width", type=int, default=4800, help="size of width")
parser.add_argument("--channels", type=int, default=1, help="number of channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
cuda = False

# Create sample and checkpoint directories
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

criterion_recon = torch.nn.L1Loss()

# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

if cuda:
    Enc1 = Enc1.cuda()
    Dec1 = Dec1.cuda()
    Enc2 = Enc2.cuda()
    Dec2 = Dec2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_recon.cuda()

if opt.epoch != 0:
    # Load pretrained models
    # Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, opt.epoch)))
    # Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, opt.epoch)))
    # Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, opt.epoch)))
    # Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, opt.epoch)))
    # D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.dataset_name, opt.epoch)))
    # D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.dataset_name, opt.epoch)))
    Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.dataset_name, opt.epoch), map_location='cpu'))
else:
    # Initialize weights
    Enc1.apply(weights_init_normal)
    Dec1.apply(weights_init_normal)
    Enc2.apply(weights_init_normal)
    Dec2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

# Loss weights
lambda_gan = 1
lambda_id = 5
lambda_style = 1
lambda_cont = 20
lambda_cyc = 1
lambda_X12X1 = 1

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Configure dataloaders

#只有A类E:\Aa学习\研二上\A-cyc-munit实验结果\fit_data 全E:\Aa学习\研二上\实验\spe_cyclegan\fit_data

spedataset = SpeDataset(r"fit_data", unaligned=True)

dataloader = DataLoader(
    dataset=spedataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

#只有A类E:\Aa学习\研二上\A-cyc-munit实验结果\fit_data
# Test data loader
val_dataloader = DataLoader(
    SpeDataset(r"fit_data", mode="test"),
    batch_size=1,
    num_workers=0,
)


print(Enc1)
print(Dec1)
print(D1)



