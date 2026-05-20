import torch
import sys, os
sys.path += ["./src", "./configs", "./models"]
import numpy as np
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pth_dir = "./outputs/2026-05-07~15h56min-resunet"
dim = (783, 367, 700)
xs, xr, nt = dim
shot_index = 390
interpolate_begin = 230
interpolate_end = 250
sample_path = "./data/paper-real-miss40/test2-samples-miss40.dat"
label_path = "./data/paper-real-miss40/test2-labels-miss40.dat"

'''
pth_dir = "./outputs/01-pluto-miss60"
dim = (550, 550, 1000)
xs, xr, nt = dim
shot_index = 180
interpolate_begin = 161
interpolate_end = 200
sample_path = "./data/paper-pluto-new-131-230-100pair-miss60/test1-shots_full_focal-miss60.dat"
label_path = "./data/paper-pluto-new-131-230-100pair-miss60/p550.dat"
'''
'''
pth_dir = "./outputs/2026-04-01~19h13min-resunet"
dim = (501, 501, 500)
xs, xr, nt = dim
shot_index = 249
interpolate_begin = 240
interpolate_end = 259
sample_path = "./data/paper-diffr-201-300-100pair-miss20/test1-shots_full_focal-miss20.dat"
label_path = "./data/paper-diffr-201-300-100pair-miss20/shot_fullwave_jieti_use.dat"
# sample_path = "./data/diffr-201-300-100pair/labels.dat"
# label_path = "./data/diffr-201-300-100pair/labels1.dat"
'''
# ================== MODEL ==================
# networks = ['unet', 'denseunet', 'resunet', 'rdbunet']
# test_model, network_tag, args = create_model(networks[2], device)

sys.path += pth_dir

from resunet_configs import get_config
import ResUnet
args = get_config()
test_model = ResUnet.ResUnet(
    args.IN_CHANNELS,
    args.OUT_CHANNELS,
    args.BASE_CHANNELS,
    args.DEPTH,
    args.ACT_TYPE,
    args.NORM_LAYER,
    args.UPSAMPLE,
    dropout=args.DROPOUT,
    attention_layers=args.ATTN_LAYERS,
).to(device)
'''
from unet_configs import get_config
import Unet
args = get_config()
test_model = Unet.Unet(
    args.IN_CHANNELS,
    args.OUT_CHANNELS,
    args.BASE_CHANNELS,
    args.DEPTH,
    args.ACT_TYPE,
    args.NORM_LAYER,
    args.UPSAMPLE,
    dropout=args.DROPOUT,
    attention_layers=args.ATTN_LAYERS,
).to(device)
'''
test_model.load_state_dict(torch.load(pth_dir + "/model.pth", weights_only=True))

sample = np.fromfile(sample_path, dtype=np.float32).reshape(dim).transpose(2, 1, 0)
label = np.fromfile(label_path, dtype=np.float32).reshape(dim).transpose(2, 1, 0)

scale = np.max(np.abs(label))
sample_vol = sample / scale
label_vol = label / scale

# Extract a single shot
# nx = sample_vol[:, 0:184, shot_index]
# ny = label_vol[:, 0:184, shot_index]

nx = sample_vol[:, :, shot_index]
ny = label_vol[:, :, shot_index]
'''
nx = sample[:, :, shot_index]
nx = nx / np.max(np.abs(nx))
ny = label[:, :, shot_index]
ny = ny / np.max(np.abs(ny))
'''
# out = model_use(nx, test_model, dim=(nt, xr), device=device, block_size=args.BLOCK_SIZE, complete=True)
out = model_use_block(nx, test_model, block_size=args.BLOCK_SIZE, overlap=0.5)

out.transpose(1, 0).flatten().tofile(pth_dir + "./output.dat")
(ny - out).transpose(1, 0).flatten().tofile(pth_dir + "./res.dat")

SNR_ = snr(ny, out)
MSE_ = mse(ny, out)
PSNR_ = psnr(torch.from_numpy(ny), torch.from_numpy(out))
SSIM_ = ssim(
    torch.from_numpy(ny).unsqueeze(0).unsqueeze(0).float(),
    torch.from_numpy(out).unsqueeze(0).unsqueeze(0).float()
)

print(f"SNR={SNR_:.4f}, MSE={MSE_:.6f}, PSNR={PSNR_:.4f}, SSIM={SSIM_:.4f}")

plot_seismic_results(
    ny=ny, out=out,
    SNR_=SNR_, MSE_=MSE_, PSNR_=PSNR_, SSIM_=SSIM_,
    pth_dir=pth_dir, nt=nt, xr=xr, interpolate_begin=interpolate_begin, interpolate_end=interpolate_end,
    dt=0.004, dx=10, ox=0, ot=0, vmin=-0.15, vmax=0.15
)
