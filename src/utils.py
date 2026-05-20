import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.psnr import PeakSignalNoiseRatio as PSNR

import os
from datetime import datetime
import sys
sys.path += ["./src", "./configs", "./models"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssim = SSIM(data_range=1.0).to(device)
psnr = PSNR(data_range=1.0).to(device)


def visualize_samples(dataset, aug_num, num_samples=3, enhancement_names=None):
    """
    Group visualization of data augmentation results - clearly showing different augmentation effects

    Parameters:
        dataset: seismic_dataset instance
        aug_num: Number of augmentations per original patch
        num_samples: Number of original patches to display
        enhancement_names: List of augmentation type names
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Default augmentation names
    if enhancement_names is None:
        enhancement_names = [f'Aug_{i}' for i in range(aug_num)]
        enhancement_names[0] = 'Original'  # First is original data

    total_samples = len(dataset)
    original_num = total_samples // aug_num
    sample_indices = np.random.choice(original_num, min(num_samples, original_num), replace=False)

    # Create figure - clearer layout
    fig, axes = plt.subplots(num_samples, aug_num * 2,
                             figsize=(4 * aug_num, 3 * num_samples))

    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, original_idx in enumerate(sample_indices):
        print(f"Visualizing original sample {original_idx}")

        for j in range(aug_num):
            augmented_idx = original_idx * aug_num + j

            if augmented_idx >= total_samples:
                continue

            try:
                sample, label = dataset[augmented_idx]
                sample_img = sample.squeeze().cpu().numpy() if hasattr(sample, 'cpu') else sample.squeeze()
                label_img = label.squeeze().cpu().numpy() if hasattr(label, 'cpu') else label.squeeze()

                # Sample column
                ax_sample = axes[i, j * 2]
                im1 = ax_sample.imshow(sample_img, cmap='seismic', aspect='auto')
                ax_sample.set_title(f'{enhancement_names[j]}\nSample', fontsize=9)
                ax_sample.axis('off')
                plt.colorbar(im1, ax=ax_sample, fraction=0.046, pad=0.02)

                # Label column
                ax_label = axes[i, j * 2 + 1]
                im2 = ax_label.imshow(label_img, cmap='seismic', aspect='auto')
                ax_label.set_title(f'{enhancement_names[j]}\nLabel', fontsize=9)
                ax_label.axis('off')
                plt.colorbar(im2, ax=ax_label, fraction=0.046, pad=0.02)

            except Exception as e:
                print(f"Error processing sample {augmented_idx}: {e}")
                axes[i, j * 2].axis('off')
                axes[i, j * 2 + 1].axis('off')

    plt.tight_layout()
    plt.savefig("./dataset_grouped_visualization.png", format='png',
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.show()


def init_weights(m):
    """
    Professional weight initialization function
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Use Kaiming initialization, optimized for ReLU and its variants
        # Your activation functions include ReLU, LeakyReLU, ELU, GELU, SiLU, which are all from the ReLU family
        nn.init.kaiming_normal_(m.weight,
                                mode='fan_out',
                                nonlinearity='relu')

        # If bias exists, initialize to a small constant (usually 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        # Normalization layer: initialize weight to 1, bias to 0
        # This approximates an identity transformation initially, preserving the previous distribution
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def snr(img1, img2):
    noise = img1 - img2
    # Compute signal power (power of the original image)
    signal_power = np.mean(img1 ** 2)
    # Compute noise power
    noise_power = np.mean(noise ** 2)
    eps = 1e-10
    snr_val = 10 * np.log10(signal_power / (noise_power + eps))
    return snr_val


def mse(xref, xcmp):
    mse = np.mean(np.abs(xref - xcmp) ** 2)
    return mse


def create_gaussian_weight(window_size, sigma=0.3):
    """
    Create Gaussian weight matrix, with higher weight at the center and lower weight at the edges
    """
    win_h, win_w = window_size
    y, x = torch.meshgrid(
        torch.arange(win_h, dtype=torch.float32),
        torch.arange(win_w, dtype=torch.float32),
        indexing='ij'
    )

    center_y, center_x = (win_h - 1) / 2, (win_w - 1) / 2
    # weight = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) /
    #                   (2 * (sigma * max(win_h, win_w)) ** 2))

    weight = torch.exp(-(((x - center_x) / (sigma * win_w)) ** 2 +
                         ((y - center_y) / (sigma * win_h)) ** 2) / 2)
    return weight.unsqueeze(0).unsqueeze(0)


def model_use_block(img, model, block_size, overlap):
    """
    Parameters
    ----------
    img : TYPE
        Input image, 2D dimensions.
    model : TYPE
        Trained network model.
    block_size : TYPE
        Window size.
    overlap : TYPE
        Overlap ratio.

    Apply the network to the input image using a sliding window.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = img.shape
    win_h, win_w = block_size

    # Compute stride
    stride_h = int(win_h * (1 - overlap))
    stride_w = int(win_w * (1 - overlap))

    i_range = list(range(0, H - win_h, stride_h))
    j_range = list(range(0, W - win_w, stride_w))

    if i_range[-1] + stride_h > H - win_h:
        i_range.append(H - win_h)  # Last valid starting position

    if j_range[-1] + stride_w > W - win_w:
        j_range.append(W - win_w)  # Last valid starting position

    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    img_tensor = img_tensor.to(device, dtype=torch.float32)
    output = torch.zeros((1, 1, H, W), dtype=torch.float32).to(device)
    weight_sum = torch.zeros((1, 1, H, W), dtype=torch.float32).to(device)

    # Precompute Gaussian weights
    gaussian_weight = create_gaussian_weight(block_size).to(device)
    '''
    model.eval()
    with torch.no_grad():
        for i in i_range:
            for j in j_range:
                window = img_tensor[:,:,i:i+win_h, j:j+win_w]
                pred = model(window)
                # Accumulate using Gaussian weights
                output[:, :, i:i+win_h, j:j+win_w] += pred * gaussian_weight
                weight_sum[:, :, i:i+win_h, j:j+win_w] += gaussian_weight
    '''

    # Collect all windows in batches for single forward propagation
    windows, coords = [], []
    for i in i_range:
        for j in j_range:
            windows.append(img_tensor[:, :, i:i + win_h, j:j + win_w])
            coords.append((i, j))

    model.eval()
    with torch.no_grad():
        batch = torch.cat(windows, dim=0)  # (N, 1, win_h, win_w)
        preds = model(batch)  # (N, 1, win_h, win_w)

    for idx, (i, j) in enumerate(coords):
        pred = preds[idx:idx + 1].unsqueeze(0) if preds.dim() == 3 else preds[idx:idx + 1]
        output[:, :, i:i + win_h, j:j + win_w] += pred * gaussian_weight
        weight_sum[:, :, i:i + win_h, j:j + win_w] += gaussian_weight
    output = output / (weight_sum + 1e-8)

    return output.squeeze(0).squeeze(0).cpu().numpy()


def model_use(img, model, dim, device, block_size=(128, 128), complete=True):
    """
    Pad the image to multiples of block_size, apply network inference, then remove padding.
    """
    h, w = dim[0], dim[1]
    block_h, block_w = block_size

    # Pad image to fit block size
    pad_h = (block_h - h % block_h) % block_h if h % block_h != 0 else 0
    pad_w = (block_w - w % block_w) % block_w if w % block_w != 0 else 0
    # img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='zero')
    # img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='wrap')

    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    img_tensor = img_tensor.to(device, dtype=torch.float32)

    if complete:
        model.eval()
        with torch.no_grad():
            out = model(img_tensor)
        y_ = out.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
    else:
        model.eval()
        num_blocks_h = (h + pad_h) // block_h
        num_blocks_w = (w + pad_w) // block_w
        y = torch.zeros((1, 1, h + pad_h, w + pad_w), dtype=img_tensor.dtype).to(device)

        with torch.no_grad():
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    block = img_tensor[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    out = model(block)
                    y[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = out

        y_ = y.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

    return y_[0:h, 0:w]  # Crop to original image size


def model_use1(img, y, model, dim, device, block_size=(128, 128), complete=True):
    """
    Pad the image to multiples of block_size, apply network inference, then remove padding.
    This version concatenates img and y along the channel dimension (experimental).
    """
    h, w = dim[0], dim[1]
    block_h, block_w = block_size

    # Pad image to fit block size
    pad_h = (block_h - h % block_h) % block_h if h % block_h != 0 else 0
    pad_w = (block_w - w % block_w) % block_w if w % block_w != 0 else 0
    # img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='zero')
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    y_padded = np.pad(y, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    img_padded = np.expand_dims(img_padded, axis=0)
    y_padded = np.expand_dims(y_padded, axis=0)
    img_padded = np.concatenate([img_padded, y_padded], axis=0)

    img_tensor = torch.from_numpy(img_padded).unsqueeze(0).float()  # Add batch and channel dimensions
    img_tensor = img_tensor.to(device)
    model.eval()
    if complete:
        with torch.no_grad():
            out = model(img_tensor)
        y_ = out.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions
    else:
        num_blocks_h = (h + pad_h) // block_h
        num_blocks_w = (w + pad_w) // block_w
        y = torch.zeros((1, 1, h + pad_h, w + pad_w), dtype=img_tensor.dtype).to(device)

        with torch.no_grad():
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    block = img_tensor[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    out = model(block)
                    y[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = out

        y_ = y.squeeze(0).squeeze(0).cpu().numpy()  # Remove batch and channel dimensions

    return y_[0:h, 0:w]  # Crop to original image size


def plot_training_curves(train_loss, val_loss, ssim_list=None, psnr_list=None, num_epochs=None,
                         save_dir="./outputs", ):
    """
    Plot training and validation loss curves, with optional SSIM/PSNR subplot.

    Parameters:
    ----------
    train_loss : list or np.ndarray
        Training loss records.
    val_loss : list or np.ndarray
        Validation loss records.
    ssim_list : list or np.ndarray, optional
        SSIM metric values per epoch.
    psnr_list : list or np.ndarray, optional
        PSNR metric values per epoch.
    num_epochs : int, optional
        Total number of training epochs, auto-determined from train_loss length if None.
    save_dir : str, default="./outputs"
        Directory to save the figure.
    """
    if num_epochs is None:
        num_epochs = len(train_loss)

    epochs_axis = np.arange(1, num_epochs + 1)

    # Create main figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_axis, train_loss, '-', color='red', linewidth=2.5, label='Train Loss')
    ax.plot(epochs_axis, val_loss, '-', color='blue', linewidth=2.5, label='Val Loss')

    ax.set_xlabel('Epochs', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid(color='grey', linestyle='-', linewidth=0.1)
    ax.legend(fontsize=12)

    # Add subplot if SSIM and PSNR are provided
    if ssim_list is not None and psnr_list is not None:
        left, bottom, width, height = [0.68, 0.5, 0.2, 0.2]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(epochs_axis, ssim_list, '-', color='black', linewidth=2.0, label='SSIM')
        ax2.plot(epochs_axis, np.array(psnr_list) / np.max(psnr_list), '-', color='green', linewidth=2.0, label='PSNR')

        # ax2.set_yscale('log')
        ax2.set_xlabel('Epochs', fontsize=8)
        ax2.set_ylabel('Metric Value', fontsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.legend(fontsize=8, loc='lower right')
        ax2.grid(True, linestyle='--', linewidth=0.2)

    # Save figure
    plt.savefig(f"{save_dir}/loss.png", format='png', bbox_inches='tight', dpi=1000)
    plt.show()

    print(f"✅ Loss curves saved to: {save_dir}")

def plot_seismic_results(ny, out, SNR_, MSE_, PSNR_, SSIM_, pth_dir,
                         nt, xr, interpolate_begin, interpolate_end, dt=0.004, dx=10, ox=0, ot=0, vmin=-0.5, vmax=0.5):
    """
    Plot seismic imaging results from neural network output (original/predicted/difference/interpolated).
    """
    labels = ['a)', 'b)', 'c)', 'd)']

    x = (np.arange(xr) + ox)  # Trace Number
    x = np.reshape(x, (xr, 1))
    t = ((np.arange(nt) + ot) * dt)
    t = np.reshape(t, (nt, 1))

    fig, axs = plt.subplots(1, 4, sharey=False, figsize=(16, 5),
                            gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    lx = -0.1
    ly = 1.1
    fontsize = 15
    plt.rcParams.update({
        'axes.labelsize': fontsize,  # Axis label size
        'xtick.labelsize': fontsize,  # X-axis tick label size
        'ytick.labelsize': fontsize,  # Y-axis tick label size
    })
    # 1️⃣ Original data
    im0 = axs[0].imshow(ny, cmap='gray', vmin=vmin, vmax=vmax,
                        extent=(x[0, 0], x[-1, -1], t[-1, -1], t[0, 0]))
    axs[0].xaxis.set_ticks_position('top')  # X-axis ticks on top
    axs[0].xaxis.set_label_position('top')  # X-axis label on top
    axs[0].tick_params(direction='in')
    axs[0].set_xlabel('Distance(m)')
    axs[0].set_ylabel('Time (s)')
    # axs[0].set_title('Original')
    axs[0].text(lx, ly, labels[0], fontsize=20, weight='bold', ha='left', va='top', transform=axs[0].transAxes)  # Add label
    axs[0].axis('tight')

    # 2️⃣ Network output
    im1 = axs[1].imshow(out, cmap='gray', vmin=vmin, vmax=vmax,
                        extent=(x[0, 0], x[-1, -1], t[-1, -1], t[0, 0]))
    axs[1].xaxis.set_ticks_position('top')  # X-axis ticks on top
    axs[1].xaxis.set_label_position('top')  # X-axis label on top
    axs[1].tick_params(direction='in')
    axs[1].set_xlabel('Distance(m)')
    axs[1].set_ylabel('Time (s)')
    # axs[1].set_title(f'Network Output')
    axs[1].text(0.02, 0.98, f"SSIM:{SSIM_:.4f}\nPSNR:{PSNR_:.4f}\nSNR:{SNR_:.4f}",
                fontsize=10, color='black', ha='left', va='top', transform=axs[1].transAxes)
    axs[1].text(lx, ly, labels[1], fontsize=20, weight='bold', ha='left', va='top', transform=axs[1].transAxes)  # Add label
    axs[1].axis('tight')

    # 3️⃣ Difference plot
    diff = ny - out
    im2 = axs[2].imshow(diff, cmap='gray', vmin=vmin, vmax=vmax,
                        extent=(x[0, 0], x[-1, -1], t[-1, -1], t[0, 0]))
    axs[2].xaxis.set_ticks_position('top')  # X-axis ticks on top
    axs[2].xaxis.set_label_position('top')  # X-axis label on top
    axs[2].tick_params(direction='in')
    axs[2].set_xlabel('Distance(m)')
    axs[2].set_ylabel('Time (s)')
    axs[2].text(0.02, 0.98, f'Difference (MSE={MSE_:.4f})',
                fontsize=10, color='black', ha='left', va='top', transform=axs[2].transAxes)
    # axs[2].set_title(f'Difference (MSE={MSE_:.4f})')
    axs[2].text(lx, ly, labels[2], fontsize=20, weight='bold', ha='left', va='top', transform=axs[2].transAxes)  # Add label
    axs[2].axis('tight')

    # 4️⃣ Interpolation result
    inter = ny.copy()
    inter[:, interpolate_begin:interpolate_end] = out[:, interpolate_begin:interpolate_end]
    inter.transpose(1, 0).flatten().tofile(pth_dir + "./inter_output.dat")
    im3 = axs[3].imshow(inter, cmap='gray', vmin=vmin, vmax=vmax,
                        extent=(x[0, 0], x[-1, -1], t[-1, -1], t[0, 0]))
    axs[3].xaxis.set_ticks_position('top')  # X-axis ticks on top
    axs[3].xaxis.set_label_position('top')  # X-axis label on top
    axs[3].tick_params(direction='in')
    axs[3].set_xlabel('Distance(m)')
    axs[3].set_ylabel('Time (s)')
    # axs[3].set_title('Interpolation Result')
    axs[3].text(lx, ly, labels[3], fontsize=20, weight='bold', ha='left', va='top', transform=axs[3].transAxes)  # Add label
    axs[3].axis('tight')

    plt.tight_layout()
    plt.savefig(f"{pth_dir}/result.png", dpi=1000, bbox_inches='tight')
    plt.show()


# ================== λ Adaptive Controller ==================
class LambdaScheduler:
    """
    Scheduler for adaptively adjusting the SSIM weight λ.
    Changes λ based on validation set SSIM to dynamically balance structural similarity and pixel error.
    """

    def __init__(self, initial_lambda=0.1, min_lambda=0.1, max_lambda=1.0,
                 step=0.01, patience=5, smooth_factor=0.9):
        self.lambda_value = initial_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.step = step
        self.patience = patience
        self.smooth_factor = smooth_factor
        self.best_ssim = 0.0
        self.no_improve_epochs = 0

    def update(self, current_ssim):
        """
        Update λ based on validation set SSIM
        """
        if current_ssim > self.best_ssim + 1e-4:  # SSIM improved
            self.best_ssim = current_ssim
            self.no_improve_epochs = 0
            new_lambda = min(self.lambda_value + self.step, self.max_lambda)
        else:
            self.no_improve_epochs += 1
            if self.no_improve_epochs >= self.patience:
                new_lambda = max(self.lambda_value - self.step, self.min_lambda)
                self.no_improve_epochs = 0
            else:
                new_lambda = self.lambda_value

        # Smooth update (prevent oscillation)
        self.lambda_value = (
            self.smooth_factor * self.lambda_value +
            (1 - self.smooth_factor) * new_lambda
        )
        return self.lambda_value


def create_model(model_name, device):
    """Unified model factory function"""
    if model_name == 'denseunet':
        from denseunet_configs import get_config
        import DenseUnet
        args = get_config()
        model = DenseUnet.DenseUnet(
            args.IN_CHANNELS,
            args.OUT_CHANNELS,
            args.BASE_CHANNELS,
            args.DEPTH,
            args.GROWTH_RATE,
            args.NUM_CONVS,
            args.ACT_TYPE,
            args.NORM_LAYER,
            args.UPSAMPLE,
            attention_layers=args.ATTN_LAYERS,
        ).to(device)
        network_tag = 'denseunet'

    elif model_name == 'resunet':
        from resunet_configs import get_config
        import ResUnet
        args = get_config()
        model = ResUnet.ResUnet(
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
        network_tag = 'resunet'

    elif model_name == 'rdbunet':
        from rdbunet_configs import get_config
        import RDBUnet
        args = get_config()
        model = RDBUnet.RDBUnet(
            args.IN_CHANNELS,
            args.OUT_CHANNELS,
            args.BASE_CHANNELS,
            args.DEPTH,
            args.GROWTH_RATE,
            args.NUM_CONVS,
            args.ACT_TYPE,
            args.NORM_LAYER,
            args.UPSAMPLE,
        ).to(device)
        network_tag = 'rdbunet'

    elif model_name == 'unet':
        from unet_configs import get_config
        import Unet
        args = get_config()
        model = Unet.Unet(
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
        network_tag = 'unet'

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device), network_tag, args
