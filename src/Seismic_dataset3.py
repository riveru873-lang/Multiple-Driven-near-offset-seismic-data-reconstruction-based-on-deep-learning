import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class seismic_dataset(Dataset):

    def __init__(self, sample, label, dim, block_size, stride=(16, 8), augment=True):
        self.sample_file = sample
        self.label_file = label
        self.dim = dim  # (H, W, S)
        self.block_size = block_size
        self.stride = stride
        self.augment = augment

        self.sample, self.label = self.load_data_pair()
        self.patches = self.build_patches()

    def load_data_pair(self):
        sample = np.fromfile(self.sample_file, dtype=np.float32)
        label = np.fromfile(self.label_file, dtype=np.float32)

        sample = sample.reshape((self.dim[2], self.dim[1], self.dim[0])).transpose(2, 1, 0)
        label = label.reshape((self.dim[2], self.dim[1], self.dim[0])).transpose(2, 1, 0)

        # Unified normalization (very important)
        scale = np.max(np.abs(label))
        sample = sample / scale
        label = label / scale

        return sample, label

    def build_patches(self):
        H, W, S = self.dim
        win_h, win_w = self.block_size
        stride_h, stride_w = self.stride

        i_range = list(range(0, H - win_h, stride_h))
        j_range = list(range(0, W - win_w, stride_w))

        if i_range[-1] + stride_h > H - win_h:
            i_range.append(H - win_h)  # Last valid starting position

        if j_range[-1] + stride_w > W - win_w:
            j_range.append(W - win_w)  # Last valid starting position

        patches = []

        for s in range(S):
            for i in i_range:
                for j in j_range:
                    x = self.sample[i:i + win_h, j:j + win_w, s]
                    y = self.label[i:i + win_h, j:j + win_w, s]

                    patches.append((x, y))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx]

        if self.augment:
            x, y = self.augment_pair(x, y)

        x = torch.from_numpy(x).unsqueeze(0).float()
        y = torch.from_numpy(y).unsqueeze(0).float()

        return x, y

    def augment_pair(self, x, y):
        # Horizontal flip (trace direction)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)

        # Time flip (low probability)
        if np.random.rand() > 0.8:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)

        # Add noise (only to input)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise

        return x.copy(), y.copy()

    def random_patch_visualization(self, save_path="patches.png"):
        # Randomly select four patch indices
        idxs = np.random.choice(len(self.patches), 4, replace=False)

        # Create a 1x4 subplot grid
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        for i, idx in enumerate(idxs):
            x, y, i_start, j_start = self.patches[idx]

            # Select which subplot to display
            ax = axs[i]

            # Compute trace range (horizontal) and time range (vertical)
            trace_range = (j_start, j_start + self.block_size[1])
            time_range = (i_start * self.sample_rate, (i_start + self.block_size[0]) * self.sample_rate)

            # Display sample and label (grayscale)
            ax.imshow(x, cmap='gray', aspect='auto')  # Use grayscale colormap
            ax.imshow(y, cmap='coolwarm', alpha=0.5, aspect='auto')  # Label semi-transparent

            # Add label
            ax.text(0.1, 0.1, f'{chr(97 + i)}', color='white', fontsize=12, ha='center', va='center')

            ax.set_xlabel('Trace number')
            ax.set_ylabel('Time (s)')

            # Set x-axis (trace range) and y-axis (time range) ticks
            ax.set_xticks(np.linspace(0, x.shape[1], 5))
            ax.set_yticks(np.linspace(0, x.shape[0], 5))
            ax.set_xticklabels(np.linspace(trace_range[0], trace_range[1], 5).astype(int))
            ax.set_yticklabels(np.linspace(time_range[0], time_range[1], 5).astype(float))

            # Ensure ticks point inward
            ax.tick_params(axis='both', direction='in')

        # Save image as grayscale
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Image saved to {save_path}")
