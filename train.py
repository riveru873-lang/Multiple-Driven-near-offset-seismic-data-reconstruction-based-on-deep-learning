import sys, os, time
sys.path += ["./src", "./configs", "./models"]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from datetime import datetime

from utils import *
from Seismic_dataset3 import seismic_dataset
from torchinfo import summary

# Set global font to Times New Roman
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'

# ================== CONFIGURATION ==================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
print(torch.cuda.get_device_name())

g = torch.Generator()  # Ensure result reproducibility
g.manual_seed(42)

# ================== MODEL ==================
networks = ['unet', 'denseunet', 'resunet', 'rdbunet']
network_model = networks[2]
model, network_tag, args = create_model(network_model, device)
# Weight initialization
# model.apply(init_weights)
print(f"Using network: {network_tag}")

# Output network architecture and parameter size
summary(model)
print("Total number of parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# ================== DATASET ==================
# Split dataset into training set, validation set, and test set.
# len(train_loader) = len(train_data) / batch_size
# Each train_loader element is a 4D tensor of shape [Batch_size, 1, 64, 64], 
# with a total of len(train_data)/batch_size such elements
print("Loading dataset...")
start = time.time()
dataset = seismic_dataset(
    sample=args.SAMPLE_PATH,
    label=args.LABEL_PATH,
    dim=args.DIM,
    block_size=args.BLOCK_SIZE,
    # n_channels=args.IN_CHANNELS,
    stride=args.STRIDE,
)
'''
# dataset1.random_patch_visualization()
dataset2 = seismic_dataset(
    sample=args.SAMPLE_PATH1,
    label=args.LABEL_PATH1,
    dim=args.DIM1,
    block_size=args.BLOCK_SIZE1,
    # n_channels=args.IN_CHANNELS,
    stride=args.STRIDE1,
)
# Merge datasets
dataset = ConcatDataset([dataset1, dataset2])
'''

# Patch sample visualization
visualize_samples(dataset, aug_num=4, num_samples=1)

train_size = int(0.8 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=g)

train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
print(f"Dataset loaded: Total={len(dataset)}, Train={train_size}, Val={val_size}, Test={test_size}")
'''
patches_per_shot = len(dataset) // args.DIM[2]   # Number of patches per shot

train_shots = int(0.8 * args.DIM[2])             # 80 shots for training
val_shots = args.DIM[2] - train_shots           # 20 shots for validation

train_indices = list(range(0, train_shots * patches_per_shot))
val_indices = list(range(train_shots * patches_per_shot, len(dataset)))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False)
'''
print(f"Dataset loaded: Total={len(dataset)}, Train={len(train_dataset)}, Val={len(val_dataset)}")

end = time.time()
r = end - start
print(f"Generating Dataset takes time: {r:.6f}s")

optimizer = optim.Adam(
    model.parameters(),
    lr=args.LEARNING_RATE,
    betas=(args.BETA1, args.BETA2),
    weight_decay=args.WEIGHT_DECAY,
)
# Cosine annealing: after setting learning rate decay, the initial learning rate should be appropriately increased
# to avoid falling into local optima as the learning rate decreases with each epoch
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=args.NUM_EPOCHS,  # Maximum epoch
#     eta_min=1e-5            # Minimum learning rate
# )
# import torch.optim.lr_scheduler as lr_scheduler
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=15,           # Moderate cycle length
#     T_mult=2,         # Gradual increase
#     eta_min=5e-7,     # Very low minimum, suitable for fine-tuning
# )

criterion = nn.L1Loss(reduction='mean')

ssim = SSIM(data_range=1.0).to(device)
psnr = PSNR(data_range=1.0).to(device)

# ================== OUTPUT PATHS ==================
time_string = datetime.now().strftime("%Y-%m-%d~%Hh%Mmin")
output_dir = f"./outputs/{time_string}-{network_tag}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

# ================== TRAINING LOOP ==================
train_loss, val_loss, ssim_list, psnr_list = [], [], [], []
print("Training started...\n")
start = time.time()

# Adaptive adjustment of SSIM weight
lambda_scheduler = LambdaScheduler(initial_lambda=args.LAMBDA)

for epoch in range(args.NUM_EPOCHS):
    model.train()
    epoch_train_loss = 0.0
    for samples, labels in train_loader:
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward propagation
        out = model(samples)
        # out = mask_apply(out, labels)

        # loss = criterion(out, labels) + (1 - ssim(out, labels)) * lambda_scheduler.lambda_value
        loss = criterion(out, labels) + (1 - ssim(out, labels)) * 0.1
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # Learning rate scheduling (adjust initial learning rate)
    # scheduler.step()

    train_loss.append(epoch_train_loss / len(train_loader))

    # Validation
    model.eval()
    val_epoch_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
    with torch.no_grad():
        for samples, labels in val_loader:
            samples, labels = samples.to(device), labels.to(device)
            out = model(samples)
            # out = mask_apply(out, labels)

            # loss = criterion(out, labels) + (1 - ssim(out, labels)) * lambda_scheduler.lambda_value
            loss = criterion(out, labels) + (1 - ssim(out, labels)) * 0.1
            val_epoch_loss += loss.item()
            # Compute metrics
            val_ssim += ssim(out, labels).item()
            val_psnr += psnr(out, labels).item()
        val_loss.append(val_epoch_loss / len(val_loader))
        ssim_list.append(val_ssim / len(val_loader))
        psnr_list.append(val_psnr / len(val_loader))

        # ---- λ adjustment ----
        new_lambda = lambda_scheduler.update(ssim_list[-1])
        print(f"[Epoch {epoch+1}/{args.NUM_EPOCHS}] Train Loss: {train_loss[-1]:.5f} | Val Loss: {val_loss[-1]:.5f} | SSIM: {ssim_list[-1]:.4f} | PSNR: {psnr_list[-1]:.4f} dB")

print(f"SSIM weight: {lambda_scheduler.lambda_value:.2f}")
valstd = np.std(val_loss)
valmean = np.mean(val_loss)
corr = valstd / valmean
print(f"Validating dataset mean: {valmean:.4f} | std: {valstd:.4f} | Coefficient of variation: {corr:.2f}")
# ================== SAVE RESULTS ==================
torch.save(model.state_dict(), f"{output_dir}/model.pth")
import shutil
shutil.copy2(f"./configs/{network_model}_configs.py", os.path.join(output_dir, f"{network_model}_configs.py"))
# shutil.copy2("./models/ResUnet.py", os.path.join(output_dir, "ResUnet.py"))

print(f"✅ Configuration file copied to: {output_dir}")
plot_training_curves(train_loss, val_loss, ssim_list, psnr_list, args.NUM_EPOCHS, output_dir)

print("\nTraining completed successfully!")
end = time.time()
training_time = end - start
m, s = 0, training_time
if training_time > 60:
    m = training_time // 60
    s = round(training_time) % 60
print(f"Training takes time: {int(m)} minutes {s} seconds")
