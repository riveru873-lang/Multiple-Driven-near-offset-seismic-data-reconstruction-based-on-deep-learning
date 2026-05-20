# configs.py
from argparse import ArgumentParser

def get_config():
    parser = ArgumentParser(description="Seismic Network Training Configuration")

    # ========= Training Parameters =========
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--BATCH_SIZE", type=int, default=16, help="Batch size")
    parser.add_argument("--NUM_EPOCHS", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--BETA1", type=float, default=0.9, help="Adam optimizer beta1")
    parser.add_argument("--BETA2", type=float, default=0.999, help="Adam optimizer beta2")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--LAMBDA", type=float, default=0.3, help="Original weight of SSIM in loss function")

    # ========= Model Parameters =========
    parser.add_argument("--IN_CHANNELS", type=int, default=1, help="Input channels")
    parser.add_argument("--OUT_CHANNELS", type=int, default=1, help="Output channels")
    parser.add_argument("--BASE_CHANNELS", type=int, default=16, help="Base channel width")
    parser.add_argument("--DEPTH", type=int, default=4, help="Network depth")
    parser.add_argument("--ACT_TYPE", type=str, default="SiLU", help="Activation function")
    parser.add_argument("--NORM_LAYER", type=str, default="Group", help="Normalization type")
    parser.add_argument("--UPSAMPLE", type=bool, default=True, help="Whether to use upsampling")
    parser.add_argument("--DROPOUT", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--ATTN_LAYERS", type=list, default=None, help="List of attention layers (e.g., [0,1])")
    parser.add_argument("--NETWORK", type=str, default='Unet', help="network type")

    # ========= Dataset Parameters =========
    
    
    parser.add_argument("--SAMPLE_PATH", type=str, default="./data/paper-diffr-201-300-100pair-miss20/train2-shots-samples-miss20.dat", help="Path to training samples")
    parser.add_argument("--LABEL_PATH", type=str, default="./data/paper-diffr-201-300-100pair-miss20/train2-shots-labels-fullP.dat", help="Path to training labels")
    parser.add_argument("--DIM", type=tuple, default=(500, 501, 100), help="Data dimensions (nt, nx, ny)")
    parser.add_argument("--BLOCK_SIZE", type=tuple, default=(128, 128), help="Block size for patching")
    parser.add_argument("--STRIDE", type=tuple, default=(32, 64), help="Stride for patch extraction")
    '''
    
    parser.add_argument("--SAMPLE_PATH", type=str, default="./data/paper-pluto-new-131-230-100pair-miss60/train2-shots-samples-miss60.dat", help="Path to training samples")
    parser.add_argument("--LABEL_PATH", type=str, default="./data/paper-pluto-new-131-230-100pair-miss60/train2-shots-labels-fullP.dat", help="Path to training labels")
    parser.add_argument("--DIM", type=tuple, default=(1000, 400, 100), help="Data dimensions (nt, nx, ny)")
    parser.add_argument("--BLOCK_SIZE", type=tuple, default=(256, 128), help="Block size for patching")
    parser.add_argument("--STRIDE", type=tuple, default=(96, 32), help="Stride for patch extraction")
'''
    
    args = parser.parse_args(args=[])  # Jupyter
    return args
