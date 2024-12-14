import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets import load_dataset
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from flash_attn import flash_attn_func
from torch.nn import LayerNorm
import einops

USE_CUDA = torch.cuda.is_available()
MIXED_PRECISION = True
# Optimized for RTX 4070 12GB VRAM
BATCH_SIZE = 8
NUM_WORKERS = 4

class FlashAttentionBlock(torch.nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = LayerNorm(dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        
        x = self.norm(x)
        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, 'b n (three h d) -> three b h n d', 
                              three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        x = flash_attn_func(q, k, v, dropout_p=0.0 if self.training else 0.0)
        x = einops.rearrange(x, 'b h n d -> b n (h d)')
        
        x = self.proj(x)
        x = self.dropout(x)
        
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class EfficientConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # Optimized for RTX 4070's tensor cores
        self.depthwise = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            stride=stride, padding=kernel_size//2, groups=in_channels, bias=False
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        self.norm = torch.nn.GroupNorm(
            num_groups=1, num_channels=out_channels
        )
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x)

class OptimizedResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super().__init__()
        self.conv1 = EfficientConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = EfficientConvBlock(out_channels, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = FlashAttentionBlock(out_channels)
        
        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, 
                stride=stride, bias=False
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        if self.is_bottleneck:
            residual = self.shortcut(x)
        
        out += residual
        return out

class OptimizedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )

        # Encoder - use attention only in deeper layers for efficiency
        self.conv_down_1 = OptimizedResBlock(1, 8, use_attention=False)
        self.conv_down_2 = OptimizedResBlock(8, 16, use_attention=False)
        self.conv_down_3 = OptimizedResBlock(16, 32, use_attention=True)
        self.conv_down_4 = OptimizedResBlock(32, 64, use_attention=True)

        # Middle
        self.conv_middle = OptimizedResBlock(64, 64, use_attention=True)

        # Decoder
        self.conv_up_4 = OptimizedResBlock(128, 32, use_attention=True)
        self.conv_up_3 = OptimizedResBlock(64, 16, use_attention=True)
        self.conv_up_2 = OptimizedResBlock(32, 8, use_attention=False)
        self.conv_up_1 = OptimizedResBlock(16, 1, use_attention=False)

    def forward(self, x):
        # Encoder with selective checkpointing
        conv1 = self.conv_down_1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.conv_down_2(x)
        x = self.maxpool(conv2)
        
        # Checkpoint only deeper layers
        conv3 = torch.utils.checkpoint.checkpoint(self.conv_down_3, x)
        x = self.maxpool(conv3)
        
        conv4 = torch.utils.checkpoint.checkpoint(self.conv_down_4, x)
        x = self.maxpool(conv4)
        
        x = torch.utils.checkpoint.checkpoint(self.conv_middle, x)
        
        # Decoder
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up_4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up_3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up_2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up_1(x)
        
        return torch.sigmoid(x)

class OptimizedDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        self.dataset = load_dataset("muchlisinadi/lung_dataset")[split]
        
    @torch.compile
    def normalize(self, data):
        data_max = data.max()
        data_min = data.min()
        if data_min != data_max:
            data = ((data - data_min) / (data_max - data_min))
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        x = np.array(item['image'])
        y = np.array(item['label'])
        
        if len(x.shape) == 2:
            x = x[np.newaxis, ...]
        if len(y.shape) == 2:
            y = y[np.newaxis, ...]
            
        x = self.normalize(torch.from_numpy(x).float())
        y = self.normalize(torch.from_numpy(y).float())
        
        return x, y

def train():
    # Create model and move to GPU
    model = OptimizedUNet()
    if USE_CUDA:
        model = model.cuda()
    
    # Enable torch.compile with the new inductor backend
    model = torch.compile(model, mode="max-autotune")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )
    
    # Efficient data loading
    train_dataset = OptimizedDataset(split='train')
    test_dataset = OptimizedDataset(split='test')
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=100,
        steps_per_epoch=len(data_loader_train)
    )
    
    scaler = GradScaler() if MIXED_PRECISION else None
    
    # Training metrics
    metrics = {}
    for stage in ['train', 'test']:
        for metric in ['loss']:
            metrics[f'{stage}_{metric}'] = []
            
    for epoch in range(1, 100):
        for data_loader in [data_loader_train, data_loader_test]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            
            stage = 'train'
            torch.set_grad_enabled(True)
            model.train()
            if data_loader == data_loader_test:
                stage = 'test'
                torch.set_grad_enabled(False)
                model.eval()

            for x, y in tqdm(data_loader):
                if USE_CUDA:
                    x = x.cuda()
                    y = y.cuda()
                
                with autocast(enabled=MIXED_PRECISION):
                    y_prim = model(x)
                    loss = F.binary_cross_entropy(y_prim, y)
                
                metrics_epoch[f'{stage}_loss'].append(loss.item())
                
                if stage == 'train':
                    if MIXED_PRECISION:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                
                # Move data back to CPU for visualization
                if stage == 'test':
                    loss = loss.cpu()
                    y_prim = y_prim.cpu()
                    x = x.cpu()
                    y = y.cpu()

            # Print metrics and update plots
            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')
            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

            # Visualization during test phase
            if stage == 'test':
                plt.clf()
                plt.subplot(121)
                plts = []
                c = 0
                for key, value in metrics.items():
                    plts += plt.plot(value, f'C{c}', label=key)
                    c += 1
                plt.legend(plts, [it.get_label() for it in plts])

                # Show sample predictions
                for i, j in enumerate([4, 5, 6, 16, 17, 18]):
                    plt.subplot(4, 6, j)
                    plt.title('Ground Truth')
                    plt.imshow(x[i][0].numpy(), cmap='Greys', interpolation=None)
                    plt.imshow(y[i][0].numpy(), cmap='Reds', alpha=0.5, interpolation=None)
                    
                    plt.subplot(4, 6, j+6)
                    plt.title('Prediction')
                    y_pred = y_prim[i][0].numpy()
                    plt.imshow(x[i][0].numpy(), cmap='Greys', interpolation=None)
                    plt.imshow(np.where(y_pred > 0.8, y_pred, 0), cmap='Reds', alpha=0.5, interpolation=None)

                plt.tight_layout(pad=0.5)
                plt.show()

if __name__ == "__main__":
    train()
