import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms
from PIL import Image
import numpy as np

# ==============================================================================
# 1. 模型组件定义 (保持与训练代码完全一致) [cite: 211]
# ==============================================================================

# --- Mamba 核心组件 (包含回退机制) ---
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, n_layers=1):
        super().__init__()
        if HAS_MAMBA:
            self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=d_state, expand=2) for _ in range(n_layers)])
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2, batch_first=True)
            self.layers = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(n_layers)])
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return x

# --- CNN 基础块 ---
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                   nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(cin, cout)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, cin, skip_ch, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(cin + skip_ch, cout)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]: x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
        return self.conv(torch.cat([x, skip], dim=1))

# --- 特色 Mamba/FiLM 模块 ---
class SpatialMambaBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, n_layers=1):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, out_ch or in_ch, kernel_size=1)
        self.mamba = MambaBlock(d_model=out_ch or in_ch, n_layers=n_layers)
    def forward(self, x):
        B, C, H, W = x.shape
        seq = rearrange(self.proj_in(x), "b c h w -> b (h w) c")
        return rearrange(self.mamba(seq), "b (h w) c -> b c h w", h=H, w=W)

class TemporalMambaBlock(nn.Module):
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.mamba = MambaBlock(d_model=d_model, n_layers=n_layers)
    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x = rearrange(x_seq, "b t c h w -> (b h w) t c")
        out = self.mamba(x)[:, -1, :] # 提取动态末帧特征
        return rearrange(out, "(b h w) c -> b c h w", b=B, h=H, w=W)

class FiLMGenerator(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.film_gen = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=1), nn.ReLU(),
                                       nn.Conv2d(num_features, 2 * num_features, kernel_size=1))
    def forward(self, target_seq, condition):
        params = self.film_gen(condition)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return target_seq * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

# ==============================================================================
# 2. 核心架构集成 (BmodeSmmFilmCEUSTmmFusion) [cite: 169]
# ==============================================================================

class ST_SAMamba_Inference(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, bott_ch=256):
        super().__init__()
        # Encoders & Tails [cite: 170]
        self.b_enc = nn.Sequential(ConvBlock(in_ch, base_ch), Down(base_ch, base_ch*2), Down(base_ch*2, base_ch*4))
        self.c_enc = nn.Sequential(ConvBlock(in_ch, base_ch), Down(base_ch, base_ch*2), Down(base_ch*2, base_ch*4))
        self.b_tail = nn.Sequential(Down(base_ch*4, bott_ch), Down(bott_ch, bott_ch))
        self.c_tail = nn.Sequential(Down(base_ch*4, bott_ch), Down(bott_ch, bott_ch))
        
        # Interaction & Mamba [cite: 171]
        self.b_spatial_mamba = SpatialMambaBlock(in_ch=bott_ch)
        self.film_layer = FiLMGenerator(num_features=bott_ch)
        self.fusion_proj = nn.Sequential(nn.Conv2d(bott_ch*2, bott_ch, 1), nn.BatchNorm2d(bott_ch), nn.ReLU())
        self.c_temporal_mamba = TemporalMambaBlock(d_model=bott_ch)
        
        # Decoder & Cls Head [cite: 171]
        self.decoder = nn.Sequential(nn.Conv2d(bott_ch, base_ch, 1), nn.Upsample(scale_factor=16, mode='bilinear'), nn.Conv2d(base_ch, 1, 1))
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bott_ch, 1))

    def forward(self, ceus_seq, bmode_img):
        B, T = ceus_seq.shape[0], ceus_seq.shape[1]
        # B-mode 特征提取 [cite: 172]
        x3_B = self.b_enc(bmode_img)
        bottleneck_B = self.b_tail(x3_B)
        feat_b_spatial = self.b_spatial_mamba(bottleneck_B)

        # CEUS 序列处理 [cite: 172]
        ceus_2d = rearrange(ceus_seq, "b t c h w -> (b t) c h w")
        x3_C_seq = rearrange(self.c_enc(ceus_2d), "(b t) c h w -> b t c h w", b=B, t=T)
        bottleneck_C_seq = rearrange(self.c_tail(rearrange(x3_C_seq, "b t c h w -> (b t) c h w")), "(b t) c h w -> b t c h w", b=B, t=T)

        # FiLM 调制与时空融合 [cite: 173]
        feat_c_modulated = self.film_layer(bottleneck_C_seq, feat_b_spatial)
        feat_b_expanded = feat_b_spatial.unsqueeze(1).repeat(1, T, 1, 1, 1)
        fused_seq = self.fusion_proj(rearrange(torch.cat([feat_c_modulated, feat_b_expanded], dim=2), "b t c h w -> (b t) c h w"))
        feat_final = self.c_temporal_mamba(rearrange(fused_seq, "(b t) c h w -> b t c h w", b=B, t=T))

        # 多任务输出 [cite: 173]
        return self.cls_head(feat_final), self.decoder(feat_b_spatial)

# ==============================================================================
# 3. 推理接口封装 [cite: 211]
# ==============================================================================

class STMambaPredictor:
    def __init__(self, weight_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ST_SAMamba_Inference().to(self.device)
        # 加载“模型灵魂” [cite: 211]
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, bmode_path, ceus_frame_paths):
        """
        输入: B-mode路径, CEUS帧路径列表
        输出: 恶性概率, 分割掩膜(numpy) 
        """
        # 1. 预处理
        bmode = self.transform(Image.open(bmode_path).convert('RGB')).unsqueeze(0).to(self.device)
        ceus_frames = [self.transform(Image.open(p).convert('RGB')) for p in ceus_frame_paths]
        ceus_seq = torch.stack(ceus_frames).unsqueeze(0).to(self.device)

        # 2. 推理
        cls_out, seg_out = self.model(ceus_seq, bmode)
        
        # 3. 结果解析
        prob = torch.sigmoid(cls_out).item()
        mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
        return prob, mask