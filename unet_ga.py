import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# 用於提供模型一種「知道時間在哪裡」的方式
def sinusoidal_embedding(t, dim, device):
    """
    生成正弦嵌入表示
    t: [batch_size]，時間步長
    dim: 嵌入維度（例如 32）
    返回: [batch_size, dim] 的嵌入張量
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None].float() * emb[None, :]  # [batch_size, half_dim]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [batch_size, dim]
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),  # 或 nn.SiLU()
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if not self.same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class DownBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        skip1 = self.res1(x)    # 第一層 skip output
        skip2 = self.res2(skip1)  # 第二層 skip output
        out = skip2
        return out, skip1, skip2  # 回傳所有你需要的 skip 給對應的 UpBlock
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        skip1 = self.res1(x)    # 第一層 skip output
        skip2 = self.res2(skip1)  # 第二層 skip output
        out = self.pool(skip2)  # pool 後的輸出
        return out, skip1, skip2  # 回傳所有你需要的 skip 給對應的 UpBlock
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, skip1_channels, skip2_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.res1 = ResidualBlock(in_channels + skip1_channels, out_channels)
        self.res2 = ResidualBlock(out_channels + skip2_channels, out_channels)

    # def forward(self, x, skip1, skip2):
    #     x = self.upsample(x)
    #     x = torch.cat([skip1, x], dim=1)
    #     x = self.res1(x)
    #     x = torch.cat([skip2, x], dim=1)
    #     x = self.res2(x)
    #     return x

    def forward(self, x, skip1, skip2):
        x = self.upsample(x)

        # 保證上採樣後與 skip1 對齊
        if x.shape[2:] != skip1.shape[2:]:
            x = F.interpolate(x, size=skip1.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([skip1, x], dim=1)

        x = self.res1(x)

        # 保證與 skip2 對齊
        if x.shape[2:] != skip2.shape[2:]:
            x = F.interpolate(x, size=skip2.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([skip2, x], dim=1)
        x = self.res2(x)

        return x
    
class SkipConnection(nn.Module):
    def __init__(self, mode="concat"):
        super(SkipConnection, self).__init__()
        self.mode = mode

    def forward(self, encoder_feat, decoder_feat):
        if self.mode == "concat":
            return torch.cat([decoder_feat, encoder_feat], dim=1)  # channel 方向 concat
        elif self.mode == "add":
            return decoder_feat + encoder_feat
        else:
            raise ValueError("Unsupported skip connection mode")
        
class CrossAttention2D(nn.Module):
    def __init__(self, query_channels, cond_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_channels // num_heads
        assert query_channels % num_heads == 0, "query_channels must be divisible by num_heads"
        
        # 線性投影
        self.to_q = nn.Conv2d(query_channels, query_channels, kernel_size=1)
        self.to_k = nn.Conv2d(cond_channels, query_channels, kernel_size=1)  # key 與 query 同維度
        self.to_v = nn.Conv2d(cond_channels, query_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(query_channels, query_channels, kernel_size=1)

    def forward(self, x, cond):
        B, C, H, W = x.shape
        _, Cc, Hc, Wc = cond.shape

        # flatten 2D -> sequence
        q = self.to_q(x).view(B, self.num_heads, self.head_dim, H*W)  # [B, heads, head_dim, HW]
        k = self.to_k(cond).view(B, self.num_heads, self.head_dim, Hc*Wc)
        v = self.to_v(cond).view(B, self.num_heads, self.head_dim, Hc*Wc)

        # transpose for matmul: [B, heads, HW, head_dim]
        q = q.permute(0,1,3,2)
        k = k.permute(0,1,3,2)
        v = v.permute(0,1,3,2)

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)  # [B, heads, HW, HW_cond]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]
        out = out.permute(0,1,3,2).contiguous().view(B, C, H, W)  # [B, C, H, W]

        out = self.out_proj(out)
        return out + x  # residual connection

# U-Net 模型 (與圖 6 對應)
class UNet(nn.Module):
    def __init__(self, config):
        """
        使用 config 物件來動態建立 U-Net.
        """
        super(UNet, self).__init__()
        
        self.config = config
        self.out_channels = config["out_channels"]
        self.time_dim = config["time_dim"]
        
        # 時間嵌入層
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim)
        )

        # 初始卷積
        init_channels = config["base_channels"]
        self.init_conv = nn.Conv2d(config["in_channels"] + config["cond_channels"], init_channels, kernel_size=3, padding=1)
        
        self.time_proj = nn.Linear(self.time_dim, init_channels)

        # 動態建立下採樣層
        self.down_blocks = nn.ModuleList()
        current_channels = init_channels
        ch_mults = config["channel_mults"]
        
        for i in range(config["depth"]):
            out_ch = init_channels * ch_mults[i]
            self.down_blocks.append(
                nn.Sequential(
                    ResidualBlock(current_channels, out_ch),
                    ResidualBlock(out_ch, out_ch)
                )
            )
            current_channels = out_ch
        
        self.down_pool = nn.AvgPool2d(2, 2)

        self.mid_block1 = ResidualBlock(current_channels, current_channels)
        self.mid_attn = CrossAttention2D(query_channels=current_channels, cond_channels=config["cond_channels"])
        self.mid_block2 = ResidualBlock(current_channels, current_channels)
        
        # 動態建立上採樣層
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(config["depth"])):
            in_ch = init_channels * ch_mults[i]
            out_ch = init_channels * ch_mults[i-1] if i > 0 else init_channels
            
            self.up_blocks.append(
                nn.Sequential(
                    ResidualBlock(in_ch * 2, in_ch),
                    ResidualBlock(in_ch, out_ch)
                )
            )
        
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.final_conv = nn.Conv2d(init_channels, config["out_channels"], kernel_size=1)

    @staticmethod
    def resize_to_match(source_tensor, target_tensor):
        target_h, target_w = target_tensor.shape[2], target_tensor.shape[3]
        return F.interpolate(source_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def forward(self, x, cond, t, beta):
        B, C, H, W = x.shape
        device = x.device
        
        t_emb = sinusoidal_embedding(t, self.time_dim, device)
        t_emb = self.time_embed(t_emb)
        
        x_in = torch.cat([x, cond], dim=1)
        x_in = self.init_conv(x_in)

        time_bias = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x_in = x_in + time_bias
        
        skips = []
        # 下採樣
        for i, block in enumerate(self.down_blocks):
            x_in = block(x_in)
            skips.append(x_in)
            # <<< 關鍵修改點: 增加池化前的檢查 >>>
            if i < len(self.down_blocks) - 1:
                # 只有當 H 和 W 都大於1時，才進行池化
                if x_in.shape[2] > 1 and x_in.shape[3] > 1:
                    x_in = self.down_pool(x_in)
            
        x_in = self.mid_block1(x_in)
        # cond_resized = F.interpolate(cond, size=x_in.shape[2:], mode='bilinear', align_corners=False)
        # x_in = self.mid_attn(x_in, cond_resized)
        x_in = self.mid_block2(x_in)
        
        # 上採樣
        for i, block in enumerate(self.up_blocks):
            skip_conn = skips.pop() # 注意：這裡的 pop 順序要和 append 對應

            # 在上採樣前，先檢查當前 x_in 的尺寸是否小於 skip_conn
            # 如果是，就需要先對 x_in 進行上採樣
            if x_in.shape[2] < skip_conn.shape[2] or x_in.shape[3] < skip_conn.shape[3]:
                 x_in = F.interpolate(x_in, size=skip_conn.shape[2:], mode='bilinear', align_corners=False)
            
            x_in = torch.cat([x_in, skip_conn], dim=1)
            x_in = block(x_in)

        output = self.final_conv(x_in)

        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output
# 訓練函數 (Algorithm 1)
import torch
import torch.nn as nn
from unet_ga import UNet # 使用我們可配置的 UNet
import torch.optim as optim
from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std
from torch.utils.data import DataLoader
# from unet import train # 我們需要修改原有的 train 函數，所以這裡不再直接導入

# 引入計算指標的庫
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os # 用於創建儲存目錄
from torch.utils.checkpoint import checkpoint # 引入梯度檢查點

# =========================================================
# <<< 沿用並修改原有的 train 函數以包含更多指標 >>>
# =========================================================
import torch
import torch.nn as nn
# 假設 unet_ga 包含了你的 UNet 模型定義
# from unet_ga import UNet 
import torch.optim as optim
# from data_utils import LazyWeatherDataset, prepare_file_list, compute_mean_std
from torch.utils.data import DataLoader

# 引入計算指標的庫
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os # 用於創建儲存目錄
from torch.utils.checkpoint import checkpoint # 引入梯度檢查點

def train(model, train_loader, num_epochs, device,
          optimizer, criterion, train_loss_history,
          use_checkpoint=False,
          checkpoint_dir="./checkpoints", # 添加一個 checkpoint_dir 參數
          beta=None, alpha=None, alpha_cumprod=None):

    if use_checkpoint and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 如果沒有外部傳入，才自己生成
    if beta is None or alpha is None or alpha_cumprod is None:
        timesteps = 1000 # 這裡的 timesteps 應該和前面定義的 alpha_cumprod 長度一致
        beta_start = 1e-4
        beta_end = 0.02
        beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

    beta = beta.to(device)
    alpha = alpha.to(device)
    alpha_cumprod = alpha_cumprod.to(device)

    # 擴展 history 記錄
    rmse_history = []
    ssim_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        running_ssim = 0.0
        num_batches = 0 # 用於計算平均值

        # <<< 注意這裡的迭代，應該是 cond_data, target_data，而不是 time >>>
        # 如果你的 LazyWeatherDataset 確實返回了第三個元素，那麼這裡就需要修改
        # 為了和您之前的代碼一致，這裡暫時保留 `time`，但如果 `LazyWeatherDataset` 不返回，會報錯
        # 更好的做法是： for cond_data, target_data in train_loader:
        for cond_data, target_data, time in train_loader: # 假設 LazyWeatherDataset 只返回 cond_data 和 target_data
            cond_data = cond_data.to(device, non_blocking=True)
            target_data = target_data.to(device, non_blocking=True)

            # ====== 關鍵步驟：展平成 (B, T*V, H, W) ======
            # 注意：這裡的 H, W 應該是從 cond_data 得到的，但 target_data 可能有不同通道數
            # B, T_c, V_c, H, W = cond_data.shape # H, W 應該是圖像的空間維度
            # cond_data_flat = cond_data.view(B, T_c * V_c, H, W)
            # 簡化為直接使用 target_data 的 H, W
            B, T_t, V_t, H, W = target_data.shape 
            
            cond_data_flat = cond_data.view(B, cond_data.shape[1] * cond_data.shape[2], H, W) # 根據實際的 cond_data shape 調整
            target_data_flat = target_data.view(B, T_t * V_t, H, W)

            batch_size = cond_data_flat.shape[0]
            # 這裡的 1000 應該是 timesteps 的長度
            t = torch.randint(0, len(beta), (batch_size,), device=device) # 確保時間步數與 beta 長度一致

            noise = torch.randn_like(target_data_flat)
            alpha_cumprod_t = alpha_cumprod[t][:, None, None, None] # 擴展維度以便廣播
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            x_t = sqrt_alpha_cumprod_t * target_data_flat + sqrt_one_minus_alpha_cumprod_t * noise

            optimizer.zero_grad(set_to_none=True)
            if use_checkpoint:
                # 傳遞 beta 參數給模型
                output = checkpoint(model, x_t, cond_data_flat, t, beta, use_reentrant=False)
            else:
                # 傳遞 beta 參數給模型
                output = model(x_t, cond_data_flat, t, beta)

            loss = criterion(output, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ======= 計算 RMSE 和 SSIM 指標 =======
            # 從預測的噪音中估計原始數據 (一步去噪的近似)
            # x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * output) / sqrt_alpha_cumprod_t
            
            # 將數據移回 CPU 並轉換為 NumPy 進行計算
            # detach() 是必要的，避免梯度計算影響指標
            # cpu() 是必要的，因為 skimage 和 sklearn 通常在 CPU 上運行
            # numpy() 是必要的，因為 skimage 和 sklearn 需要 numpy 數組
            target_data_np = target_data_flat.detach().cpu().numpy()
            predicted_x0_np = ((x_t - sqrt_one_minus_alpha_cumprod_t * output) / sqrt_alpha_cumprod_t).detach().cpu().numpy()

            # 計算每個樣本的 RMSE 和 SSIM
            batch_rmse = 0.0
            batch_ssim = 0.0
            
            for k in range(batch_size): # 遍歷 batch 中的每個樣本
                # RMSE: 將多通道圖像展平為一維來計算 MSE
                rmse_val = sqrt(mean_squared_error(target_data_np[k].flatten(), predicted_x0_np[k].flatten()))
                batch_rmse += rmse_val
                
                # SSIM: 對每個通道獨立計算，然後取平均
                ssim_per_sample_channels = []
                num_flat_channels = target_data_np.shape[1] # T_t * V_t
                
                if num_flat_channels > 0:
                    for c in range(num_flat_channels):
                        img1 = target_data_np[k, c, :, :] # 取出單個通道圖像
                        img2 = predicted_x0_np[k, c, :, :]
                        
                        # 確保 SSIM 計算時的 data_range 正確
                        # 使用每個圖像對的最大值和最小值來確定範圍
                        # 由於數據是標準化的，我們需要確保這個範圍能覆蓋實際數據
                        # 這裡使用一個較為保守的估計，確保 data_range 非零且足夠大
                        current_data_min = min(np.min(img1), np.min(img2))
                        current_data_max = max(np.max(img1), np.max(img2))
                        current_data_range = current_data_max - current_data_min
                        
                        if current_data_range <= 1e-6: # 避免除以零或非常小的數
                            ssim_per_sample_channels.append(1.0) # 如果數據幾乎相同，則 SSIM 為 1
                        else:
                            # multichannel=False，因為我們一個通道一個通道處理
                            ssim_per_sample_channels.append(ssim(img1, img2, data_range=current_data_range, channel_axis=None))
                    
                    batch_ssim += (np.mean(ssim_per_sample_channels) if ssim_per_sample_channels else 0.0)
            
            running_rmse += batch_rmse / batch_size # 平均每個樣本的 RMSE
            running_ssim += batch_ssim / batch_size # 平均每個樣本的 SSIM
            num_batches += 1

        avg_loss = running_loss / num_batches
        avg_rmse = running_rmse / num_batches
        avg_ssim = running_ssim / num_batches

        train_loss_history.append(avg_loss)
        rmse_history.append(avg_rmse)
        ssim_history.append(avg_ssim)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, RMSE: {avg_rmse:.4f}, SSIM: {avg_ssim:.4f}")

        if use_checkpoint and (epoch + 1) % 5 == 0: # 每 5 個 epoch 儲存一次檢查點
            checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "rmse": avg_rmse,
                "ssim": avg_ssim,
                "config": model.config # 儲存模型設定 (假設 model 有 config 屬性)
            }, checkpoint_path)
            print(f"檢查點已儲存至 {checkpoint_path}")

    return train_loss_history, rmse_history, ssim_history, beta.cpu(), alpha.cpu(), alpha_cumprod.cpu()


@torch.no_grad()
def ddim_inference(model, cond, beta, device, eta=0.0, num_steps=15):
    alpha = 1.0 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0).to(device)

    B, T_c, V_c, H, W = cond.shape
    cond_flat = cond.view(B, T_c * V_c, H, W)
    x_t = torch.randn(B, model.out_channels, H, W, device=device)

    total_timesteps = len(beta)
    step_size = total_timesteps // num_steps
    timesteps = list(range(0, total_timesteps, step_size))[::-1]

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        pred_noise = model(x_t, cond_flat, t_tensor, beta)  # 如果 UNet 需要 beta

        alpha_t = alpha_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        x0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

        if i < len(timesteps) - 1:
            t_next = timesteps[i+1]
            alpha_next = alpha_cumprod[t_next]
            sigma_t = eta * torch.sqrt((1 - alpha_next)/(1 - alpha_t) * (1 - alpha_t / alpha_next))
            noise = sigma_t * torch.randn_like(x_t)

            x_t = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next - sigma_t**2) * pred_noise + noise
        else:
            x_t = x0_pred

    return x_t