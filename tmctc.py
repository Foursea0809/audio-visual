
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 視覺前端（Video Front-end）
# =====================================================================
class VideoFrontEnd(nn.Module):
    """
    負責接收經前處理裁切為 112x112 的灰階嘴唇影像序列，
    透過 3D 卷積與 2D ResNet 轉化為每影格 512 維度的特徵向量。
    """
    def __init__(self):
        super(VideoFrontEnd, self).__init__()
        # 3D 卷積層：專門用來捕捉連續影格間的「時空（Spatiotemporal）」動態特徵
        # 輸入形狀: (Batch, Channels=1, Frames, H=112, W=112)
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # 2D ResNet 骨幹網絡層 (此處以簡化版 ResNet 架構示範，確保輸出降維至 512 維)
        self.resnet2d = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，將空間維度壓縮為 1x1
        )

    def forward(self, x):
        # 預期輸入 x: (Batch, 1, Video_Frames, 112, 112)
        x = self.conv3d(x)  # 輸出: (Batch, 64, Video_Frames, H_out, W_out)
        
        B, C, F_len, H, W = x.size()
        # 為了餵入 2D ResNet，必須將 Batch 與 Video_Frames 維度合併
        x = x.transpose(1, 2).contiguous().view(B * F_len, C, H, W)
        
        x = self.resnet2d(x)       # 輸出: (B * F_len, 512, 1, 1)
        x = x.view(B, F_len, 512)  # 還原維度，輸出為每幀 512 維度的特徵向量
        return x


# =====================================================================
# 2. 多模態融合與 Backbone（Transformer Encoder Stack）
# =====================================================================
class AVTransformerBackbone(nn.Module):
    """
    建構基於 Transformer 的堆疊架構（包含 Self-Attention 與 Feedforward 模組），
    將時序不對等的音訊特徵與視覺特徵進行對齊與多模態融合。
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, audio_feat_dim=80):
        super(AVTransformerBackbone, self).__init__()
        # 音訊特徵投影層 (假設輸入的音訊 STFT 聲學特徵維度為 80)
        self.audio_project = nn.Linear(audio_feat_dim, d_model)
        
        # 多模態串聯融合層 (將 512維視訊 + 512維音訊 = 1024維 投影回 512維)
        self.fusion_projection = nn.Linear(d_model * 2, d_model)
        
        # 論文核心 Backbone：由多層自注意力與前饋層組成的 Transformer 編碼器堆疊
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True  # 設定 Batch 在第一維度，完美相容影像序列
        )
        self.transformer_backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, video_feats, audio_features):
        # video_feats 形狀: (Batch, T_video, 512)
        # audio_features 形狀: (Batch, T_audio, 80)
        
        # 將音訊特徵投影至 512 維
        audio_feats = self.audio_project(audio_features)  # (Batch, T_audio, 512)
        
        # 💡 遵循論文關鍵規格：25 fps 影片中每 1 幀影像對應 4 幀音訊特徵（10ms 一幀）
        # 使用 repeat_interleave 將視覺特徵在時間維度（dim=1）複製 4 倍以對齊音訊頻率
        video_feats_aligned = torch.repeat_interleave(video_feats, repeats=4, dim=1)  # (Batch, T_audio_aligned, 512)
        
        # 預防因四捨五入產生的微小長度不一致，強制裁切至相同時間長度
        min_time_len = min(video_feats_aligned.size(1), audio_feats.size(1))
        video_feats_aligned = video_feats_aligned[:, :min_time_len, :]
        audio_feats = audio_feats[:, :min_time_len, :]
        
        # 將兩個模態的特徵在特徵軸（維度 -1）進行 Concatenation 串聯
        fused_concat = torch.cat((video_feats_aligned, audio_feats), dim=-1)  # (Batch, min_time_len, 1024)
        
        # 投影回 512 維度以符合 Transformer 輸入規格
        transformer_input = self.fusion_projection(fused_concat)  # (Batch, min_time_len, 512)
        
        # 送入基於自注意力機制的 Transformer Backbone
        backbone_output = self.transformer_backbone(transformer_input)  # (Batch, min_time_len, 512)
        return backbone_output


# =====================================================================
# 3. CTC 損失層（Connectionist Temporal Classification Layer）
# =====================================================================
class CTCLossLayer(nn.Module):
    """
    後端連接分類映射，並將輸出轉化為符合 PyTorch nn.CTCLoss 格式的對數機率值。
    """
    def __init__(self, d_model=512, num_classes=40):
        super(CTCLossLayer, self).__init__()
        # 全連接層：將 512 維特徵投影至 40 個字元類別
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x 形狀: (Batch, Time, 512)
        logits = self.classifier(x)  # (Batch, Time, 40)
        
        # 💡 PyTorch 的 nn.CTCLoss 嚴格要求輸入形狀為 (Time, Batch, Class_Probabilities)
        # 且必須通過 Log-Softmax 計算出對數機率值
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (Time, Batch, 40)
        return log_probs


# =====================================================================
# 4. 完整端到端 TM-CTC 模型封裝
# =====================================================================
class TM_CTC_AVSR_Model(nn.Module):
    def __init__(self, num_classes=40):
        super(TM_CTC_AVSR_Model, self).__init__()
        self.video_frontend = VideoFrontEnd()
        self.transformer_backbone = AVTransformerBackbone(d_model=512)
        self.ctc_layer = CTCLossLayer(d_model=512, num_classes=num_classes)

    def forward(self, video_input, audio_input):
        """
        video_input: (Batch, 1, T_video, 112, 112) -> 嘴唇灰階視訊序列
        audio_input: (Batch, T_audio, 80)           -> 音訊 STFT 特徵向量
        """
        # 步驟 1: 視覺前端特徵萃取
        video_feats = self.video_frontend(video_input)
        
        # 步驟 2: 多模態時序對齊、融合與自注意力 Backbone 計算
        backbone_output = self.transformer_backbone(video_feats, audio_input)
        
        # 步驟 3: 映射並輸出 CTC 對數機率
        log_probs = self.ctc_layer(backbone_output)
        return log_probs

# --- 測試模型維度相容性驗證 ---
if __name__ == "__main__":
    # 模擬一個 Batch 包含 2 個樣本
    # 影片：假設 25 影格 (1秒)，嘴唇大小 112x112
    mock_video = torch.randn(2, 1, 25, 112, 112)
    # 音訊：依照論文 1 影格影片對應 4 影格音訊，25 * 4 = 100 影格，特徵維度 80
    mock_audio = torch.randn(2, 100, 80)
    
    model = TM_CTC_AVSR_Model(num_classes=40)
    output = model(mock_video, mock_audio)
    
    print("\n=== 模型建構成功！維度測試結果 ===")
    print(f"輸入影片維度: {mock_video.shape}")
    print(f"輸入音訊維度: {mock_audio.shape}")
    print(f"CTC 輸出機率維度 (Time, Batch, Classes): {output.shape}")