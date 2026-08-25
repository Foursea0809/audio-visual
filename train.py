import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# 導入核心模型與資料載入器
from tmctc import TM_CTC_AVSR_Model
from dataset_loader import AVSRDataset, avsr_collate_fn

# =====================================================================
# 1. 字典、CTC 解碼與【純 Python 內建標準 WER 演算法】
# =====================================================================
VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 "
ID_TO_CHAR = {i + 1: ch for i, ch in enumerate(VOCAB)}

def ctc_greedy_decode(log_probs):
    """
    CTC 貪婪解碼器：提取機率最高字元，過濾連續重複 Token 與 Blank (0)
    """
    preds = torch.argmax(log_probs, dim=-1).transpose(0, 1)  # (Batch, Time_Steps)
    decoded_sentences = []
    for pred in preds:
        pred_list = pred.tolist()
        char_list = []
        prev_idx = -1
        for idx in pred_list:
            if idx != prev_idx and idx != 0:
                char_list.append(ID_TO_CHAR.get(idx, ''))
            prev_idx = idx
        decoded_text = "".join(char_list).strip()
        decoded_sentences.append(decoded_text)
    return decoded_sentences

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    純 Python 實作動態規劃 (Levenshtein Distance) 單句字錯率
    100% 絕對防崩潰：相容空字串、全空格、無辨識字元等所有邊界情況
    """
    ref_words = str(reference).strip().split()
    hyp_words = str(hypothesis).strip().split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    if len(hyp_words) == 0 or hypothesis == "<空白/未辨識出字元>":
        return 1.0
        
    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
        
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dlt = dp[i - 1][j] + 1
                dp[i][j] = min(sub, ins, dlt)
                
    distance = dp[r_len][h_len]
    return min(distance / r_len, 1.0)

# =====================================================================
# 2. 參數與超參數配置 (含 AMP 與梯度累積設定)
# =====================================================================
BATCH_SIZE = 2             # 💡 實體 Batch Size 設為 2，顯存佔用降到最低
ACCUMULATION_STEPS = 4     # 💡 梯度累積步數：累積 2 步才更新一次 (等效 Batch Size = 4)
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 40
BLANK_IDX = 0

VISUAL_DIR = 'D:/Paper_code/en_dataset/batch_Mediaoutput'
AUDIO_DIR = 'D:/Paper_code/en_dataset/add_noise_audio'

LABEL_FILE = 'D:/Paper_code/en_dataset/labels_en.txt'
if not os.path.exists(LABEL_FILE):
    LABEL_FILE = 'D:/Paper_code/dataset/labels_en.txt'

EXCEL_OUTPUT_PATH = 'D:/Paper_code/en_dataset/avsr_training_records.xlsx'
BEST_MODEL_PATH = 'D:/Paper_code/en_dataset/best_tmctc_model.pth'
LATEST_MODEL_PATH = 'D:/Paper_code/en_dataset/tmctc_avsr_model.pth'

if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用運算設備: {device}")
print(f"⚡ 已啟用 AMP 自動混合精度 (FP16) 訓練！")
print(f"⚡ 梯度累積設定: Batch Size = {BATCH_SIZE}, 累積步數 = {ACCUMULATION_STEPS} (等效 Batch = {BATCH_SIZE * ACCUMULATION_STEPS})")

# =====================================================================
# 3. 載入雙模態資料集 (200 筆樣本，保持完整長度輸入)
# =====================================================================
train_dataset = AVSRDataset(
    visual_dir=VISUAL_DIR, 
    audio_dir=AUDIO_DIR, 
    label_file=LABEL_FILE
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=avsr_collate_fn
)

# =====================================================================
# 4. 初始化模型、優化器與 AMP 縮放器 (GradScaler)
# =====================================================================
model = TM_CTC_AVSR_Model(num_classes=NUM_CLASSES).to(device)

# 💡 安全載入檢查機制
loaded_success = False
for ckpt_path, name in [(BEST_MODEL_PATH, "歷史最佳權重"), (LATEST_MODEL_PATH, "最新權重")]:
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1024:
        try:
            print(f"\n🔄 偵測到{name}！正在嘗試載入「{ckpt_path}」繼續訓練...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"✅ {name}載入成功！將在先前訓練成果上繼續優化。")
            loaded_success = True
            break
        except Exception as e:
            print(f"⚠️ {name}載入略過 ({e})")

if not loaded_success:
    print("\n🆕 未載入現有權重，從零隨機初始化模型開始全新訓練。")

criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 💡 初始化 AMP 梯度縮放器 (GradScaler)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# =====================================================================
# 5. 核心訓練迴圈 (AMP + 梯度累積)
# =====================================================================
epoch_summary_list = []
sample_details_list = []

best_loss = float('inf')

print("\n--- 開始執行端到端影音多模態訓練 ---")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()
    
    epoch_gts = []
    epoch_preds = []
    epoch_s_ids = []
    
    for batch_idx, (videos, audios, targets, input_lens, target_lens, sample_ids, raw_texts) in enumerate(train_loader):
        videos = videos.to(device)
        audios = audios.to(device)
        targets = targets.to(device)
        target_lens = target_lens.to(device)
        
        # 💡 使用 autocast 進行 FP16 半精度前向傳播 (省顯存關鍵)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            log_probs = model(videos, audios)
            
            input_lens_adjusted = torch.full(
                size=(log_probs.size(1),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device
            )
            
            # 計算 CTC 損失並依累積步數縮放
            loss = criterion(log_probs, targets, input_lens_adjusted, target_lens)
            loss_scaled = loss / ACCUMULATION_STEPS

        # 💡 使用 Scaler 進行混合精度反向傳播
        scaler.scale(loss_scaled).backward()
        
        # 💡 達到累積步數或是該 Epoch 的最後一個 Batch 時，執行優化器更新
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item()
        
        # 即時解碼與文字收集
        with torch.no_grad():
            batch_preds = ctc_greedy_decode(log_probs)
            epoch_preds.extend(batch_preds)
            epoch_gts.extend(raw_texts)
            epoch_s_ids.extend(sample_ids)

    avg_loss = epoch_loss / len(train_loader)
    
    # 逐句計算 WER
    epoch_wers = []
    for i in range(len(epoch_gts)):
        gt = epoch_gts[i]
        pred = epoch_preds[i] if epoch_preds[i] else "<空白/未辨識出字元>"
        s_id = epoch_s_ids[i]
        s_wer = compute_wer(gt, pred)
        epoch_wers.append(s_wer)
        
        sample_details_list.append({
            "Epoch": epoch + 1,
            "Sample_ID (樣本名稱)": s_id,
            "Ground_Truth (原始文字)": gt,
            "Decoded_Pred (還原文字)": pred,
            "Sample_WER (%)": round(s_wer * 100, 2)
        })

    avg_wer = sum(epoch_wers) / len(epoch_wers) if len(epoch_wers) > 0 else 1.0

    epoch_summary_list.append({
        "Epoch": epoch + 1,
        "Average_Loss (平均損失)": round(avg_loss, 4),
        "Overall_WER (%) (字錯率)": round(avg_wer * 100, 2)
    })

    # 控制台即時輸出
    print(f"\n{'='*75}")
    print(f"📌 Epoch [{epoch+1}/{EPOCHS}] | 平均 Loss: {avg_loss:.4f} | 總體 WER: {avg_wer*100:.2f}%")
    print(f"{'='*75}")
    
    preview_count = min(3, len(epoch_gts))
    for i in range(preview_count):
        gt = epoch_gts[i]
        pred = epoch_preds[i] if epoch_preds[i] else "<空白/未辨識出字元>"
        s_wer = compute_wer(gt, pred)
        print(f"  [{i+1}] 樣本 ID        : {epoch_s_ids[i]}")
        print(f"      原始文字 (GT)  : {gt}")
        print(f"      還原文字 (Pred): {pred}")
        print(f"      字錯率 (WER)   : {s_wer*100:.2f}%\n")

    # 最佳模型保存 (破紀錄時自動更新)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"🌟 發現歷史最佳模型 (Loss: {best_loss:.4f})！已保存至: {BEST_MODEL_PATH}")

    # 即時寫入 Excel (含檔案防鎖定保護)
    os.makedirs(os.path.dirname(EXCEL_OUTPUT_PATH), exist_ok=True)
    for target_path in [EXCEL_OUTPUT_PATH, EXCEL_OUTPUT_PATH.replace('.xlsx', '_backup.xlsx')]:
        try:
            with pd.ExcelWriter(target_path, engine='openpyxl') as writer:
                pd.DataFrame(epoch_summary_list).to_excel(writer, sheet_name='Epoch_Summary', index=False)
                pd.DataFrame(sample_details_list).to_excel(writer, sheet_name='Sample_Details', index=False)
            break
        except Exception:
            continue

# 儲存最新一輪權重
torch.save(model.state_dict(), LATEST_MODEL_PATH)
print(f"\n🎉 訓練完成！")
print(f"   ├─ 最佳模型權重保存於: {BEST_MODEL_PATH}")
print(f"   ├─ 最新模型權重保存於: {LATEST_MODEL_PATH}")
print(f"   └─ Excel 訓練紀錄匯出至: {EXCEL_OUTPUT_PATH}")