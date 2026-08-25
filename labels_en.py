import os
import re

# =====================================================================
# 1. 路徑與預設對白設定
# =====================================================================
VISUAL_DIR = 'D:/Paper_code/en_dataset/batch_Mediaoutput'
OUTPUT_LABEL_FILE = 'D:/Paper_code/dataset/labels_en.txt'
DEFAULT_TRANSCRIPT = "words"

# =====================================================================
# 2. 自然排序（Natural Sort）鍵值解析函式
# =====================================================================
def natural_keys(text):
    """
    將字串中的數字部分轉為整數，其餘保持字串，確保型別一致可安全比較。
    例如: "1_gen" -> ['', 1, '_gen']，"2" -> ['', 2, '']
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# =====================================================================
# 3. 自動掃描資料夾並寫入 txt
# =====================================================================
def create_labels_from_folders():
    output_dir = os.path.dirname(OUTPUT_LABEL_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(VISUAL_DIR):
        print(f"❌ 錯誤：找不到視覺資料夾 ({VISUAL_DIR})，請確認路徑是否正確！")
        return

    sample_ids = []

    # 遍歷資料夾結構
    for root, dirs, files in os.walk(VISUAL_DIR):
        if 'Media' in dirs or any(f.endswith('.png') for f in files):
            rel_path = os.path.relpath(root, VISUAL_DIR)
            
            if rel_path != '.':
                if os.path.basename(rel_path) == 'Media':
                    s_id = os.path.dirname(rel_path)
                else:
                    s_id = rel_path
                
                s_id = s_id.replace('\\', '/')
                if s_id and s_id not in sample_ids:
                    sample_ids.append(s_id)

    # 💡 使用自然排序函式，完美相容數字與非數字檔名混合的情況
    sample_ids = sorted(sample_ids, key=natural_keys)

    total_count = len(sample_ids)
    if total_count == 0:
        print(f"⚠️ 警告：在 {VISUAL_DIR} 中未找到任何包含 Media 的樣本資料夾！")
        return

    print(f"🔍 成功掃描到 {total_count} 個樣本 ID。")

    # 寫入 labels_en.txt
    with open(OUTPUT_LABEL_FILE, 'w', encoding='utf-8') as f:
        for s_id in sample_ids:
            f.write(f"{s_id} {DEFAULT_TRANSCRIPT}\n")

    print(f"\n🎉 標籤檔已成功生成並儲存至：{OUTPUT_LABEL_FILE}")
    print("📄 檔案內容預覽：")
    
    for idx, s_id in enumerate(sample_ids, start=1):
        print(f"{s_id} {DEFAULT_TRANSCRIPT}")

if __name__ == "__main__":
    create_labels_from_folders()