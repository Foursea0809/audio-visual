import os
import random
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd  # 🎯【新增】用於生成 Excel 檔案

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, fps=25, audio_hop_ms=10, snr_db=5, p_n=0.25):
        """
        初始化音訊前處理器，完美對應論文規格
        """
        self.sr = sample_rate          # 16 kHz 採樣率 
        self.fps = fps                 # 影片幀率 25 fps
        self.hop_length = int(self.sr * (audio_hop_ms / 1000.0))  # 160 samples
        self.n_fft = self.hop_length * 2.5                        # 400 samples (對應 25ms 視窗寬度)
        self.snr_db = snr_db           # 雜訊比 5 dB 
        self.p_n = p_n                 # 雜訊增強機率 0.25 

    def load_audio(self, file_path):
        """
        載入原始音訊，並強制重新採樣至 16 kHz 。
        """
        y, _ = librosa.load(file_path, sr=self.sr)
        return y

    def add_babble_noise(self, clean_audio, babble_noise_path, force_add=False):
        """
        以 p_n 的機率隨機混入 5 dB SNR 的 Babble Noise。
        回傳: mixed_audio, scaled_noise, power_signal, power_noise_scaled, measured_snr
        """
        # 1. 先計算原始乾淨音訊功率（不論有無加噪都算，方便登記在 Excel）
        power_signal = np.mean(clean_audio ** 2) + 1e-10

        # 決定是否加入雜訊
        if not force_add and random.random() > self.p_n:
            print("🔊 本次前處理：保持乾淨音訊。")
            return clean_audio, None, power_signal, 0.0, None

        print(f"🔊 本次前處理：隨機混入 {self.snr_db} dB Babble Noise！")
        # 載入雜訊檔案並重採樣至 16 kHz
        noise, _ = librosa.load(babble_noise_path, sr=self.sr)

        # 確保雜訊長度大於或等於乾淨音訊，若太短則重複拼貼
        if len(noise) < len(clean_audio):
            repeats = int(np.ceil(len(clean_audio) / len(noise)))
            noise = np.tile(noise, repeats)

        # 隨機從長雜訊中擷取與 clean_audio 等長的部分
        start_idx = random.randint(0, len(noise) - len(clean_audio))
        noise_segment = noise[start_idx : start_idx + len(clean_audio)]

        # 計算原始雜訊功率
        power_noise_current = np.mean(noise_segment ** 2) + 1e-10

        # 2. 計算目標縮放倍率
        power_noise_target = power_signal / (10 ** (self.snr_db / 10))
        scale = np.sqrt(power_noise_target / power_noise_current)

        # 3. 縮放後的雜訊片段與其功率
        scaled_noise = scale * noise_segment
        power_noise_scaled = np.mean(scaled_noise ** 2) + 1e-10

        # 4. 實測 SNR 驗證
        measured_snr = 10 * np.log10(power_signal / power_noise_scaled)

        # --- 印出驗證資訊 ---
        print(f"   ├─ 乾淨音訊平均功率 (P_signal): {power_signal:.6e}")
        print(f"   ├─ 縮放後雜訊平均功率 (P_noise):  {power_noise_scaled:.6e}")
        print(f"   └─ 實測計算 SNR: {measured_snr:.2f} dB (目標: {self.snr_db} dB)")

        mixed_audio = clean_audio + scaled_noise
        return mixed_audio, scaled_noise, power_signal, power_noise_scaled, measured_snr

    def extract_log_mel_spectrogram(self, audio, video_frames):
        """
        1. 進行時序長度對齊，確保音訊長度輸出能達成「1 幀影片對 4 幀音訊特徵」的精確比例 。
        2. 透過短時傅立葉轉換 (STFT) 提取 Mel-spectrogram 特徵。
        3. 標準化特徵矩陣 (Mean-Variance Normalization)。
        """
        target_audio_samples = video_frames * 4 * self.hop_length

        if len(audio) < target_audio_samples:
            audio = np.pad(audio, (0, target_audio_samples - len(audio)), mode='constant')
        else:
            audio = audio[:target_audio_samples]

        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=int(self.n_fft), 
            hop_length=self.hop_length, 
            n_mels=80, 
            center=False
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        features = log_mel_spec.T

        target_time_steps = video_frames * 4
        if features.shape[0] < target_time_steps:
            features = np.pad(features, ((0, target_time_steps - features.shape[0]), (0, 0)), mode='edge')
        elif features.shape[0] > target_time_steps:
            features = features[:target_time_steps, :]

        mean = np.mean(features)
        std = np.std(features) + 1e-10
        normalized_features = (features - mean) / std

        return normalized_features

def plot_audio_comparison(clean_audio, noise_audio, mixed_audio, sr, save_path="waveform_comparison.png"):
    """
    繪製並儲存混音前後的波形對比圖
    """
    time = np.linspace(0, len(clean_audio) / sr, num=len(clean_audio))

    plt.figure(figsize=(12, 8))

    # 1. 原始乾淨音訊
    plt.subplot(3, 1, 1)
    plt.plot(time, clean_audio, color='b', alpha=0.7)
    plt.title("1. Clean Audio (Original)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. 縮放後的 5dB 雜訊
    plt.subplot(3, 1, 2)
    plt.plot(time, noise_audio, color='r', alpha=0.7)
    plt.title("2. Scaled Babble Noise (Target 5 dB SNR)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. 混合後音訊
    plt.subplot(3, 1, 3)
    plt.plot(time, mixed_audio, color='g', alpha=0.7)
    plt.title("3. Mixed Audio (Clean + Noise)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 波形對比圖已成功儲存至: {save_path}")


# --- 批次處理執行區塊 ---
if __name__ == "__main__":
    # 1. 設定資料夾路徑與參數 (請根據實際環境調整路徑)
    input_audio_dir = 'D:/Paper_code/en_dataset/Audio'
    output_feature_dir = 'D:/Paper_code/en_dataset/add_noise_audio'
    noise_dir = 'D:/Paper_code/en_dataset/Babble Noise/Audio'
    
    fps = 25  # 影片幀率
    
    # 2. 自動創建輸出資料夾（若不存在）
    os.makedirs(output_feature_dir, exist_ok=True)

    # 3. 初始化音訊前處理器
    preprocessor = AudioPreprocessor(fps=fps)

    # 4. 支援的音訊格式副檔名
    valid_extensions = ('.mp3', '.wav', '.flac', '.m4a')

    # 取得原始音訊資料夾內所有檔案
    audio_files = [f for f in os.listdir(input_audio_dir) if f.lower().endswith(valid_extensions)]
    
    # 取得雜訊資料夾內的所有雜訊檔案清單
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.lower().endswith(valid_extensions)]

    if not noise_files:
        raise FileNotFoundError(f"❌ 在雜訊資料夾中找不到任何音訊檔案: {noise_dir}")

    # 🎯【新增】用來蒐集所有音訊功率數據的 List
    power_records = []

    print(f"📁 找到 {len(audio_files)} 個測試音訊，{len(noise_files)} 個雜訊檔，準備開始批次處理...\n" + "="*50)

    for idx, filename in enumerate(audio_files, 1):
        # 建立檔案完整路徑
        file_path = os.path.join(input_audio_dir, filename)
        
        # 設定輸出檔名（替換副檔名為 .npy）
        base_name = os.path.splitext(filename)[0]
        output_npy_path = os.path.join(output_feature_dir, f"{base_name}.npy")
        output_img_path = os.path.join(output_feature_dir, f"{base_name}_waveform.png")

        print(f"\n[{idx}/{len(audio_files)}] 正在處理: {filename}")

        # A. 載入音訊
        raw_audio = preprocessor.load_audio(file_path)

        # 隨機從雜訊檔中挑選一個出來使用
        random_noise_path = random.choice(noise_files)

        # B. 隨機加噪（若要強制每條都加噪，可改為 force_add=True）
        augmented_audio, scaled_noise, p_signal, p_noise, snr = preprocessor.add_babble_noise(
            raw_audio, random_noise_path, force_add=False
        )

        # 🎯【新增】記錄本次音訊處理的各項功率數值
        power_records.append({
            "檔案名稱": filename,
            "是否加噪": "是" if scaled_noise is not None else "否",
            "使用的雜訊檔": os.path.basename(random_noise_path) if scaled_noise is not None else "N/A",
            "乾淨音訊平均功率 (P_signal)": p_signal,
            "縮放後雜訊平均功率 (P_noise)": p_noise if scaled_noise is not None else 0.0,
            "實測 SNR (dB)": snr if scaled_noise is not None else "N/A"
        })

        # 若本次有加入雜訊，則自動繪製並儲存波形圖
        if scaled_noise is not None:
            plot_audio_comparison(
                clean_audio=raw_audio, 
                noise_audio=scaled_noise, 
                mixed_audio=augmented_audio, 
                sr=preprocessor.sr, 
                save_path=output_img_path
            )

        # C. 自動根據音訊實際長度計算 video_frames
        duration_sec = len(augmented_audio) / preprocessor.sr
        video_frames = int(round(duration_sec * fps))

        # D. 提取 Log-Mel 特徵並對齊 1:4 時序
        features = preprocessor.extract_log_mel_spectrogram(augmented_audio, video_frames)

        # E. 儲存成 .npy 檔
        np.save(output_npy_path, features)
        print(f"✅ 成功儲存特徵矩陣 {features.shape} 至: {output_npy_path}")

    # 🎯【新增】將收集到的資料存入 Excel 檔
    print("\n" + "="*50 + "\n📊 正在產生功率分析報告...")
    df = pd.DataFrame(power_records)
    excel_output_path = os.path.join(output_feature_dir, "power_analysis_report.xlsx")
    df.to_excel(excel_output_path, index=False)
    
    print(f"🎉 所有音訊處理完成！")
    print(f" Excel 報告已成功儲存至: {excel_output_path}")