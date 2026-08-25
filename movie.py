import os
import subprocess
import imageio_ffmpeg

# ================= 配置路徑 =================
INPUT_FOLDER = 'D:/Paper_code/en_dataset/Babble Noise/video'          # 放原始影片的資料夾
AUDIO_FOLDER = 'D:/Paper_code/en_dataset/Babble Noise/Audio'     # 存放音訊的資料夾
VIDEO_FOLDER = 'D:/Paper_code/en_dataset/Babble Noise/Visual'    # 存放純影像的資料夾

# 自動建立輸出資料夾（如果不存在的話）
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# 定義要處理的影片副檔名
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

def batch_process_videos():
    # 【修正 1】改為正確的函式名稱：get_ffmpeg_exe()
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # 1. 取得資料夾內所有檔案，並篩選出影片
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(VIDEO_EXTENSIONS)]
    
    # 2. 照檔名順序排序（確保照順序處理）
    files.sort()
    
    total_files = len(files)
    print(f"找到 {total_files} 個影片檔案，開始依序處理...\n")

    for index, file_name in enumerate(files, start=1):
        input_path = os.path.join(INPUT_FOLDER, file_name)
        
        # 取得不含副檔名的主檔名
        base_name = os.path.splitext(file_name)[0]
        
        # 設定輸出的音訊與影像路徑
        audio_output = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")
        video_output = os.path.join(VIDEO_FOLDER, f"{base_name}.mp4")
        
        print(f"[{index}/{total_files}] 正在處理: {file_name}")
        
        # 3. 提取音訊 (轉為 MP3)
        # 【修正 2】將 'ffmpeg_path' 的單引號拿掉，改成變數 ffmpeg_path
        cmd_audio = [ffmpeg_path, '-y', '-i', input_path, '-vn', '-c:a', 'mp3', audio_output]
        subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 4. 提取純影像 (不重新編碼 -c:v copy，極快且無損，並關閉音訊 -an)
        # 【修正 2】將 'ffmpeg_path' 的單引號拿掉，改成變數 ffmpeg_path
        cmd_video = [ffmpeg_path, '-y', '-i', input_path, '-an', '-c:v', 'copy', video_output]
        subprocess.run(cmd_video, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("\n恭喜！所有影片已依序處理完成！")

if __name__ == "__main__":
    batch_process_videos()