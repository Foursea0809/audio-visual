import os
import glob
import pickle
import random
import argparse
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. 命令列參數設定 ---
parser = argparse.ArgumentParser(description="MediaPipe 批次視覺前處理與嘴唇裁切腳本 (OpenCV 原生版)")
parser.add_argument("-i", "--input_dir", default='D:/Paper_code/en_dataset/Dataset Expansion', help="資料集擴充影片根目錄")
parser.add_argument("-o", "--output_dir", default='D:/Paper_code/en_dataset/batch_Mediaoutput', help="特徵與圖片導出根目錄")
parser.add_argument("-m", "--model_path", default='D:/Paper_code/MediaPipe/face_landmarker.task', help="MediaPipe 模型路徑")
args = parser.parse_args()

# --- 2. 論文規格與 MediaPipe 嘴唇索引定義 ---
MOUTH_TOP = 0
MOUTH_BOTTOM = 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

width_crop_max = 112
height_crop_max = 112

def process_single_video(video_path, rel_path, output_base_dir, model_path):
    """
    處理單一影片：偵測嘴唇、112x112 灰階裁切、隨機跳格/翻轉數據增強，並導出 PNG 序列與 activation.pkl。
    完全使用 OpenCV (cv2.VideoCapture & cv2.VideoWriter) 進行原生影片讀寫，拋棄 skvideo 依賴。
    """
    # 建構該影片對應的導出目錄結構
    file_stem = os.path.splitext(rel_path)[0]  # 例: subject_1/session1/0
    sample_output_dir = os.path.join(output_base_dir, file_stem)
    mouth_destination_path = os.path.join(sample_output_dir, 'Media')
    os.makedirs(mouth_destination_path, exist_ok=True)

    annotated_video_path = os.path.join(sample_output_dir, 'Media.mp4')

    # 初始化全新的 FaceLandmarker (確保每個影片的時間戳記從 0 重新計算)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options)

    # 1. 使用 OpenCV 打開影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 無法讀取影片檔 {video_path}")
        face_mesh.close()
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # 預設為 25.0 fps

    # 2. 初始化 OpenCV 影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (w, h))

    activation = []
    counter = 0

    # 3. 逐幀讀取與處理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 論文常規增強：隨機移除/跳過影格 (p_drop)
        p_drop = random.uniform(0.01, 0.20)
        if random.random() < p_drop:
            counter += 1
            activation.append(0)
            continue

        # OpenCV 讀入預設為 BGR，轉為 RGB 供 MediaPipe 處理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 動態計算單調遞增的時間戳記 (毫秒)
        frame_timestamp_ms = int(counter * (1000.0 / fps))

        # 執行新版 MediaPipe Tasks API 影片模式檢測
        results = face_mesh.detect_for_video(mp_image, frame_timestamp_ms)

        # 複製 BGR 畫布進行繪圖標註
        display_frame = frame.copy()
        found_any_mouth = False

        if results.face_landmarks:
            for k, face_landmarks in enumerate(results.face_landmarks):
                top_pt = face_landmarks[MOUTH_TOP]
                bottom_pt = face_landmarks[MOUTH_BOTTOM]
                left_pt = face_landmarks[MOUTH_LEFT]
                right_pt = face_landmarks[MOUTH_RIGHT]

                # 還原像素中心點
                x_center = ((left_pt.x + right_pt.x) / 2.0) * w
                y_center = ((top_pt.y + bottom_pt.y) / 2.0) * h

                # 計算 112x112 裁切邊界
                x1 = int(x_center - width_crop_max / 2.0)
                y1 = int(y_center - height_crop_max / 2.0)
                x2 = x1 + width_crop_max
                y2 = y1 + height_crop_max

                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    mouth_roi = display_frame[y1:y2, x1:x2]

                    # 論文常規增強：50% 機率水平翻轉
                    if random.random() < 0.5:
                        mouth_roi = cv2.flip(mouth_roi, 1)

                    # 轉為灰階
                    mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)

                    # 儲存灰階圖片 Patch
                    cv_filename = os.path.join(mouth_destination_path, f"frame_{counter}_p{k}.png")
                    cv2.imwrite(cv_filename, mouth_gray)

                    # 標註綠色成功框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    found_any_mouth = True
                else:
                    # 標註紅色貼邊失敗框
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        activation.append(1 if found_any_mouth else 0)

        # 直接將繪製好框線的 BGR 畫布寫入影片
        writer.write(display_frame)
        counter += 1

    # 資源釋放與狀態導出
    cap.release()
    writer.release()
    face_mesh.close()

    with open(os.path.join(sample_output_dir, 'activation.pkl'), 'wb') as f:
        pickle.dump(activation, f)

# --- 3. 批次主程序 ---
def main():
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_path = args.model_path

    # 支持的影片副檔名格式
    extensions = ('*.avi', '*.mp4', '*.mkv', '*.mov')
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    print(f"🔍 在 {input_dir} 中找到 {len(video_files)} 個影片檔案，開始執行批次預處理...")

    for idx, v_path in enumerate(video_files):
        rel_path = os.path.relpath(v_path, input_dir)
        print(f"[{idx+1}/{len(video_files)}] 正在處理: {rel_path}")
        process_single_video(v_path, rel_path, output_dir, model_path)

    print(f"\n🎉 批次預處理完成！所有灰階圖片與標註檔已儲存至: {output_dir}")

if __name__ == "__main__":
    main()