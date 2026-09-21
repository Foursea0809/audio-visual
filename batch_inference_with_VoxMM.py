import os
import cv2
import sys
import numpy as np
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import seaborn as sns
from openpyxl import Workbook
from openpyxl.styles import Font
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ================= 1. 核心參數與路徑設定 =================
ROOT = Path(__file__).resolve().parent
# Only the requested evaluation split is used as input.  Generated clips are
# kept separately under original/train, so the supplied test media is never
# overwritten.
VISUAL_DIR = ROOT / 'LRS-VoxMM' / 'lrs-voxmm' / 'original' / 'train'
OUTPUT_DIR = ROOT / 'LRS-VoxMM' / 'lrs-voxmm' / 'original' / 'gan'
WAV2LIP_DIR = ROOT / 'Wav2Lip'
CHECKPOINT_PATH = WAV2LIP_DIR / 'checkpoints' / 'wav2lip_gan.pth'
os.environ.setdefault('NUMBA_DISABLE_JIT', '1')

# 控制是否計算 PSNR/SSIM 指標（True: 開啟, False: 純生成以加快速度）
ENABLE_METRICS = os.environ.get('VOXMM_ENABLE_METRICS', '0') == '1'
METRICS_REPORT_PATH = OUTPUT_DIR / 'quality_metrics.xlsx'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# 建立輸出目錄
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================= 2. 數據評估與報表函式 =================
def evaluate_quality(orig_video_path, gen_video_path):
    """逐幀計算原始影片與生成影片的 PSNR 與 SSIM 指標"""
    cap_orig = cv2.VideoCapture(str(orig_video_path))
    cap_gen = cv2.VideoCapture(str(gen_video_path))
    
    psnr_list, ssim_list = [], []
    
    while cap_orig.isOpened() and cap_gen.isOpened():
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_gen.read()
        if not ret1 or not ret2:
            break
            
        # 轉換為灰階進行品質評估
        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 計算 PSNR 與 SSIM
        psnr_list.append(psnr(g1, g2))
        ssim_list.append(ssim(g1, g2))
        
    cap_orig.release()
    cap_gen.release()
    return psnr_list, ssim_list


def save_metrics_report(rows, report_path):
    """將每支影片的品質指標彙整為 Excel 報表。"""
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = '影片品質指標'

    headers = ['影片檔名', '平均 PSNR (dB)', '平均 SSIM', '評估幀數', '處理狀態', '備註']
    worksheet.append(headers)

    for row in rows:
        worksheet.append(row)

    for cell in worksheet[1]:
        cell.font = Font(bold=True)

    for row_index in range(2, worksheet.max_row + 1):
        worksheet.cell(row=row_index, column=2).number_format = '0.00'
        worksheet.cell(row=row_index, column=3).number_format = '0.0000'

    column_widths = [24, 18, 16, 12, 14, 60]
    for column_index, width in enumerate(column_widths, start=1):
        worksheet.column_dimensions[chr(64 + column_index)].width = width

    worksheet.freeze_panes = 'A2'
    workbook.save(report_path)


def configure_figure_style():
    """設定適合論文使用的圖表樣式。"""
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#D9D9D9',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def plot_quality_metric(video_ids, values, ylabel, title, color, output_stem, decimals):
    """輸出單一品質指標的 PDF、SVG 與 600 dpi PNG 圖表。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    values = np.asarray(values, dtype=float)
    mean_value = values.mean()
    value_range = values.max() - values.min()
    padding = max(value_range * 0.12, 0.004 if ylabel == 'SSIM' else 0.5)

    ax.plot(
        video_ids,
        values,
        color=color,
        linewidth=1.8,
        marker='o',
        markersize=3.2,
        markerfacecolor='white',
        markeredgewidth=1.0,
        label='Per-video mean',
    )
    ax.axhline(
        mean_value,
        color='#4D4D4D',
        linewidth=1.2,
        linestyle='--',
        label=f'Overall mean = {mean_value:.{decimals}f}',
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel('Video index')
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(video_ids) - 1, max(video_ids) + 1)
    ax.set_ylim(values.min() - padding, values.max() + padding)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower left', frameon=False, handlelength=2.4)

    for extension in ('pdf', 'svg', 'png'):
        save_options = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if extension == 'png':
            save_options['dpi'] = 600
        fig.savefig(FIGURES_DIR / f'{output_stem}.{extension}', **save_options)
    plt.close(fig)


def save_quality_figures(rows):
    """將成功生成影片的平均 PSNR／SSIM 彙整成論文用圖表。"""
    valid_rows = []
    for row in rows:
        file_name, avg_psnr, avg_ssim, _, status, _ = row
        try:
            video_id = int(file_name)
        except (TypeError, ValueError):
            continue
        if status == '成功' and avg_psnr is not None and avg_ssim is not None:
            valid_rows.append((video_id, float(avg_psnr), float(avg_ssim)))

    if not valid_rows:
        print('⚠️ 沒有可用的 PSNR／SSIM 資料，略過圖表輸出。')
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_figure_style()
    valid_rows.sort(key=lambda row: row[0])
    video_ids, psnr_values, ssim_values = map(list, zip(*valid_rows))

    plot_quality_metric(
        video_ids,
        psnr_values,
        'PSNR (dB)',
        'Wav2Lip Generation Quality: PSNR',
        '#1F77B4',
        'figure_psnr',
        2,
    )
    plot_quality_metric(
        video_ids,
        ssim_values,
        'SSIM',
        'Wav2Lip Generation Quality: SSIM',
        '#C03A2B',
        'figure_ssim',
        4,
    )
    print(f'📊 論文圖表已儲存: {FIGURES_DIR}')



# ================= 3. 批次處理主流程 =================
def main():
    visual_files = sorted(VISUAL_DIR.rglob("*.mp4"))
    limit = int(os.environ.get('VOXMM_MAX_ITEMS', '0'))
    if limit > 0:
        visual_files = visual_files[:limit]

    if not visual_files:
        print(f"❌ 在 {VISUAL_DIR} 找不到任何 .mp4 影片！")
        return

    print(f"🚀 找到 {len(visual_files)} 部影片，準備開始批次處理...\n")
    metrics_rows = []

    for visual_path in visual_files:
        file_name = visual_path.stem
        audio_path = visual_path.with_suffix('.wav')
        output_path = OUTPUT_DIR / visual_path.relative_to(VISUAL_DIR)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # A finished clip is a safe resume point for this long-running batch.
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"⏭️ [略過] 已生成: {output_path}")
            metrics_rows.append([file_name, None, None, 0, '已存在', '保留既有生成影片'])
            continue
        
        # 檢查對應音訊檔案是否存在
        if not audio_path.exists():
            print(f"⚠️ [跳過] 找不到對應音訊: {audio_path.name}")
            continue
            
        print(f"🎬 [正在處理] 影片: {visual_path.name} ＋ 音訊: {audio_path.name}")
        
        # 組裝推論指令
        cmd = [
            sys.executable, str(WAV2LIP_DIR / "inference.py"),
            "--checkpoint_path", str(CHECKPOINT_PATH),
            "--face", str(visual_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
            # LRS-VoxMM clips are already 224x224 face crops.  Passing the
            # full crop skips an unreliable external face detector on Windows.
            "--box", "0", "224", "0", "224",
            "--wav2lip_batch_size", "4",
        ]
        
        try:
            # 1. 執行推論生成影片
            subprocess.run(cmd, check=True, cwd=WAV2LIP_DIR)
            print(f"✅ [生成成功] 影片已儲存: {output_path.name}")
            
            # 2. 依設定執行指標計算與繪圖
            if ENABLE_METRICS:
                psnr_list, ssim_list = evaluate_quality(visual_path, output_path)
                if psnr_list and ssim_list:
                    avg_psnr = np.mean(psnr_list)
                    avg_ssim = np.mean(ssim_list)
                    
                    print(f"📊 [指標分析] 平均 PSNR: {avg_psnr:.2f} dB | 平均 SSIM: {avg_ssim:.4f}")
                    metrics_rows.append([
                        file_name,
                        float(avg_psnr),
                        float(avg_ssim),
                        len(psnr_list),
                        '成功',
                        '',
                    ])
                else:
                    metrics_rows.append([file_name, None, None, 0, '成功', '無可評估的影格'])
            else:
                metrics_rows.append([file_name, None, None, 0, '成功', '未啟用品質指標'])
                    
            print("-" * 50)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [推論失敗] 處理 {file_name} 時發生錯誤: {e}\n")
            metrics_rows.append([file_name, None, None, 0, '推論失敗', str(e)])
        except Exception as e:
            print(f"❌ [指標分析失敗] {file_name}: {e}\n")
            metrics_rows.append([file_name, None, None, 0, '指標分析失敗', str(e)])

    save_metrics_report(metrics_rows, METRICS_REPORT_PATH)
    print(f"📄 品質指標 Excel 已儲存: {METRICS_REPORT_PATH}")
    if ENABLE_METRICS:
        save_quality_figures(metrics_rows)
    print("🎉 所有批次影片處理完成！")

if __name__ == "__main__":
    main()
