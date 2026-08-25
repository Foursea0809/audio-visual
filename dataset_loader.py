import os
import glob
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def safe_natural_key(text):
    """
    100% 防崩潰自然排序演算法，相容 1, 1_lip, 10, 10_lip 等所有格式
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

class AVSRDataset(Dataset):
    def __init__(self, visual_dir, audio_dir, label_file=None, transform=None):
        self.visual_dir = visual_dir
        self.audio_dir = audio_dir
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.421], std=[0.165])
        ])
        
        # 1. 載入文字標籤檔
        self.label_map = self._load_labels(label_file) if label_file else {}
        print(f"📖 標籤檔讀取完成，共載入 {len(self.label_map)} 筆真實對白標籤！")
        
        # 2. 掃描視覺目錄並智慧配對音訊
        self.samples = self._match_modalities()
        print(f"✅ 雙模態資料庫載入完成：成功配對 {len(self.samples)} 組影音樣本！\n")

    def _load_labels(self, label_file):
        label_map = {}
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        s_id = parts[0].replace('\\', '/')
                        label_map[s_id] = parts[1].strip()
        else:
            print(f"❌ 找不到標籤檔：{label_file}")
        return label_map

    def _text_to_tensor(self, text):
        char_map = {ch: i+1 for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ")}
        indices = [char_map[c.lower()] for c in str(text) if c.lower() in char_map]
        return torch.tensor(indices, dtype=torch.long)

    def _match_modalities(self):
        matched = []
        if not os.path.exists(self.visual_dir):
            print(f"❌ 找不到視覺資料夾：{self.visual_dir}")
            return matched

        for root, dirs, files in os.walk(self.visual_dir):
            if 'Media' in dirs or any(f.endswith('.png') for f in files):
                rel_path = os.path.relpath(root, self.visual_dir)
                if rel_path != '.':
                    if os.path.basename(rel_path) == 'Media':
                        s_id = os.path.dirname(rel_path)
                        v_folder = root
                    else:
                        s_id = rel_path
                        v_folder = os.path.join(self.visual_dir, s_id, 'Media') if os.path.exists(os.path.join(self.visual_dir, s_id, 'Media')) else root

                    s_id = s_id.replace('\\', '/')
                    img_files = sorted(glob.glob(os.path.join(v_folder, '*.png')))
                    if len(img_files) == 0:
                        continue

                    # 自動處理生成影片的音訊共用邏輯 (例如 1_lip 使用 1_audio.npy)
                    base_id = s_id.replace('_lip', '').replace('_gen', '').replace('_wav2lip', '')
                    
                    possible_audio_names = [
                        f"{s_id}_audio.npy", f"{s_id}.npy",
                        f"{base_id}_audio.npy", f"{base_id}.npy"
                    ]
                    
                    audio_path = None
                    for name in possible_audio_names:
                        candidate = os.path.join(self.audio_dir, name)
                        if os.path.exists(candidate):
                            audio_path = candidate
                            break

                    if audio_path and os.path.exists(audio_path):
                        if not any(m['sample_id'] == s_id for m in matched):
                            matched.append({
                                'sample_id': s_id,
                                'visual_folder': v_folder,
                                'audio_path': audio_path
                            })

        # 安全自然排序
        matched = sorted(matched, key=lambda x: safe_natural_key(x['sample_id']))
        return matched

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        s_id = sample['sample_id']
        
        # A. 視覺序列讀取
        img_paths = sorted(glob.glob(os.path.join(sample['visual_folder'], '*.png')))
        frames = []
        for img_path in img_paths:
            img = Image.open(img_path)
            frames.append(self.transform(img))
        video_tensor = torch.stack(frames, dim=1)
        T_video = video_tensor.size(1)
        
        # B. 音訊特徵讀取與時間軸對齊 (T_video * 4)
        audio_feat = np.load(sample['audio_path'])
        audio_tensor = torch.from_numpy(audio_feat).float()
        target_audio_len = T_video * 4
        current_audio_len = audio_tensor.size(0)
        
        if current_audio_len < target_audio_len:
            pad_len = target_audio_len - current_audio_len
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 0, 0, pad_len))
        elif current_audio_len > target_audio_len:
            audio_tensor = audio_tensor[:target_audio_len, :]

        # C. 標籤文字讀取
        raw_text = self.label_map.get(s_id, "")
        target_tensor = self._text_to_tensor(raw_text)
        
        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'target': target_tensor,
            'video_len': video_tensor.size(1),
            'audio_len': audio_tensor.size(0),
            'target_len': len(target_tensor),
            'sample_id': s_id,
            'raw_text': raw_text
        }

def avsr_collate_fn(batch):
    videos = [b['video'] for b in batch]
    audios = [b['audio'] for b in batch]
    targets = [b['target'] for b in batch]
    sample_ids = [b['sample_id'] for b in batch]
    raw_texts = [b['raw_text'] for b in batch]
    
    video_lens = torch.tensor([b['video_len'] for b in batch], dtype=torch.long)
    audio_lens = torch.tensor([b['audio_len'] for b in batch], dtype=torch.long)
    target_lens = torch.tensor([b['target_len'] for b in batch], dtype=torch.long)
    
    max_v_len = max(v.size(1) for v in videos)
    padded_videos = [torch.nn.functional.pad(v, (0, 0, 0, 0, 0, max_v_len - v.size(1))) for v in videos]
    padded_videos = torch.stack(padded_videos, dim=0)

    max_a_len = max(a.size(0) for a in audios)
    padded_audios = [torch.nn.functional.pad(a, (0, 0, 0, max_a_len - a.size(0))) for a in audios]
    padded_audios = torch.stack(padded_audios, dim=0)

    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return padded_videos, padded_audios, padded_targets, audio_lens, target_lens, sample_ids, raw_texts