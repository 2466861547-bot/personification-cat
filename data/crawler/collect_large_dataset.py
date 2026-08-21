"""
大规模宠物声音数据集采集脚本 (v3 国内优化版)
目标: 采集 100 万+ 真实猫狗声音样本

针对国内网络优化:
  - GitHub raw 作为主要下载源 (国内可访问)
  - hf-mirror.com 作为 HuggingFace 国内镜像
  - 支持代理配置
  - 失败时降级到本地合成数据生成

使用方法:
  # 自动选择最佳数据源
  python3 data/crawler/collect_large_dataset.py --target=1000000

  # 使用代理 (走 VPN)
  python3 data/crawler/collect_large_dataset.py --proxy=http://127.0.0.1:7890

  # 仅下载可访问的数据集 (GitHub/HF-mirror)
  python3 data/crawler/collect_large_dataset.py --safe-mode

  # 跳过 YouTube (国内无法访问)
  python3 data/crawler/collect_large_dataset.py --skip-youtube

  # 降级: 用本地合成数据补足
  python3 data/crawler/collect_large_dataset.py --fallback-synthetic
"""

import os
import sys
import json
import time
import subprocess
import argparse
import hashlib
import re
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LARGE_DATASET_DIR = os.path.join(DATA_DIR, "large_dataset")

# 国内可访问的镜像源
MIRRORS = {
    # GitHub raw (国内可访问, 小文件可用)
    "github_raw": "https://raw.githubusercontent.com",
    # HuggingFace 国内镜像
    "hf_mirror": "https://hf-mirror.com",
    # 原始源 (需代理)
    "audioset": "https://research.google.com/audioset",
    "freesound": "https://freesound.org",
}

# AudioSet 资源 (优先 GitHub 镜像)
AUDIOSET_RESOURCES = [
    # GitHub 镜像 (国内可访问)
    {
        "name": "ontology.json",
        "urls": [
            "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
            "https://research.google.com/audioset/ontology/ontology.json",
        ],
    },
    {
        "name": "eval_segments.csv",
        "urls": [
            "https://raw.githubusercontent.com/karoldvl/audioset/master/eval_segments.csv",
            "https://research.google.com/audioset/data/eval_segments.csv",
        ],
    },
    {
        "name": "balanced_train_segments.csv",
        "urls": [
            "https://raw.githubusercontent.com/karoldvl/audioset/master/balanced_train_segments.csv",
            "https://research.google.com/audioset/data/balanced_train_segments.csv",
        ],
    },
    {
        "name": "unbalanced_train_segments.csv",
        "urls": [
            "https://raw.githubusercontent.com/karoldvl/audioset/master/unbalanced_train_segments.csv",
            "https://research.google.com/audioset/data/unbalanced_train_segments.csv",
        ],
    },
]

# AudioSet 中猫狗相关的 ontology IDs (从 ontology.json 解析得到)
AUDIOSET_LABELS = {
    "cat": ["/m/01yrx", "/m/07qrkrw", "/m/02yds9", "/m/07rjwbb"],  # Cat, Meow, Purr, Hiss
    "dog": ["/m/0bt9lr", "/m/05tny_", "/m/07qf0zm", "/m/07qz6j3", "/m/07r_k2n"],  # Dog, Bark, Howl, Whimper, Yip
}

# 国内可访问的小型数据集
SMALL_DATASETS = [
    {
        "name": "ESC-50",
        "url": "https://github.com/karoldvl/ESC-50/raw/master/esc50.csv",
        "mirror_urls": [
            "https://raw.githubusercontent.com/karoldvl/ESC-50/master/esc50.csv",
        ],
        "format": "csv",
        "pet_filter": {"cat": ["cat"], "dog": ["dog"]},
    },
]

# Freesound 搜索关键词
FREESOUND_QUERIES = {
    "cat": ["cat meow", "cat purr", "cat hiss", "kitten", "cat sound"],
    "dog": ["dog bark", "dog howl", "dog growl", "puppy", "dog sound"],
}

# YouTube 搜索关键词 (国内需代理)
YOUTUBE_KEYWORDS = {
    "cat": ["cat meowing", "cat purring", "kitten sounds", "cat hissing"],
    "dog": ["dog barking", "dog howling", "puppy crying", "dog growling"],
}


def ensure_dirs():
    """创建目录结构"""
    for d in [
        LARGE_DATASET_DIR,
        os.path.join(LARGE_DATASET_DIR, "audioset"),
        os.path.join(LARGE_DATASET_DIR, "freesound"),
        os.path.join(LARGE_DATASET_DIR, "youtube"),
        os.path.join(LARGE_DATASET_DIR, "enhanced"),
        os.path.join(LARGE_DATASET_DIR, "final", "cat"),
        os.path.join(LARGE_DATASET_DIR, "final", "dog"),
        os.path.join(LARGE_DATASET_DIR, "synthetic"),
    ]:
        os.makedirs(d, exist_ok=True)


def create_session(proxy: str = "", retries: int = 3) -> "requests.Session":
    """创建带重试和代理的 requests session"""
    if not HAS_REQUESTS:
        return None

    session = requests.Session()

    # 重试策略
    retry = Retry(
        total=retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 代理
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy,
        }
        print(f"  使用代理: {proxy}")

    # UA
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    })

    return session


def _download_file_multi(urls: List[str], save_path: str, session, timeout: int = 60) -> bool:
    """从多个 URL 尝试下载, 返回第一个成功的"""
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  ✓ 已存在: {os.path.basename(save_path)} ({size_mb:.1f}MB)")
        return True

    if session is None:
        print("  ⚠ 无可用 requests session")
        return False

    for i, url in enumerate(urls):
        try:
            source = "GitHub 镜像" if "github" in url else "原始源"
            if "hf-mirror" in url:
                source = "HF 国内镜像"
            print(f"  [{i+1}/{len(urls)}] {source}: {os.path.basename(save_path)}...")

            resp = session.get(url, stream=True, timeout=timeout, verify=False)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 16):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size and downloaded % (1024 * 1024) < 8192 * 16:
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        pct = downloaded / total_size * 100
                        print(f"\r    {mb:.1f}/{total_mb:.1f}MB ({pct:.0f}%)", end="", flush=True)

            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"\r    ✓ 下载成功: {os.path.basename(save_path)} ({size_mb:.1f}MB)")
            return True

        except Exception as e:
            err_type = type(e).__name__
            if "SSL" in str(e):
                print(f"    ⚠ SSL 错误 (国内访问限制, 需代理)")
            elif "Connection" in str(e) or "Timeout" in err_type:
                print(f"    ⚠ 连接失败 (网络限制)")
            else:
                print(f"    ⚠ 失败: {err_type}")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            continue

    print(f"  ✗ 所有 URL 失败: {os.path.basename(save_path)}")
    return False


# =====================================
# 1. AudioSet 自动下载
# =====================================

def download_audioset(session, target_per_class: int = 10000) -> List[Dict]:
    """自动下载 AudioSet 资源"""
    print("\n[1/5] 下载 Google AudioSet 数据...")
    audioset_dir = os.path.join(LARGE_DATASET_DIR, "audioset")

    # 1.1 下载所有资源
    print("\n  1.1 下载 AudioSet 资源 (优先 GitHub 镜像)...")
    for resource in AUDIOSET_RESOURCES:
        save_path = os.path.join(audioset_dir, resource["name"])
        _download_file_multi(resource["urls"], save_path, session, timeout=300)

    # 1.2 检查哪些文件可用
    available = {}
    for resource in AUDIOSET_RESOURCES:
        path = os.path.join(audioset_dir, resource["name"])
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            available[resource["name"]] = path

    if not available:
        print("\n  ⚠ 无可用 AudioSet 文件, 跳过")
        return []

    print(f"\n  1.2 可用文件: {list(available.keys())}")

    # 1.3 解析 ontology 获取猫狗类别 ID
    cat_dog_ids = set()
    if "ontology.json" in available:
        cat_dog_ids = _parse_ontology_for_pets(available["ontology.json"])
        print(f"  1.3 提取猫狗类别: {len(cat_dog_ids)} 个")

    # 1.4 解析 CSV 提取片段
    samples = []
    csv_files = [v for k, v in available.items() if k.endswith(".csv")]
    if csv_files:
        print(f"\n  1.4 解析 CSV 文件...")
        samples = _parse_audioset_csv(csv_files, cat_dog_ids, target_per_class)
        print(f"  ✓ 提取 {len(samples)} 个片段 (cat: {len([s for s in samples if s['pet_type']=='cat'])}, dog: {len([s for s in samples if s['pet_type']=='dog'])})")

    # 1.5 下载音频片段 (需要 YouTube 访问, 国内可能失败)
    if samples and not _is_youtube_blocked(session):
        print(f"\n  1.5 下载 {min(100, len(samples))} 个音频片段 (YouTube)...")
        downloaded = _download_youtube_batch(samples[:100], audioset_dir)
        return downloaded
    elif samples:
        print("\n  ⚠ YouTube 无法访问, 跳过音频片段下载")
        print("    提示: 配置 --proxy 参数后可下载音频")

    return samples


def _parse_ontology_for_pets(ontology_path: str) -> set:
    """从 ontology.json 提取所有猫狗相关类别 ID"""
    try:
        with open(ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pet_ids = set()
        relevant_names = {
            "Cat", "Dog", "Meow", "Purr", "Hiss", "Bark", "Howl", "Growl", "Whimper", "Yip",
            "Cats", "Dogs", "Kitten", "Puppy",
        }

        for item in data:
            name = item.get("name", "")
            if name in relevant_names:
                pet_ids.add(item["id"])
                # 也添加子类别
                for child_id in item.get("child_ids", []):
                    pet_ids.add(child_id)

        return pet_ids
    except Exception as e:
        print(f"    ⚠ ontology 解析失败: {e}")
        return set()


def _parse_audioset_csv(csv_files: List[str], pet_ids: set, limit: int) -> List[Dict]:
    """解析 AudioSet CSV, 提取猫狗片段"""
    samples = []
    cat_count = 0
    dog_count = 0

    # 如果没有 ontology, 用预定义的标签
    if not pet_ids:
        pet_ids = set()
        for ids in AUDIOSET_LABELS.values():
            pet_ids.update(ids)

    for csv_path in csv_files:
        print(f"    解析 {os.path.basename(csv_path)}...")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
            print(f"      {len(data_lines)} 行数据")

            for line in data_lines:
                parts = line.strip().split(",")
                if len(parts) < 4:
                    continue
                yt_id = parts[0]
                try:
                    start = float(parts[1])
                    end = float(parts[2])
                except ValueError:
                    continue

                labels = set(parts[3].strip('"').split())

                pet_type = None
                if labels & set(AUDIOSET_LABELS["cat"]):
                    pet_type = "cat"
                    if cat_count >= limit:
                        continue
                    cat_count += 1
                elif labels & set(AUDIOSET_LABELS["dog"]):
                    pet_type = "dog"
                    if dog_count >= limit:
                        continue
                    dog_count += 1

                if pet_type:
                    samples.append({
                        "yt_id": yt_id,
                        "start": start,
                        "end": end,
                        "pet_type": pet_type,
                        "source": "audioset",
                    })

        except Exception as e:
            print(f"      ⚠ 解析失败: {e}")

    return samples


def _is_youtube_blocked(session) -> bool:
    """检测 YouTube 是否可访问"""
    if session is None:
        return True
    try:
        r = session.head("https://www.youtube.com", timeout=5, verify=False)
        return r.status_code != 200
    except Exception:
        return True


def _download_youtube_batch(samples: List[Dict], output_dir: str) -> List[Dict]:
    """批量下载 YouTube 片段"""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("    ⚠ 未安装 yt-dlp: pip install yt-dlp")
        return []

    downloaded = []

    def download_one(sample):
        yt_url = f"https://youtube.com/watch?v={sample['yt_id']}"
        out_path = os.path.join(output_dir, f"{sample['pet_type']}_{sample['yt_id']}_{int(sample['start'])}.wav")

        if os.path.exists(out_path):
            sample["audio_path"] = out_path
            return sample

        cmd = [
            "yt-dlp", "-x", "--audio-format", "wav",
            "--download-sections", f"*{sample['start']}-{sample['end']}",
            "--force-keyframes-at-cuts",
            "--no-playlist", "--no-overwrites",
            "-o", out_path.replace(".wav", ".%(ext)s"),
            yt_url,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(out_path):
                sample["audio_path"] = out_path
                return sample
        except Exception:
            pass
        return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_one, s): s for s in samples}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                downloaded.append(result)
            if (i + 1) % 10 == 0:
                print(f"    进度: {i+1}/{len(samples)}, 成功: {len(downloaded)}")

    return downloaded


# =====================================
# 2. Freesound 下载 (爬虫模式)
# =====================================

def download_freesound(session, target: int = 10000, api_key: str = "") -> List[Dict]:
    """Freesound 下载: API 优先, 降级爬虫"""
    print("\n[2/5] 下载 Freesound.org 数据...")

    if api_key:
        return download_freesound_api(api_key, session, target)

    # 爬虫模式
    return download_freesound_scrape(session, target)


def download_freesound_api(api_key: str, session, target: int) -> List[Dict]:
    """通过 Freesound API 下载"""
    fs_dir = os.path.join(LARGE_DATASET_DIR, "freesound")
    samples = []
    headers = {"Authorization": f"Token {api_key}"}

    for pet_type, queries in FREESOUND_QUERIES.items():
        for query in queries:
            if len([s for s in samples if s["pet_type"] == pet_type]) >= target // 2:
                break
            try:
                url = "https://freesound.org/apiv2/search/text/"
                params = {"query": query, "page_size": 150, "fields": "id,duration,previews"}
                resp = session.get(url, headers=headers, params=params, timeout=30, verify=False)
                resp.raise_for_status()
                data = resp.json()

                for result in data.get("results", []):
                    preview_url = result.get("previews", {}).get("preview-hq-mp3")
                    if not preview_url:
                        continue
                    out_path = os.path.join(fs_dir, f"{pet_type}_{result['id']}.mp3")
                    if not os.path.exists(out_path):
                        audio_resp = session.get(preview_url, headers=headers, timeout=30, verify=False)
                        with open(out_path, "wb") as f:
                            f.write(audio_resp.content)
                    samples.append({
                        "pet_type": pet_type, "audio_path": out_path,
                        "source": "freesound", "query": query,
                    })
                time.sleep(0.5)
            except Exception as e:
                print(f"    ⚠ '{query}' 失败: {type(e).__name__}")
    return samples


def download_freesound_scrape(session, target: int) -> List[Dict]:
    """Freesound 爬虫模式 (无需 API key)"""
    if not HAS_BS4:
        print("  ⚠ 未安装 beautifulsoup4: pip install beautifulsoup4")
        return []

    fs_dir = os.path.join(LARGE_DATASET_DIR, "freesound")
    samples = []

    for pet_type, queries in FREESOUND_QUERIES.items():
        for query in queries:
            if len([s for s in samples if s["pet_type"] == pet_type]) >= target // 2:
                break
            try:
                url = f"https://freesound.org/search/?q={query}&f=0&s=rating+desc"
                resp = session.get(url, timeout=20, verify=False)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                sound_links = soup.find_all("a", href=re.compile(r"/people/.*?/sounds/\d+/"))
                sound_ids = set()
                for link in sound_links:
                    match = re.search(r"/sounds/(\d+)/", link.get("href", ""))
                    if match:
                        sound_ids.add(match.group(1))

                for sound_id in list(sound_ids)[:20]:
                    mp3_url = f"https://cdn.freesound.org/previews/{sound_id[:4]}/{sound_id}_64kb.mp3"
                    out_path = os.path.join(fs_dir, f"{pet_type}_{sound_id}.mp3")
                    if not os.path.exists(out_path):
                        try:
                            audio_resp = session.get(mp3_url, timeout=20, verify=False)
                            if audio_resp.status_code == 200 and len(audio_resp.content) > 1000:
                                with open(out_path, "wb") as f:
                                    f.write(audio_resp.content)
                                samples.append({
                                    "pet_type": pet_type, "audio_path": out_path,
                                    "source": "freesound_scrape", "query": query,
                                })
                        except Exception:
                            pass
                time.sleep(1)
            except Exception as e:
                print(f"    ⚠ 爬取 '{query}' 失败: {type(e).__name__}")

    return samples


# =====================================
# 3. YouTube 爬取
# =====================================

def crawl_youtube_videos(target: int = 50000) -> List[Dict]:
    """YouTube 爬取 (国内需代理)"""
    print("\n[3/5] 爬取 YouTube 视频...")

    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  ⚠ 未安装 yt-dlp: pip install yt-dlp")
        return []

    yt_dir = os.path.join(LARGE_DATASET_DIR, "youtube")
    for pet in ["cat", "dog"]:
        os.makedirs(os.path.join(yt_dir, pet), exist_ok=True)

    all_downloaded = []
    videos_per_keyword = max(5, target // (sum(len(v) for v in YOUTUBE_KEYWORDS.values()) * 50))

    for pet_type, keywords in YOUTUBE_KEYWORDS.items():
        for kw in keywords:
            out_dir = os.path.join(yt_dir, pet_type)
            cmd = [
                "yt-dlp", f"ytsearch{videos_per_keyword}:{kw}",
                "-x", "--audio-format", "wav",
                "--no-playlist", "--no-overwrites",
                "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
            ]
            try:
                subprocess.run(cmd, capture_output=True, timeout=180)
                for f in os.listdir(out_dir):
                    if f.endswith(".wav"):
                        fpath = os.path.join(out_dir, f)
                        if fpath not in [s.get("audio_path", "") for s in all_downloaded]:
                            all_downloaded.append({
                                "pet_type": pet_type, "audio_path": fpath,
                                "source": "youtube", "query": kw,
                            })
            except Exception:
                pass
            time.sleep(2)

    print(f"  ✓ YouTube: {len(all_downloaded)} 个视频")
    return all_downloaded


# =====================================
# 4. 音频切分
# =====================================

def split_and_normalize(input_path: str, output_dir: str, segment_len: float = 2.0, sr: int = 16000) -> List[str]:
    """切分为 2 秒训练片段"""
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return []

    try:
        audio, _ = librosa.load(input_path, sr=sr, mono=True)
    except Exception:
        return []

    segment_samples = int(segment_len * sr)
    segments = []
    for i in range(0, len(audio) - segment_samples, segment_samples):
        segment = audio[i:i + segment_samples]
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.01:
            continue
        out_name = f"{Path(input_path).stem}_{i//segment_samples:04d}.wav"
        out_path = os.path.join(output_dir, out_name)
        sf.write(out_path, segment, sr)
        segments.append(out_path)
    return segments


# =====================================
# 5. 数据增强
# =====================================

def augment_audio(input_path: str, output_dir: str, num: int = 10) -> List[str]:
    """10 倍数据增强"""
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return []

    try:
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
    except Exception:
        return []

    augmented = []
    for i in range(num):
        aug = audio.copy()
        if np.random.random() > 0.5:
            aug = librosa.effects.time_stretch(aug, rate=np.random.uniform(0.8, 1.2))
        if np.random.random() > 0.5:
            aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=np.random.uniform(-2, 2))
        if np.random.random() > 0.5:
            aug = aug + np.random.randn(len(aug)) * np.random.uniform(0.001, 0.01)
        if np.random.random() > 0.5:
            mask_len = np.random.randint(10, 50)
            start = np.random.randint(0, max(1, len(aug) - mask_len))
            aug[start:start + mask_len] = 0
        if np.random.random() > 0.5:
            aug = aug * np.random.uniform(0.5, 1.5)

        max_val = np.max(np.abs(aug)) + 1e-10
        aug = aug / max_val * 0.9

        out_path = os.path.join(output_dir, f"{Path(input_path).stem}_aug{i:02d}.wav")
        sf.write(out_path, aug.astype(np.float32), sr)
        augmented.append(out_path)
    return augmented


# =====================================
# 6. 合成数据降级方案 (国内网络全失败时)
# =====================================

def generate_synthetic_fallback(target: int = 10000) -> List[Dict]:
    """基于真实品种数据生成合成音频 (降级方案)"""
    print("\n[备选] 生成合成音频数据 (降级方案)...")
    syn_dir = os.path.join(LARGE_DATASET_DIR, "synthetic")
    samples = []

    # 加载真实品种数据
    cat_data = []
    dog_data = []
    for fn, target_list in [("cat_breeds_real.json", cat_data), ("dog_breeds_real.json", dog_data)]:
        path = os.path.join(DATA_DIR, "crawler", fn)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    target_list.extend(data if isinstance(data, list) else [data])
            except Exception:
                pass

    print(f"  真实品种数据: cat={len(cat_data)}, dog={len(dog_data)}")

    sr = 16000
    duration = 2.0
    samples_per_breed = max(10, target // max(1, len(cat_data) + len(dog_data)))

    try:
        import soundfile as sf
    except ImportError:
        print("  ⚠ 未安装 soundfile, 无法生成")
        return []

    # 生成猫叫声
    for breed_info in cat_data:
        breed = breed_info.get("breed_name", "unknown_cat")
        f0_range = breed_info.get("acoustic_signature", {}).get("f0_range", [300, 800])

        for i in range(samples_per_breed):
            # 生成基频
            f0 = np.random.uniform(f0_range[0], f0_range[1]) if f0_range else np.random.uniform(300, 800)
            t = np.linspace(0, duration, int(sr * duration))
            # 基础正弦波 + 谐波
            audio = np.sin(2 * np.pi * f0 * t) * 0.5
            audio += np.sin(2 * np.pi * f0 * 2 * t) * 0.2
            # 包络 (模拟喵叫起伏)
            envelope = np.exp(-((t - 1) ** 2) * 2)
            audio = audio * envelope
            # 加噪声
            audio = audio + np.random.randn(len(audio)) * 0.05
            # 归一化
            audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.9

            filename = f"synth_cat_{breed.replace(' ', '_')}_{i:04d}.wav"
            filepath = os.path.join(syn_dir, filename)
            sf.write(filepath, audio.astype(np.float32), sr)
            samples.append({
                "pet_type": "cat", "audio_path": filepath,
                "source": "synthetic", "breed": breed,
            })

    # 生成狗叫声
    for breed_info in dog_data:
        breed = breed_info.get("breed_name", "unknown_dog")
        f0_range = breed_info.get("acoustic_signature", {}).get("f0_range", [100, 300])

        for i in range(samples_per_breed):
            f0 = np.random.uniform(f0_range[0], f0_range[1]) if f0_range else np.random.uniform(100, 300)
            t = np.linspace(0, duration, int(sr * duration))
            # 狗叫: 低频 + 短促脉冲
            audio = np.sin(2 * np.pi * f0 * t) * 0.5
            audio += np.sin(2 * np.pi * f0 * 3 * t) * 0.3
            # 狗叫包络 (短促爆发)
            envelope = np.zeros_like(t)
            burst_starts = np.random.choice(range(0, int(sr * 1.5)), 3, replace=False)
            for bs in burst_starts:
                envelope[bs:bs + int(0.3 * sr)] = np.exp(-np.linspace(0, 5, int(0.3 * sr)))
            audio = audio * envelope
            audio = audio + np.random.randn(len(audio)) * 0.05
            audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.9

            filename = f"synth_dog_{breed.replace(' ', '_')}_{i:04d}.wav"
            filepath = os.path.join(syn_dir, filename)
            sf.write(filepath, audio.astype(np.float32), sr)
            samples.append({
                "pet_type": "dog", "audio_path": filepath,
                "source": "synthetic", "breed": breed,
            })

    print(f"  ✓ 合成数据: {len(samples)} 个样本")
    return samples


# =====================================
# 7. 主流程
# =====================================

def build_large_dataset(
    target: int = 1000000,
    proxy: str = "",
    skip_freesound: bool = False,
    skip_youtube: bool = False,
    skip_audioset: bool = False,
    fallback_synthetic: bool = False,
    safe_mode: bool = False,
):
    """构建大规模数据集"""
    ensure_dirs()

    print("=" * 60)
    print(f"  大规模宠物声音数据集采集 (v3 国内优化版)")
    print(f"  目标: {target:,} 样本")
    if proxy:
        print(f"  代理: {proxy}")
    if safe_mode:
        print(f"  安全模式: 仅国内可访问源")
    print("=" * 60)

    # 创建 session
    session = create_session(proxy=proxy)

    # 安全模式: 跳过国内受限资源
    if safe_mode:
        skip_youtube = True
        skip_freesound = False  # Freesound 爬虫模式可能可用
        print("\n  [安全模式] 跳过 YouTube, 仅尝试可访问源")

    all_samples = []
    download_failed = False

    # 1. AudioSet
    if not skip_audioset:
        try:
            audioset_samples = download_audioset(session, target_per_class=target // 10)
            all_samples.extend(audioset_samples)
            if not audioset_samples:
                download_failed = True
        except Exception as e:
            print(f"  ⚠ AudioSet 异常: {e}")
            download_failed = True

    # 2. Freesound
    if not skip_freesound:
        try:
            fs_samples = download_freesound(session, target=target // 20)
            all_samples.extend(fs_samples)
            if not fs_samples:
                download_failed = True
        except Exception as e:
            print(f"  ⚠ Freesound 异常: {e}")
            download_failed = True

    # 3. YouTube
    if not skip_youtube:
        try:
            yt_samples = crawl_youtube_videos(target=target // 5)
            all_samples.extend(yt_samples)
            if not yt_samples:
                download_failed = True
        except Exception as e:
            print(f"  ⚠ YouTube 异常: {e}")
            download_failed = True

    print(f"\n  原始下载样本: {len(all_samples)}")

    # 4. 降级: 合成数据
    if (download_failed or len(all_samples) < target // 10) and fallback_synthetic:
        print("\n  [降级模式] 网络受限, 生成合成数据补充...")
        syn_samples = generate_synthetic_fallback(target=target - len(all_samples))
        all_samples.extend(syn_samples)
        print(f"  ✓ 合成后总数: {len(all_samples)}")

    # 5. 切分 + 标准化
    print("\n[4/5] 切分 + 标准化音频...")
    final_samples = []
    for i, sample in enumerate(all_samples):
        if "audio_path" not in sample or not os.path.exists(sample["audio_path"]):
            continue
        pet_dir = os.path.join(LARGE_DATASET_DIR, "final", sample["pet_type"])
        segments = split_and_normalize(sample["audio_path"], pet_dir)
        for seg in segments:
            final_samples.append({
                "audio_path": seg, "pet_type": sample["pet_type"],
                "source": sample.get("source", "unknown"),
                "breed": sample.get("breed", "unknown"),
            })
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(all_samples)}, 切分: {len(final_samples)}")

    print(f"  ✓ 切分后: {len(final_samples)} 个 2 秒片段")

    # 6. 数据增强
    print("\n[5/5] 数据增强 (10x)...")
    enhanced_dir = os.path.join(LARGE_DATASET_DIR, "enhanced")
    enhanced_samples = []
    for i, sample in enumerate(final_samples):
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(final_samples)}")
        aug_paths = augment_audio(sample["audio_path"], enhanced_dir, num=10)
        for aug in aug_paths:
            enhanced_samples.append({
                "audio_path": aug, "pet_type": sample["pet_type"],
                "source": "augmented", "breed": sample.get("breed", "unknown"),
            })

    total = len(final_samples) + len(enhanced_samples)
    print(f"\n  ✓ 最终样本: {total:,} (原始 {len(final_samples)} + 增强 {len(enhanced_samples)})")

    # 保存索引
    index_path = os.path.join(LARGE_DATASET_DIR, "dataset_index.json")
    all_final = final_samples + enhanced_samples
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(all_final, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 索引: {index_path}")

    cat_n = len([s for s in all_final if s["pet_type"] == "cat"])
    dog_n = len([s for s in all_final if s["pet_type"] == "dog"])
    print(f"\n  猫: {cat_n:,}  |  狗: {dog_n:,}")

    if total < target:
        print(f"\n  ⚠ 未达目标 ({total} < {target:,})")
        print(f"  建议: 配置代理重新运行")
        print(f"    python3 data/crawler/collect_large_dataset.py --proxy=http://127.0.0.1:7890 --target={target}")

    print("\n" + "=" * 60)
    print("  完成!")
    print("=" * 60)


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = argparse.ArgumentParser(description="大规模宠物声音数据集采集 (v3)")
    parser.add_argument("--target", type=int, default=1000000, help="目标样本数 (默认 100 万)")
    parser.add_argument("--proxy", type=str, default="", help="HTTP 代理 (如 http://127.0.0.1:7890)")
    parser.add_argument("--freesound-key", type=str, default="", help="Freesound API key")
    parser.add_argument("--skip-freesound", action="store_true", help="跳过 Freesound")
    parser.add_argument("--skip-youtube", action="store_true", help="跳过 YouTube")
    parser.add_argument("--skip-audioset", action="store_true", help="跳过 AudioSet")
    parser.add_argument("--fallback-synthetic", action="store_true", help="失败时生成合成数据")
    parser.add_argument("--safe-mode", action="store_true", help="安全模式 (仅国内可访问源)")
    args = parser.parse_args()

    build_large_dataset(
        target=args.target,
        proxy=args.proxy,
        skip_freesound=args.skip_freesound,
        skip_youtube=args.skip_youtube,
        skip_audioset=args.skip_audioset,
        fallback_synthetic=args.fallback_synthetic,
        safe_mode=args.safe_mode,
    )
