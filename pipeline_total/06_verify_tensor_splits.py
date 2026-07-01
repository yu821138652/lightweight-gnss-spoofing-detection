
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# --- 核心配置 (必须与 dataset_builder 下达的指令保持一致) ---
WHALE_THRESHOLD = 0  # [Fix] 设为 0，因为我们要求强制肢解
IGNORE_INDEX = -100

def style_print(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def check_tensor_distribution(npz_path, split_name):
    """
    物理验证：检查 .npz 文件的内部真实分布
    """
    if not npz_path.exists():
        logger.error(f"❌ File not found: {npz_path}")
        return

    logger.info(f"📂 Loading {split_name} tensor: {npz_path.name} ...")
    try:
        data = np.load(npz_path)
    except Exception as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return

    # 兼容性处理 Key
    if 'y' in data:
        y = data['y']
        mask = data['mask']
        x = data['x']
    elif 'labels' in data:
        y = data['labels']
        mask = data['mask']
        x = data['data']
    else:
        logger.error(f"Unknown keys in npz: {list(data.keys())}")
        return
    
    # 1. 有效样本统计
    total_slots = y.size
    valid_slots = np.sum(mask)
    
    # 2. 标签分布 (忽略 -100)
    valid_labels = y[y != IGNORE_INDEX]
    unique, counts = np.unique(valid_labels, return_counts=True)
    dist = dict(zip(unique, counts))
    
    # 3. 计算比率
    total_valid = sum(counts)
    
    print(f"  Shape: {x.shape} (Features={x.shape[-1]})")
    
    # Feature Dim Check
    if x.shape[-1] == 10:
        print("  ✅ Feature Dim is 10 (Matches dataset_builder default)")
    elif x.shape[-1] == 8:
        print("  ⚠️ Feature Dim is 8 (Matches original plan)")
    else:
        print(f"  ❓ Feature Dim is {x.shape[-1]} (Unexpected)")

    print(f"  Satellites Utilization: {valid_slots / total_slots * 100:.2f}% (Active vs Padding)")
    print(f"  Label Distribution (Model View):")
    
    print(f"    {'Class':<10} | {'Count':<10} | {'Ratio':<10}")
    print(f"    {'-'*36}")
    
    class_names = {0: "Normal", 1: "L1 Spoof", 2: "L5 Spoof", 3: "Mix Spoof"}
    
    for cls in [0, 1, 2, 3]:
        count = dist.get(cls, 0)
        ratio = count / total_valid * 100 if total_valid > 0 else 0
        mark = "✅" if count > 0 else "❌ MISSING!"
        if cls == 2 and count == 0: mark = "💀 FATAL (L5 Missing)"
        if cls == 3 and count == 0: mark = "💀 FATAL (Mix Missing)"
        
        print(f"    {cls:<2} {class_names.get(cls, str(cls)):<10} | {count:<10} | {ratio:>6.2f}% {mark}")
    
    print(f"    {'-'*36}")
    print(f"    {'Total':<10} | {total_valid:<10} | 100.00%")

    if split_name in ['val', 'test'] and (dist.get(2, 0) == 0 or dist.get(3, 0) == 0):
        print("\n  🚩 [Analysis]: Evaluation/Test set is missing critical classes (L5 or Mix). Metric will be flawed.")

def check_logical_split_simulation(csv_path):
    """
    逻辑验证：精确复现 dataset_builder 的切分逻辑
    验证是否存在 'Zero Train' 现象 (即某类设备只出现在 Test，没出现在 Train)
    """
    style_print("LOGICAL SPLIT SIMULATION (Domain Shift Check)")
    
    if not Path(csv_path).exists():
        logger.error("CSV file not found, skipping logical check.")
        return

    logger.info("Reading CSV (This may take a moment)...")
    cols = ['DeviceName', 'TimeNanos', 'Label']
    # 尝试读取额外列帮助判断
    df_head = pd.read_csv(csv_path, nrows=1)
    if 'SpoofingType' in df_head.columns: cols.append('SpoofingType')
    
    df = pd.read_csv(csv_path, usecols=cols)
    
    logger.info("Simulating Session Assignment...")
    
    # 1. 复现 Session 分配和切分逻辑
    # 我们不真分配 Session ID，而是直接对每个 (Device, Label) 组应用切分规则
    
    stats = [] # (Device, Label, TrainCount, ValCount, TestCount)
    
    # 为了精确，我们必须按 Device 分组处理
    for device, group in tqdm(df.groupby('DeviceName'), desc="Analyzing Devices"):
        # 在 dataset_builder 中，一个 Session 是连续的时间段
        # 这里我们简化：假设我们要检查的是 (Device, Label) 粒度的覆盖
        
        # 再次分组 Label
        for label, sub_group in group.groupby('Label'):
            times = np.sort(sub_group['TimeNanos'].values)
            diffs = np.diff(times, prepend=times[0])
            # 找到 Session 边界
            jumps = np.where(diffs > 2e9)[0]
            starts = np.insert(jumps, 0, 0)
            ends = np.append(jumps, len(times))
            
            s_train = 0
            s_val = 0
            s_test = 0
            
            for i in range(len(starts)):
                s_len = ends[i] - starts[i]
                
                # [CORE CHECK] Apply WHALE_THRESHOLD = 0 Logic
                # 无论多长，都切分
                if s_len > WHALE_THRESHOLD:
                    n_train = int(s_len * 0.7)
                    n_val = int(s_len * 0.15)
                    n_test = s_len - n_train - n_val
                    
                    s_train += n_train
                    s_val += n_val
                    s_test += n_test
                else:
                    # 如果逻辑回退到 Hash (不应该发生，如果 WHALE_THRESHOLD=0)
                    # 但为了模拟旧逻辑的影响，这里只模拟 split
                    pass
            
            stats.append({
                'Device': device, 
                'Label': label,
                'Train': s_train,
                'Val': s_val,
                'Test': s_test
            })
            
    # 2. 打印分析报告
    stats_df = pd.DataFrame(stats)
    
    print("\n[Domain Shift Forensic Report]")
    print(f"Goal: Ensure every (Device, Label) pair appears in TRAIN set.")
    print(f"{'Device':<20} | {'Label':<6} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Risk Level'}")
    print("-" * 80)
    
    risk_found = False
    
    for _, row in stats_df.iterrows():
        train_n = row['Train']
        test_n = row['Test']
        
        risk = "Safe"
        if train_n == 0 and test_n > 0:
            risk = "🚨 CRITICAL (Zero-Shot)" 
            risk_found = True
        elif train_n < 100 and test_n > 100:
            risk = "⚠️ High (Few-Shot)"
        
        # 只打印有风险的或者 Mix/L5 类别的
        if risk != "Safe" or row['Label'] in [2, 3]:
            print(f"{row['Device']:<20} | {row['Label']:<6} | {train_n:<8} | {row['Val']:<8} | {test_n:<8} | {risk}")
            
    if risk_found:
        print("\n❌ Conclusion: DATA DISASTER FOUND. Training set is missing specific Device/Label combos found in Test.")
        print("   Solution: Confirm WHALE_THRESHOLD=0 is applied and RE-RUN dataset_builder.")
    else:
        print("\n✅ Conclusion: Data Split looks healthy. No Zero-Shot Domain Shift detected.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', default='output/tensor_data', help='Directory containing train.npz/val.npz/test.npz')
    parser.add_argument('--csv', default='', help='Path to processed CSV (Required for logical check)')
    args = parser.parse_args()
    
    npz_dir = Path(args.npz_dir)
    
    # 1. 物理检查 (.npz)
    style_print("1. TENSOR FILE CHECK")
    check_tensor_distribution(npz_dir / 'train.npz', 'train')
    check_tensor_distribution(npz_dir / 'val.npz', 'val')
    check_tensor_distribution(npz_dir / 'test.npz', 'test')
    
    # 2. 逻辑模拟 (CSV)
    if args.csv:
        check_logical_split_simulation(args.csv)
    else:
        print("\n⚠️ Skipping CSV simulation. Please provide --csv to debug Domain Shift issues.")

if __name__ == '__main__':
    main()