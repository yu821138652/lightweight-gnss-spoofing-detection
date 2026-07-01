import pandas as pd
import numpy as np
import torch
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- 核心超参 ---
MAX_SATS = 64
TIME_STEPS = 5
IGNORE_INDEX = -100  # Masked Label
WHALE_THRESHOLD = 0  # [Fix] 设为 0，强制所有 Session 都在内部按时间切分 (消除异域问题)

FEATURE_COLS = [
    'Cn0DbHz', 'Cn0DbHz_dt', 'Cn0DbHz_std',
    'AgcDb', 'ReceivedSvTimeUncertaintyNanos',
    'PseudorangeRateUncertaintyMetersPerSecond',
    'AccumulatedDeltaRangeUncertaintyMeters',
]

FEATURE_PRESETS = {
    'all': FEATURE_COLS,
    'cn0_only': ['Cn0DbHz', 'Cn0DbHz_dt', 'Cn0DbHz_std'],
    'no_agc': [
        'Cn0DbHz', 'Cn0DbHz_dt', 'Cn0DbHz_std',
        'ReceivedSvTimeUncertaintyNanos',
        'PseudorangeRateUncertaintyMetersPerSecond',
        'AccumulatedDeltaRangeUncertaintyMeters',
    ],
    'no_uncertainty': ['Cn0DbHz', 'Cn0DbHz_dt', 'Cn0DbHz_std', 'AgcDb'],
}

def stable_hash_mod(value, modulo=100):
    digest = hashlib.md5(str(value).encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % modulo

# ====================================================================
# 新增：场景和设备过滤功能 (支持实验 D/E)
# ====================================================================
def filter_by_scenario(df, scenario):
    """
    根据场景类型过滤数据
    
    Args:
        df: 输入 DataFrame
        scenario: 'dynamic' | 'static' | 'mixed' (或 None 表示不过滤)
    
    Returns:
        过滤后的 DataFrame
    """
    if scenario is None or scenario == 'mixed':
        logging.info(f"Scenario Filter: Using ALL data (mixed)")
        return df
    
    if 'SpoofingType' not in df.columns:
        logging.warning("SpoofingType column not found, skipping scenario filter")
        return df
    
    if scenario == 'dynamic':
        # [Fix] Strict filtering: Only data from dynamic folders (Authentic or Spoofing)
        mask = df['SpoofingType'].str.startswith('dy_')
        filtered_df = df[mask].copy()
        logging.info(f"Scenario Filter: DYNAMIC - kept {len(filtered_df)}/{len(df)} rows")
    elif scenario == 'static':
        # [Fix] Strict filtering: Only data from static folders (Authentic or Spoofing)
        mask = df['SpoofingType'].str.startswith('st_')
        filtered_df = df[mask].copy()
        logging.info(f"Scenario Filter: STATIC - kept {len(filtered_df)}/{len(df)} rows")
    else:
        logging.warning(f"Unknown scenario '{scenario}', using all data")
        return df
        
    # [Debug] Check what types are present
    if not filtered_df.empty:
        logging.info(f"  Types found: {filtered_df['SpoofingType'].unique()}")
    
    return filtered_df


def filter_by_device(df, devices):
    """
    根据设备列表过滤数据
    
    Args:
        df: 输入 DataFrame
        devices: 设备名称列表，如 ['HUAWEI_Mate40', 'Google_Pixel6']
                 或 None 表示不过滤
    
    Returns:
        过滤后的 DataFrame
    """
    if devices is None or len(devices) == 0:
        logging.info(f"Device Filter: Using ALL devices")
        return df
    
    if 'DeviceName' not in df.columns:
        logging.warning("DeviceName column not found, skipping device filter")
        return df
    
    filtered_df = df[df['DeviceName'].isin(devices)].copy()
    logging.info(f"Device Filter: Kept devices {devices} - {len(filtered_df)}/{len(df)} rows")
    
    return filtered_df


def preprocess_features(df):
    logging.info("--- 0. Feature Cleaning & Casting ---")
    
    # 1. 别名处理
    if 'Rate_Diff' in df.columns and 'Rate_Consistency' not in df.columns:
        df['Rate_Consistency'] = df['Rate_Diff']
    
    # 2. 几何特征 (已移除 - 不再使用 Az/El)
    # 如果 CSV 中有这些列，跳过即可

    # 3. [Fix Warning] 强制转 float32，避免 incompatible dtype
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # 4. 填充 NaN 和 缺失列
    # 必须确保 FEATURE_COLS 中的所有列都存在，否则后续 Tensor 构建会挂
    for col in FEATURE_COLS:
        if col not in df.columns:
            logging.warning(f"Feature {col} missing in CSV, filling with 0.0")
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 再次确保类型 (因为 fillna 可能导致类型变化)
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)

    return df

def perform_stratified_split(df, holdout_device=None):
    if holdout_device:
        logging.info(f"--- 1. LODO Split (Holdout: {holdout_device}) ---")
    else:
        logging.info("--- 1. Robust Stratified Split (Volume-Based) ---")
    
    # 1. 分配 Session ID
    df['session_id'] = -1
    global_session_counter = 0
    
    # 快速分配
    for device, group in tqdm(df.groupby('DeviceName'), desc="Assigning Sessions"):
        group = group.sort_values('TimeNanos')
        times = group['TimeNanos'].values
        indices = group.index.values
        
        diffs = np.diff(times, prepend=times[0])
        # 间隙 > 2秒 视为新 Session
        jump_indices = np.where(diffs > 2e9)[0] 
        
        session_starts = np.insert(jump_indices, 0, 0)
        session_ends = np.append(jump_indices, len(times))
        
        for i in range(len(session_starts)):
            s_idx = session_starts[i]
            e_idx = session_ends[i]
            df.loc[indices[s_idx:e_idx], 'session_id'] = global_session_counter
            global_session_counter += 1

    # 2. 统计每个 Session 的长度
    session_counts = df.groupby('session_id').size()
    df['split'] = 'skip'
    
    # 获取唯一的 (Label, Device) 组合
    # 注意：这里假设一个 Session 内 Label 是一致的 (取 max)
    session_meta = df.groupby('session_id').agg({
        'Label': 'max',
        'DeviceName': 'first'
    })
    
    # 3. 切分逻辑
    unique_sids = session_counts.index.values
    
    # [New] LODO: hold out one complete device for testing. The remaining
    # devices are split by deterministic time blocks inside each session, so
    # validation keeps a representative static/dynamic mix.
    if holdout_device:
        for sid in tqdm(unique_sids, desc="Splitting Sessions (LODO)"):
            meta = session_meta.loc[sid]
            device = meta['DeviceName']
            
            mask = df['session_id'] == sid
            indices = df.index[mask]
            
            if device == holdout_device:
                df.loc[indices, 'split'] = 'test'
            else:
                times = df.loc[indices, 'TimeNanos'].values
                sorted_arg = np.argsort(times)
                sorted_indices = indices[sorted_arg]
                total_samples = len(sorted_indices)
                assignments = np.full(total_samples, 'train', dtype=object)
                offset = stable_hash_mod(sid, modulo=10)

                block_size = 60
                num_blocks = (total_samples // block_size) + 1
                for b in range(num_blocks):
                    b_start = b * block_size
                    b_end = min((b + 1) * block_size, total_samples)
                    if b_start >= b_end:
                        continue
                    if (b + offset) % 10 in (7, 9):
                        assignments[b_start:b_end] = 'val'

                df.loc[sorted_indices, 'split'] = assignments
        
        # 统计结果
        split_counts = df['split'].value_counts()
        logging.info(f"LODO Split Result: {split_counts.to_dict()}")
        return df

    # --- 以下为原有的分层切分逻辑 ---
    for sid in tqdm(unique_sids, desc="Splitting Sessions"):
        count = session_counts[sid]
        meta = session_meta.loc[sid]
        label = meta['Label']
        device = meta['DeviceName']
        
        # 获取该 Session 的所有索引
        # (这里假设 session_id 是连续分配的，但为了安全用 boolean indexing)
        # 为了速度，我们已经在上面分配时保证了局部性，但这里为了逻辑简单：
        mask = df['session_id'] == sid
        indices = df.index[mask] # 原始索引
        
        # --- [Critical Fix] 强制大鲸鱼肢解 ---
        # 只要数据量大，就强制切分，不管它是唯一的还是多个之一
        if count > WHALE_THRESHOLD:
            # 必须先按时间排序索引
            # 获取该部分数据的 TimeNanos 并排序
            times = df.loc[indices, 'TimeNanos'].values
            sorted_arg = np.argsort(times)
            sorted_indices = indices[sorted_arg]
            
            # [Fix] 使用交错分块 (Interleave) 代替尾部切分，解决 L5 结尾掉点问题
            # 将数据按时间切分成小块 (例如每 60 个点一块)，模 10 分配
            # 0-6: Train (70%), 7: Val (10%), 8: Train (10%), 9: Test (10%) -> Train 80%, Val 10%, Test 10%
            # 调整为: 0-6 Train(70%), 7 Val(10%), 8 Test(10%), 9 Val(10%) -> Train 70%, Val 20%, Test 10%
            
            BLOCK_SIZE = 60 # 约 1 分钟
            total_samples = len(sorted_indices)
            assignments = np.full(total_samples, 'train', dtype=object)
            
            num_blocks = (total_samples // BLOCK_SIZE) + 1
            for b in range(num_blocks):
                b_start = b * BLOCK_SIZE
                b_end = min((b + 1) * BLOCK_SIZE, total_samples)
                if b_start >= b_end: continue
                
                mod_val = b % 10
                if mod_val == 7 or mod_val == 9: # 20% Val
                    assignments[b_start:b_end] = 'val'
                elif mod_val == 8: # 10% Test
                    assignments[b_start:b_end] = 'test'
                else:
                    assignments[b_start:b_end] = 'train'
            
            df.loc[sorted_indices, 'split'] = assignments
        else:
            # 小 Session：先标记为 'pending'，稍后统一按 Session 数量分配
            # 为了简化逻辑，这里我们可以做一个简单的随机分配：
            # 70% 概率进 Train，15% Val...
            # 但为了保证 (Class, Device) 的覆盖率，我们使用哈希取模实现确定性分配
            
            # 使用简单的 Hash 策略保证同一类设备分布均匀
            hash_val = stable_hash_mod(f"{sid}_{label}_{device}")
            if hash_val < 70:
                df.loc[indices, 'split'] = 'train'
            elif hash_val < 85:
                df.loc[indices, 'split'] = 'val'
            else:
                df.loc[indices, 'split'] = 'test'

    # 打印统计
    train_count = len(df[df['split']=='train'])
    val_count = len(df[df['split']=='val'])
    test_count = len(df[df['split']=='test'])
    
    logging.info(f"Split Result: Train={train_count}, Val={val_count}, Test={test_count}")
    
    if val_count < 1000:
        logging.warning("⚠️ Warning: Validation set is still very small! Check WHALE_THRESHOLD.")
        
    return df[df['split'] != 'skip'].copy()

def perform_per_device_norm(df, output_dir):
    logging.info("--- 2. Per-Device Normalization ---")
    train_df = df[df['split'] == 'train']
    
    device_stats = {}
    
    # 计算统计量
    for device, group in train_df.groupby('DeviceName'):
        stats = {}
        for col in FEATURE_COLS:
            col_mean = group[col].mean()
            col_std = group[col].std()
            # [Fix] 零方差处理：常数特征归一化为 0
            if np.isnan(col_std) or col_std < 1e-6:
                col_std = 1.0
                col_mean = group[col].iloc[0] if len(group) > 0 else 0.0  # 保持常数值减去自身=0
            stats[col] = {'mean': float(col_mean), 'std': float(col_std)}
        device_stats[device] = stats
    
    with open(Path(output_dir) / 'device_scaler.json', 'w') as f:
        json.dump(device_stats, f, indent=4)
        
    # 应用归一化 (Apply)
    # [Fix] 使用 transform 加速
    logging.info("Applying Normalization (Vectorized)...")
    
    # 预先构建映射表
    # 为了速度，我们将 df 按 Device 分组处理然后合并
    chunks = []
    
    # Global Fallback
    global_mean = train_df[FEATURE_COLS].mean()
    global_std = train_df[FEATURE_COLS].std().replace(0, 1.0)
    
    for device, group in df.groupby('DeviceName'):
        stats = device_stats.get(device)
        if stats:
            # 构建均值和方差 Series
            means = pd.Series({c: stats[c]['mean'] for c in FEATURE_COLS})
            stds = pd.Series({c: stats[c]['std'] for c in FEATURE_COLS})
        else:
            means = global_mean
            stds = global_std
            
        # 批量操作
        group_copy = group.copy()
        for col in FEATURE_COLS:
            group_copy[col] = (group[col] - means[col]) / stds[col]
        
        chunks.append(group_copy)
        
    return pd.concat(chunks)

def perform_global_norm(df, output_dir):
    logging.info("--- 2. Global Train-Only Normalization ---")
    train_df = df[df['split'] == 'train']
    global_mean = train_df[FEATURE_COLS].mean()
    global_std = train_df[FEATURE_COLS].std().replace(0, 1.0)

    global_stats = {
        col: {'mean': float(global_mean[col]), 'std': float(global_std[col])}
        for col in FEATURE_COLS
    }
    with open(Path(output_dir) / 'global_scaler.json', 'w') as f:
        json.dump(global_stats, f, indent=4)

    df_norm = df.copy()
    for col in FEATURE_COLS:
        df_norm[col] = (df_norm[col] - global_mean[col]) / global_std[col]
    return df_norm

def build_tensor_dataset(df, split_name, output_path, device_to_id=None):
    """
    [Core] 生成 Tensor - 纯 Numpy 版 (Fix Speed)
    """
    logging.info(f"--- 3. Building Tensor for {split_name} (Fast Mode) ---")
    
    # 仅保留需要的列，且转为 Numpy 能够快速处理的格式
    keep_cols = ['session_id', 'TimeNanos', 'sv_id', 'Label'] + FEATURE_COLS
    data_df = df[df['split'] == split_name][keep_cols].copy()
    
    # 按照 session_id 和 TimeNanos 排序 (关键)
    data_df = data_df.sort_values(['session_id', 'TimeNanos'])
    
    # 提取为 Numpy 数组以加速
    arr_session_ids = data_df['session_id'].values
    arr_times = data_df['TimeNanos'].values
    arr_sv_ids = data_df['sv_id'].values
    arr_labels = data_df['Label'].values
    arr_features = data_df[FEATURE_COLS].values
    
    # 找出 Session 的边界
    # 此时 arr_session_ids 是有序的
    unique_sids, idx_start, count = np.unique(arr_session_ids, return_index=True, return_counts=True)
    
    X_list = []
    Mask_list = []
    Y_list = []
    
    # [Dynamic Analysis] Metadata
    IsDynamic_list = [] # Bool
    DeviceId_list = []
    
    # Pre-compute is_dynamic for all sessions (to speed up loop)
    # df has columns 'SpoofingType'
    
    # Create lookup map
    # session_meta sorted by session_id
    meta_df = df.drop_duplicates('session_id').set_index('session_id').sort_index()
    if 'SpoofingType' in meta_df.columns:
        meta_is_dynamic_by_sid = {
            int(sid): str(spoof_type).startswith('dy')
            for sid, spoof_type in meta_df['SpoofingType'].items()
        }
    else:
        # Fallback if column missing
        meta_is_dynamic_by_sid = {}

    if device_to_id is not None and 'DeviceName' in meta_df.columns:
        meta_device_id_by_sid = {
            int(sid): int(device_to_id.get(str(device), -1))
            for sid, device in meta_df['DeviceName'].items()
        }
    else:
        meta_device_id_by_sid = {}
    
    # 遍历每个 Session (现在是纯 Numpy 循环，非常快)
    for i in tqdm(range(len(unique_sids)), desc=f"Processing {split_name}"):
        start = idx_start[i]
        cnt = count[i]
        end = start + cnt
        
        # 当前 Session 的数据切片
        s_times = arr_times[start:end]
        s_svs = arr_sv_ids[start:end]
        s_labels = arr_labels[start:end]
        s_feats = arr_features[start:end]
        
        # 找出唯一时间点
        unique_times = np.unique(s_times) # sorted
        if len(unique_times) < TIME_STEPS:
            continue
            
        # 时间 -> 索引映射 (加速查找)
        # 因为 unique_times 是有序的，我们可以用 searchsorted
        
        # 滑动窗口循环
        # i_t 是 unique_times 的下标
        for i_t in range(len(unique_times) - TIME_STEPS + 1):
            window_ts = unique_times[i_t : i_t + TIME_STEPS]
            
            # 找到属于当前窗口的行
            # 利用 searchsorted 找到起止范围 (因为 s_times 在 session 内并不一定严格按 sv 排列，
            # 但通常我们按 TimeNanos 排序了。不过同一个时间点有多个 SV，所以 s_times 会有重复值)
            # 更好的方法：直接利用 s_times 的值匹配
            
            # 使用 Numpy boolean masking (在小数组上很快)
            t_start, t_end = window_ts[0], window_ts[-1]
            
            # 这里的 mask 可能会慢，优化：
            # 利用 searchsorted 找 range
            idx_in_session_start = np.searchsorted(s_times, t_start, side='left')
            idx_in_session_end = np.searchsorted(s_times, t_end, side='right')
            
            w_svs = s_svs[idx_in_session_start:idx_in_session_end]
            w_times = s_times[idx_in_session_start:idx_in_session_end]
            w_feats = s_feats[idx_in_session_start:idx_in_session_end]
            w_labels = s_labels[idx_in_session_start:idx_in_session_end]
            
            # 再次过滤 (因为 searchsorted 是 range，可能包含中间缺失时间点的数据，虽然 window_ts 是连续抽取的)
            # 但我们需要精确匹配这 5 个时间点
            # 构造一个 mask
            mask_in_window = np.isin(w_times, window_ts)
            if not np.any(mask_in_window): continue
            
            w_svs = w_svs[mask_in_window]
            w_times = w_times[mask_in_window]
            w_feats = w_feats[mask_in_window]
            w_labels = w_labels[mask_in_window]
            
            # --- 填充 Tensor ---
            unique_sv_in_window = np.unique(w_svs)
            if len(unique_sv_in_window) > MAX_SATS:
                unique_sv_in_window = unique_sv_in_window[:MAX_SATS]
                
            x_tensor = np.zeros((MAX_SATS, TIME_STEPS, len(FEATURE_COLS)), dtype=np.float32)
            mask_vector = np.zeros((MAX_SATS,), dtype=bool)
            y_vector = np.full((MAX_SATS,), IGNORE_INDEX, dtype=int)
            
            # Map time to 0..4
            # w_times -> [0, 1, 2, 3, 4]
            # 依然用 searchsorted
            t_indices_map = np.searchsorted(window_ts, w_times)
            
            for sv_idx, sv_id in enumerate(unique_sv_in_window):
                # 选出该卫星的行
                sv_mask = (w_svs == sv_id)
                
                # 填 Label (Max)
                y_vector[sv_idx] = np.max(w_labels[sv_mask])
                mask_vector[sv_idx] = True
                
                # 填 Features
                # t_indices: 该卫星出现的时刻在 0..4 中的位置
                sv_t_ind = t_indices_map[sv_mask]
                sv_f_val = w_feats[sv_mask]
                
                # Fancy Indexing Assignment
                x_tensor[sv_idx, sv_t_ind, :] = sv_f_val
            
            X_list.append(x_tensor)
            Mask_list.append(mask_vector)
            Y_list.append(y_vector)
            
            # [Dynamic Analysis] Metadata
            curr_sid = arr_session_ids[start] 
            sid_val = int(unique_sids[i])
            is_dyn = meta_is_dynamic_by_sid.get(sid_val, False)
            IsDynamic_list.append(is_dyn)
            DeviceId_list.append(meta_device_id_by_sid.get(sid_val, -1))

    if not X_list:
        logging.warning(f"No data for {split_name}!")
        return
        
    X_final = np.stack(X_list)
    Mask_final = np.stack(Mask_list)
    Y_final = np.stack(Y_list)
    IsDynamic_final = np.array(IsDynamic_list, dtype=bool)
    DeviceId_final = np.array(DeviceId_list, dtype=np.int64)
    
    logging.info(f"{split_name} Ready. Shape: {X_final.shape}, Dynamic Ratio: {IsDynamic_final.mean():.2%}")
    np.savez_compressed(
        output_path,
        x=X_final,
        mask=Mask_final,
        y=Y_final,
        is_dynamic=IsDynamic_final,
        device_id=DeviceId_final,
    )

def main():
    parser = argparse.ArgumentParser(description='GNSS Dataset Builder with Filtering Support')
    parser.add_argument('--csv', required=False, help='Path to processed CSV file')
    parser.add_argument('--config', required=False, help='Path to YAML config file (overrides --csv)')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--scenario', choices=['dynamic', 'static', 'mixed'], 
                        default=None, help='Scenario filter (overrides config)')
    parser.add_argument('--devices', nargs='+', default=None, 
                        help='Device filter (overrides config)')
    # [New] LODO Support
    parser.add_argument('--holdout_device', type=str, default=None,
                        help='Device name to use as Test set (Leave-One-Device-Out)')
    parser.add_argument('--norm_mode', choices=['per_device', 'global'], default='per_device',
                        help='Normalization mode')
    parser.add_argument('--feature_preset', choices=sorted(FEATURE_PRESETS), default='all',
                        help='Feature subset preset for robustness/ablation tensor builds')
    args = parser.parse_args()
    global FEATURE_COLS
    FEATURE_COLS = FEATURE_PRESETS[args.feature_preset]
    logging.info(f"Feature preset: {args.feature_preset}; columns={FEATURE_COLS}")
    
    # =============================================
    # 1. 加载配置文件或命令行参数
    # =============================================
    config = {}
    if args.config:
        logging.info(f"Loading config from: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从配置中读取路径
        paths = config.get('paths', {})
        csv_path = paths.get('processed_csv_path', args.csv)
        
        # [Fix] 命令行参数优先级最高 (CLI > Config > Default)
        output_dir = args.output_dir or paths.get('output_dir') or 'output/tensor_data'
        
        # 从配置中读取过滤参数
        filtering = config.get('filtering', {})
        scenario = args.scenario or filtering.get('scenario', None)
        devices = args.devices or filtering.get('devices', None)
    else:
        if not args.csv:
            parser.error("Either --csv or --config must be provided")
        csv_path = args.csv
        output_dir = args.output_dir or 'output/tensor_data'
        scenario = args.scenario
        devices = args.devices
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # =============================================
    # 2. 加载数据
    # =============================================
    logging.info(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} rows")

    # [Fix] Enforce Binary Labels (0=Authentic, 1=Spoofing)
    # Even if CSV has 2, 3... convert to 1
    if 'Label' in df.columns:
        df.loc[df['Label'] > 0, 'Label'] = 1
        logging.info("Enforced Binary Labels: All labels > 0 set to 1")
    
    # =============================================
    # 3. 应用过滤 (实验 D/E 核心功能)
    # =============================================
    logging.info("=== Applying Filters ===")
    df = filter_by_scenario(df, scenario)
    df = filter_by_device(df, devices)
    
    if len(df) == 0:
        logging.error("No data left after filtering! Check your filter parameters.")
        return
    
    logging.info(f"After filtering: {len(df)} rows")
    
    # =============================================
    # 4. 标准处理流程
    # =============================================
    df = preprocess_features(df)
    df_split = perform_stratified_split(df, args.holdout_device)
    if args.norm_mode == 'global':
        df_norm = perform_global_norm(df_split, output_dir)
    else:
        df_norm = perform_per_device_norm(df_split, output_dir)

    device_to_id = None
    if 'DeviceName' in df_norm.columns:
        devices_for_mapping = sorted(str(d) for d in df_norm['DeviceName'].dropna().unique())
        device_to_id = {device: idx for idx, device in enumerate(devices_for_mapping)}
        with open(Path(output_dir) / 'device_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(device_to_id, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved device mapping: {device_to_id}")
    
    # =============================================
    # 5. 生成 Tensor 数据集
    # =============================================
    build_tensor_dataset(df_norm, 'train', Path(output_dir) / 'train.npz', device_to_id)
    build_tensor_dataset(df_norm, 'val', Path(output_dir) / 'val.npz', device_to_id)
    build_tensor_dataset(df_norm, 'test', Path(output_dir) / 'test.npz', device_to_id)
    
    # =============================================
    # 6. 可选：备份 CSV 到本地
    # =============================================
    if config:
        local_backup = config.get('paths', {}).get('local_csv_backup')
        if local_backup:
            backup_path = Path(local_backup)
            backup_path.mkdir(parents=True, exist_ok=True)
            backup_csv = backup_path / 'filtered_data.csv'
            df_norm.to_csv(backup_csv, index=False)
            logging.info(f"Saved filtered CSV backup to: {backup_csv}")
    
    logging.info("=== Dataset Building Complete ===")

if __name__ == '__main__':
    main()
