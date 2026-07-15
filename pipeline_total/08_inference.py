import argparse
import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Import models
from models.lstm import LSTMClassifier
from models.mamba import SpatioTemporalMamba
from models.st_mamba import STMamba
from models.transformer import TransformerClassifier
from models.cnn import CNNClassifier

# Constants aligned with the signal-level tensor builder.
DEFAULT_MAX_SIGNALS = 128
TIME_STEPS = 5
FEATURE_COLS = [
    'Cn0DbHz', 'Cn0DbHz_dt', 'Cn0DbHz_std',
    'AgcDb', 'ReceivedSvTimeUncertaintyNanos',
    'PseudorangeRateUncertaintyMetersPerSecond',
    'FreqBand',
]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class GNSSInferenceDataset(Dataset):
    """
    Dataset that returns (Tensor, Valid_Mask, Original_Indices)
    Original_Indices: [max_signals, 5] matrix of dataframe indices (or -1 if padding)
    """
    def __init__(self, tensor_list, mask_list, index_list):
        self.tensors = tensor_list
        self.masks = mask_list
        self.indices = index_list
        
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.tensors[idx]).float(), 
                torch.from_numpy(self.masks[idx]).bool(),
                torch.from_numpy(self.indices[idx]).long())

def load_model(model_name, ckpt_path, device, input_dim, num_classes):
    if model_name == 'lstm':
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes)
    elif model_name == 'mamba':
        model = SpatioTemporalMamba(input_dim=input_dim, d_model=64, n_layer=2, num_classes=num_classes)
    elif model_name == 'st_mamba':
        model = STMamba(input_dim=input_dim, d_model=64, n_layer_time=1, n_layer_space=2, num_classes=num_classes)
    elif model_name == 'transformer':
        model = TransformerClassifier(input_dim=input_dim, d_model=64, num_layers=2, num_classes=num_classes)
    elif model_name == 'cnn':
        model = CNNClassifier(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    logging.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def preprocess_features(df):
    logging.info("Feature Engineering...")
    for col in FEATURE_COLS:
        if col not in df.columns:
            logging.warning("Feature %s missing in CSV; filling with 0.0", col)
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


def resolve_identity_column(df):
    if 'signal_id' in df.columns:
        return 'signal_id'
    if 'sv_id' in df.columns:
        logging.warning("signal_id is absent; using sv_id for legacy inference.")
        return 'sv_id'
    raise ValueError("Inference CSV must contain signal_id or sv_id.")

def apply_normalization(df, scaler_path):
    if not Path(scaler_path).exists():
        logging.warning(f"Scaler not found at {scaler_path}. Using Identity.")
        return df

    with open(scaler_path, 'r') as f:
        device_stats = json.load(f)
        
    logging.info("Applying Normalization...")
    chunks = []
    global_mean = df[FEATURE_COLS].mean()
    global_std = df[FEATURE_COLS].std().replace(0, 1.0)
    
    for device, group in df.groupby('DeviceName'):
        stats = device_stats.get(device)
        if stats:
            means = pd.Series({c: stats[c]['mean'] for c in FEATURE_COLS})
            stds = pd.Series({c: stats[c]['std'] for c in FEATURE_COLS})
        else:
            means = global_mean
            stds = global_std
            
        group_copy = group.copy()
        for col in FEATURE_COLS:
            group_copy[col] = (group[col] - means[col]) / stds[col]
        chunks.append(group_copy)
        
    return pd.concat(chunks).loc[df.index] # Maintain original order

def generate_inference_batches(df, identity_column, max_signals):
    """
    Generator that yields batches of (Tensor, Mask, Indices)
    Indices maps each tensor element back to df.index
    """
    logging.info("Preparing Temporal Sessions...")
    
    # 1. Assign Temp Session IDs (Gap > 2s)
    df['temp_sid'] = -1
    sid_counter = 0
    
    # Identify sessions
    # Optimize: assume data is roughly sorted or handle by device
    df_sorted = df.sort_values(['DeviceName', 'TimeNanos'])
    
    # We define a generator to avoid holding all tensors in memory if data is huge
    # But for "dataset_builder" scale it was fine. 
    # Let's process per-device to save memory and yield lists
    
    groups = df_sorted.groupby('DeviceName')
    
    X_buf, M_buf, I_buf = [], [], []
    BATCH_SIZE = 1024 # Accumulate this many windows before yielding to save RAM
    
    for device, group in groups:
        times = group['TimeNanos'].values
        indices = group.index.values # Original indices
        
        # Split into sessions
        diffs = np.diff(times, prepend=times[0])
        # Gap > 2s
        jump_indices = np.where(diffs > 2e9)[0]
        session_starts = np.insert(jump_indices, 0, 0)
        session_ends = np.append(jump_indices, len(times))
        
        # Process each session array-style
        arr_times = times
        arr_identities = group[identity_column].values
        arr_feats = group[FEATURE_COLS].values
        arr_indices = indices
        
        for i in range(len(session_starts)):
            start, end = session_starts[i], session_ends[i]
            if end - start < TIME_STEPS: continue
            
            s_times = arr_times[start:end]
            s_identities = arr_identities[start:end]
            s_feats = arr_feats[start:end]
            s_ind = arr_indices[start:end]
            
            unique_times = np.unique(s_times)
            if len(unique_times) < TIME_STEPS: continue
            
            # Sliding Window
            # Use searchsorted for speed
            for i_t in range(len(unique_times) - TIME_STEPS + 1):
                window_ts = unique_times[i_t : i_t + TIME_STEPS]
                t_start, t_end = window_ts[0], window_ts[-1]
                
                idx_start_win = np.searchsorted(s_times, t_start, side='left')
                idx_end_win = np.searchsorted(s_times, t_end, side='right')
                
                w_identities = s_identities[idx_start_win:idx_end_win]
                w_times = s_times[idx_start_win:idx_end_win]
                w_feats = s_feats[idx_start_win:idx_end_win]
                w_ind = s_ind[idx_start_win:idx_end_win]
                
                # Strict filter
                mask_in_window = np.isin(w_times, window_ts)
                if not np.any(mask_in_window): continue
                
                w_identities = w_identities[mask_in_window]
                w_times = w_times[mask_in_window]
                w_feats = w_feats[mask_in_window]
                w_ind = w_ind[mask_in_window]
                
                unique_identities = np.unique(w_identities)
                if len(unique_identities) > max_signals:
                    raise ValueError(
                        f"Window contains {len(unique_identities)} {identity_column} values, "
                        f"exceeding max_signals={max_signals}."
                    )
                
                x_tensor = np.zeros((max_signals, TIME_STEPS, len(FEATURE_COLS)), dtype=np.float32)
                mask_vec = np.zeros((max_signals,), dtype=bool)
                idx_map = np.full((max_signals, TIME_STEPS), -1, dtype=np.int64)
                
                t_indices_map = np.searchsorted(window_ts, w_times)
                
                for identity_idx, identity in enumerate(unique_identities):
                    identity_mask = (w_identities == identity)
                    x_tensor[identity_idx, t_indices_map[identity_mask], :] = w_feats[identity_mask]
                    idx_map[identity_idx, t_indices_map[identity_mask]] = w_ind[identity_mask]
                    mask_vec[identity_idx] = True
                    
                X_buf.append(x_tensor)
                M_buf.append(mask_vec)
                I_buf.append(idx_map)
                
                if len(X_buf) >= BATCH_SIZE:
                    yield np.stack(X_buf), np.stack(M_buf), np.stack(I_buf)
                    X_buf, M_buf, I_buf = [], [], []

    if X_buf:
        yield np.stack(X_buf), np.stack(M_buf), np.stack(I_buf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='st_mamba')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True, help='Raw Source CSV')
    parser.add_argument('--scaler_dir', type=str, default='output/tensor_data')
    parser.add_argument('--output_csv', type=str, default='final_inference_results.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-signals', type=int, default=DEFAULT_MAX_SIGNALS)
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()
    if args.max_signals <= 0:
        parser.error('--max-signals must be positive')
    if args.num_classes < 2:
        parser.error('--num-classes must be at least 2')
    
    # 1. Load Data
    logging.info("Loading Raw CSV...")
    df = pd.read_csv(args.csv)
    # Store original indices to map back
    # df.index is range(len(df)) usually
    
    # 2. Preprocess
    df_proc = preprocess_features(df.copy())
    df_norm = apply_normalization(df_proc, Path(args.scaler_dir) / 'device_scaler.json')
    
    # 3. Load Model
    identity_column = resolve_identity_column(df_norm)
    model = load_model(
        args.model, args.ckpt, args.device,
        input_dim=len(FEATURE_COLS), num_classes=args.num_classes,
    )
    
    # 4. Inference Loop & Aggregation
    # We need to accumulate probabilities for each row in df
    # Shape: [N_rows, num_classes] (probabilities) + [N_rows] (counts)
    
    # Use sparse update or direct numpy array? 
    # DF size might be large. Numpy array [N, 4] is efficient.
    # N rows, configurable classes
    N = len(df)
    
    # [Optimization] Use Torch for fast Scatter Max
    # accum_probs = np.zeros((N, 4), dtype=np.float32) -> accum_probs_t
    accum_probs_t = torch.zeros((N, args.num_classes), dtype=torch.float32, device=args.device)
    counts_t = torch.zeros((N,), dtype=torch.int32, device=args.device)
    
    logging.info("Running Inference with Sliding Windows (Max Aggregation)...")
    
    # Generate batches
    gen = generate_inference_batches(df_norm, identity_column, args.max_signals)
    
    for x_batch, mask_batch, idx_batch in gen:
        # x_batch: [B, max_signals, 5, feature_dim]
        # idx_batch: [B, max_signals, 5]
        
        x_t = torch.from_numpy(x_batch).to(args.device)
        m_t = torch.from_numpy(mask_batch).to(args.device)
        idx_t = torch.from_numpy(idx_batch).to(args.device) # [B, 64, 5]
        
        with torch.no_grad():
            logits = model(x_t, m_t)
            probs = torch.softmax(logits, dim=2)
            
        # Broadcast Probs to Time Dim
        # [B, S, 1, C] -> [B, S, 5, C]
        B, S, T = idx_t.shape
        probs_expanded = probs.unsqueeze(2).expand(-1, -1, T, -1) # [B, 64, 5, 4]
        
        # Flatten for scatter
        # We only care about valid indices (idx != -1)
        valid_mask = (idx_t != -1) # [B, 64, 5]
        
        indices_flat = idx_t[valid_mask].long() # [K]
        probs_flat = probs_expanded[valid_mask] # [K, 4]
        
        # [Critical Fix] Usage of scatter_reduce_ for MAX aggregation
        # We want to take the MAX probability across all windows that cover this point.
        # "Safety First": If any window says 90% spoof, the point is 90% spoof.
        # accum_probs_t[indices_flat] = max(accum_probs_t[indices_flat], probs_flat)
        
        # pytorch 1.12+ supports scatter_reduce_
        # indices_flat must be broadcasted for dim 1 (classes)
        # src: [K, 4]
        # index: [K] -> expand to [K, 4] ? No, scatter expects index to have same dim as src usually?
        # scatter_reduce(dim, index, src, reduce='amax')
        # index size mismatch if we don't expand index?
        # dim=0 is row dimension.
        
        # For multi-dim scatter, index needs to match src dimensions?
        # Actually simplest is to loop over classes 0..3 or expand index
        
        # Expand index to [K, 4]
        indices_expanded = indices_flat.unsqueeze(1).expand(-1, args.num_classes)
        
        accum_probs_t.scatter_reduce_(0, indices_expanded, probs_flat, reduce='amax', include_self=True)
        
        # Track counts (just to know which rows were touched)
        # counts_t.index_add_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.int32))
        # actually scatter_reduce 'amax' on counts too? Or just any non-zero
        counts_t.scatter_reduce_(0, indices_flat, torch.ones_like(indices_flat, dtype=torch.int32), reduce='amax', include_self=True)

    logging.info("Aggregating Results...")
    
    # Move to CPU
    final_probs = accum_probs_t.cpu().numpy()
    counts = counts_t.cpu().numpy()
    
    # Argmax
    pred_labels = np.argmax(final_probs, axis=1)
    
    # Construct Result DF
    logging.info("Constructing Output DataFrame...")
    
    # Select columns if they exist
    out_cols = [
        'TimeNanos', 'sv_id', 'signal_id', 'SignalBand', 'CodeType',
        'DeviceName', 'SpoofingType', 'FreqBand', 'Label', 'LabelStatus',
    ]
    available_cols = [c for c in out_cols if c in df.columns]
    
    res_df = df[available_cols].copy()
    res_df.rename(columns={'Label': 'groundtruth'}, inplace=True)
    
    res_df['pred_label'] = pred_labels
    res_df['confidence'] = np.max(final_probs, axis=1)
    res_df['prob_spoof_total'] = np.sum(final_probs[:, 1:], axis=1)
    
    # [New] Map Labels to Strings
    label_map = {0: 'Normal', 1: 'Spoofing'} if args.num_classes == 2 else {0: 'Normal', 1: 'L1', 2: 'L5', 3: 'Mix'}
    res_df['pred_class'] = res_df['pred_label'].map(label_map)
    if 'groundtruth' in res_df.columns:
         res_df['gt_class'] = res_df['groundtruth'].map(label_map)

    # Filter out rows that were never predicted
    valid_rows = counts > 0
    res_df = res_df[valid_rows]
    
    logging.info(f"Saving {len(res_df)} rows to {args.output_csv}...")
    res_df.to_csv(args.output_csv, index=False)
    
    # --- Post-Inference Statistics ---
    logging.info("="*60)
    logging.info("Performance Statistics (Post-Inference)")
    logging.info("="*60)
    
    if 'groundtruth' not in res_df.columns:
        logging.info("No ground-truth labels available; inference results saved without metrics.")
        return

    # Metrics calculation helper
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    
    y_true = res_df['groundtruth']
    y_pred = res_df['pred_label']
    
    # Global
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    logging.info(f"Global: Acc={acc:.4f}, Macro F1={f1:.4f}")
    
    # By Device
    if 'DeviceName' in res_df.columns:
        logging.info("-" * 40)
        logging.info("By Device:")
        for dev, group in res_df.groupby('DeviceName'):
            g_acc = accuracy_score(group['groundtruth'], group['pred_label'])
            g_f1 = f1_score(group['groundtruth'], group['pred_label'], average='macro')
            logging.info(f"  {dev:20s}: Acc={g_acc:.4f}, F1={g_f1:.4f} (N={len(group)})")
            
    # By SpoofingType (Experiment Group)
    if 'SpoofingType' in res_df.columns:
        logging.info("-" * 40)
        logging.info("By Experiment Group (SpoofingType):")
        for stype, group in res_df.groupby('SpoofingType'):
            g_acc = accuracy_score(group['groundtruth'], group['pred_label'])
            g_f1 = f1_score(group['groundtruth'], group['pred_label'], average='macro')
            logging.info(f"  {stype:20s}: Acc={g_acc:.4f}, F1={g_f1:.4f} (N={len(group)})")
            
    logging.info("Inference Complete.")
    
    # --- Debugging: Confusion Matrix & Data Check ---
    logging.info("="*60)
    logging.info("Debugging Report")
    logging.info("="*60)
    
    # 1. Check Feature Availability
    # Check if critical features are all zeros (indicating missing data)
    logging.info("[Data Check] Feature Statistics (First 1000 rows):")
    # We can check df_norm but it's local. Let's check df_proc used in generation
    # Actually df_norm is normalized. 
    # Let's check the result df if we saved features? No we didn't save features in res_df.
    # But we have df_proc in memory? No it was local to main / or overwritten.
    # `df_norm` is available in main scope.
    
    for col in FEATURE_COLS:
        # Check if column is essentially constant/zero in normalized df
        # If std is 1.0 and mean is 0, it's normalized.
        # But if raw was 0, norm is 0 (if scaler std=1). 
        # Better: check if unique values < 2 (likely constant)
        unique_vals = df_norm[col].nunique()
        if unique_vals < 2:
             logging.warning(f"⚠️ Feature '{col}' seems constant! (Unique: {unique_vals}). Is it missing in CSV?")
        else:
             pass # logging.info(f"Feature '{col}' OK (Unique: {unique_vals})")

    # 2. Label Distribution
    unique_gt = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    logging.info(f"Unique Ground Truth Labels: {unique_gt}")
    logging.info(f"Unique Predicted Labels:    {unique_pred}")
    
    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    class_labels = list(range(args.num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    logging.info("Confusion Matrix (Rows=GT, Cols=Pred):")
    logging.info(f"\n{cm}")
    logging.info("Labels: %s", class_labels)
    
    # Normalized CM
    cm_norm = np.divide(
        cm.astype(float), cm.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0,
    )
    logging.info("Normalized Confusion Matrix:")
    logging.info(f"\n{np.round(cm_norm, 2)}")


if __name__ == '__main__':
    main()
