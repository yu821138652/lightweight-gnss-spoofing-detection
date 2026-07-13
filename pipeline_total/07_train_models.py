
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
import yaml
from tqdm import tqdm

# Imports - handle different run contexts
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from models.lstm import LSTMClassifier
    from models.mamba import SpatioTemporalMamba
    from models.st_mamba import STMamba
    from models.st_mamba_adaptive_core_residual import AdaptiveCoreResidualSTMamba
    from models.st_mamba_core_residual import CoreResidualSTMamba
    from models.st_mamba_gated import GatedSTMamba
    from models.transformer import TransformerClassifier
    from models.cnn import CNNClassifier
    from models.loss import FocalLoss
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path}")
    raise

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GNSSDataset(Dataset):
    def __init__(self, npz_path):
        """
        Load data from .npz file (created by dataset_builder.py)
        Expected keys: 'x', 'mask', 'y'
        """
        logging.info(f"Loading dataset from {npz_path}...")
        try:
            with np.load(npz_path) as data:
                self.x = torch.from_numpy(data['x']).float()
                self.mask = torch.from_numpy(data['mask']).bool() # [N, 64]
                self.y = torch.from_numpy(data['y']).long()  # [N, 64]
                
                # [Dynamic Analysis] Load metadata if available
                if 'is_dynamic' in data:
                    self.is_dynamic = torch.from_numpy(data['is_dynamic']).bool() # [N]
                else:
                    self.is_dynamic = torch.zeros(len(self.x), dtype=torch.bool)
                if 'device_id' in data:
                    self.device_id = torch.from_numpy(data['device_id']).long() # [N]
                else:
                    self.device_id = torch.full((len(self.x),), -1, dtype=torch.long)
                
            logging.info(f"Loaded input shape: {self.x.shape}, Labels shape: {self.y.shape}, Dynamic Ratio: {self.is_dynamic.float().mean():.2%}")
        except Exception as e:
            logging.error(f"Failed to load {npz_path}: {e}")
            raise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx], self.y[idx], self.is_dynamic[idx], self.device_id[idx]

def evaluate(model, dataloader, device, criterion, class_names=None, measure_speed=False):
    """
    Enhanced evaluation function with comprehensive metrics.
    Returns: loss, macro_f1, full_metrics_dict
    """
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    import time
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    all_is_dynamic = [] # [Dynamic Analysis]
    
    start_time = time.time()
    total_samples = 0
    
    with torch.no_grad():
        # Iterate over dataloader (supports 3 or 4 items)
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if len(batch) >= 4:
                x, mask, y = batch[0], batch[1], batch[2]
                is_dynamic = batch[3].to(device)
            else:
                x, mask, y = batch[0], batch[1], batch[2]
                is_dynamic = torch.zeros(x.size(0), dtype=torch.bool).to(device)
            
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            
            logits = model(x, mask) # [B, 64, C]
            
            # Application of mask for loss and metrics
            # Flatten everything
            logits_flat = logits.view(-1, logits.size(-1)) # [B*64, C]
            y_flat = y.view(-1)                            # [B*64]
            mask_flat = mask.view(-1)                      # [B*64]
            
            # Expand per-window metadata to the configured signal-slot count.
            is_dynamic_expanded = is_dynamic.unsqueeze(1).expand(-1, mask.size(1)).reshape(-1)
            
            # Select valid samples (real satellites)
            valid_logits = logits_flat[mask_flat]
            valid_targets = y_flat[mask_flat]
            valid_is_dynamic = is_dynamic_expanded[mask_flat]
            
            if len(valid_targets) > 0:
                loss = criterion(valid_logits, valid_targets)
                total_loss += loss.item() * len(valid_targets)
                
                probs = torch.softmax(valid_logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(valid_targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_is_dynamic.append(valid_is_dynamic.cpu().numpy())
                total_samples += len(valid_targets)
    
    elapsed_time = time.time() - start_time
    
    if len(all_preds) == 0:
        return 0.0, 0.0, {}
        
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    all_is_dynamic = np.concatenate(all_is_dynamic)
    
    # Core Metrics
    avg_loss = total_loss / len(all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # AUC - handle binary vs multiclass
    try:
        num_classes = all_probs.shape[1]
        if num_classes == 2:
            # Binary: use probability of positive class (class 1)
            macro_auc = roc_auc_score(all_targets, all_probs[:, 1])
        else:
            # Multiclass: use OVR
            macro_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except ValueError as e:
        logging.warning(f"AUC calculation failed: {e}")
        macro_auc = 0.0
    
    # Speed
    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0
    
    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    
    # [Dynamic Analysis] Split Metrics
    static_mask = ~all_is_dynamic
    dynamic_mask = all_is_dynamic
    
    f1_static = f1_score(all_targets[static_mask], all_preds[static_mask], average='macro', zero_division=0) if np.any(static_mask) else 0.0
    f1_dynamic = f1_score(all_targets[dynamic_mask], all_preds[dynamic_mask], average='macro', zero_division=0) if np.any(dynamic_mask) else 0.0
    
    # Full Metrics Dict
    full_metrics = {
        'loss': avg_loss,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_auc': macro_auc,
        'samples_per_sec': samples_per_sec,
        'f1_static': f1_static,
        'f1_dynamic': f1_dynamic,
        'per_class': {k: v for k, v in report.items() if k in class_names} if class_names else {}
    }
    
    return avg_loss, macro_f1, full_metrics

def main():
    parser = argparse.ArgumentParser(description="Train GNSS Spoofing Detection")
    parser.add_argument('--config', type=str, default='configs/tensor_config.yml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'mamba', 'st_mamba', 'st_mamba_gated', 'st_mamba_core_residual', 'st_mamba_adaptive_core_residual', 'transformer', 'cnn'], help='Model type')
    parser.add_argument('--quick', action='store_true', help='Run extremely short training for verification')
    parser.add_argument('--log_file', type=str, default=None, help='Path to save training log')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (epochs)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate') 
    parser.add_argument('--n_layer_time', type=int, default=1, help='Number of temporal Mamba layers for ST-Mamba variants')
    parser.add_argument('--n_layer_space', type=int, default=3, help='Number of spatial Mamba layers for ST-Mamba variants; 0 disables spatial scanning')
    parser.add_argument('--disable_cross_sat_context', action='store_true', help='Disable cross-satellite correlation and consensus residual context')
    parser.add_argument('--feature_dropout', type=float, default=0.0, help='Training-only probability of dropping each input feature channel')
    parser.add_argument('--feature_noise_std', type=float, default=0.0, help='Training-only Gaussian noise std added to input features')
    parser.add_argument('--gate_l1', type=float, default=0.0, help='L1 penalty on learnable feature gates for gated models')
    parser.add_argument('--risk_gate_init', type=float, default=-2.0, help='Initial logit for risk residual gate in core-residual models')
    parser.add_argument('--group_loss', choices=['none', 'worst_device'], default='none', help='Optional source-device-aware training objective')
    parser.add_argument('--group_dro_eta', type=float, default=0.0, help='If >0, use soft worst-device weighting with this temperature')
    parser.add_argument('--metrics_name', type=str, default=None, help='Optional metrics filename stem after metrics_')
    args = parser.parse_args()
    
    # Setup file logging if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

    # Load Config
    if not Path(args.config).exists():
        # Try finding config in local directory if not found
        fallback_path = Path('configs/tensor_config.yml')
        if fallback_path.exists():
            logging.warning(f"Config not found at {args.config}, using {fallback_path}")
            args.config = str(fallback_path)
        else:
            # Last resort: try ../
            root_cfg = Path('../TensorPipeline/configs/tensor_config.yml')
            if root_cfg.exists():
                 args.config = str(root_cfg)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    # Handle relative paths better
    output_dir = Path(config['paths']['output_dir'])
    train_path = output_dir / 'train.npz'
    val_path = output_dir / 'val.npz'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Dataset
    train_dataset = GNSSDataset(train_path)
    val_dataset = GNSSDataset(val_path)
    
    if args.quick:
        logging.warning("⚡ Quick Mode: Trimming datasets to 2 batches...")
        indices = torch.arange(args.batch_size * 2)
        train_dataset.x = train_dataset.x[indices]
        train_dataset.mask = train_dataset.mask[indices]
        train_dataset.y = train_dataset.y[indices]
        train_dataset.is_dynamic = train_dataset.is_dynamic[indices]
        train_dataset.device_id = train_dataset.device_id[indices]
        val_dataset.x = val_dataset.x[indices]
        val_dataset.mask = val_dataset.mask[indices]
        val_dataset.y = val_dataset.y[indices]
        val_dataset.is_dynamic = val_dataset.is_dynamic[indices]
        val_dataset.device_id = val_dataset.device_id[indices]
        args.epochs = 2

    # [Optimization] Dataloader Performance
    # Determine num_workers: 0 for Windows (safe), 8-16 for Linux Server
    if os.name == 'nt':
        workers = 0
    else:
        workers = min(16, os.cpu_count() or 8) # Use up to 16 workers on server
    
    logging.info(f"DataLoader Config: num_workers={workers}, pin_memory=True")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=workers, 
        pin_memory=True, # Faster transfer to CUDA
        persistent_workers=(workers > 0) # Maintain workers alive
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=workers, 
        pin_memory=True,
        persistent_workers=(workers > 0)
    )

    # Model
    num_classes = config['labeling'].get('num_classes', 2)  # Binary classification
    
    # [Fix] Auto-detect input dimension from data instead of hardcoding
    # Expected: 7 (Basic) or 10 (V4 with CrossSat) or 8 (V4 w/o specific mask/etc)
    if hasattr(train_dataset, 'x') and len(train_dataset.x) > 0:
        input_dim = train_dataset.x.shape[-1]
        logging.info(f"Auto-detected input_dim={input_dim} from dataset")
    else:
        input_dim = 10 # Default fallback for V4
        logging.warning(f"Could not detect input_dim, using default {input_dim}")
    
    if args.model == 'lstm':
        model = LSTMClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    elif args.model == 'mamba':
        # Mamba Configuration
        model = SpatioTemporalMamba(
            input_dim=input_dim, 
            d_model=64, 
            n_layer=2, 
            num_classes=num_classes
        ).to(device)
    elif args.model == 'st_mamba':
        # [Aggressive Optimization] ST-Mamba V3: Larger capacity
        model = STMamba(
            input_dim=input_dim,
            d_model=128,          # Upgraded from 64
            n_layer_time=args.n_layer_time,
            n_layer_space=args.n_layer_space,      # Upgraded from 2
            num_classes=num_classes,
            dropout=args.dropout,
            use_cross_sat_context=not args.disable_cross_sat_context,
        ).to(device)
    elif args.model == 'st_mamba_gated':
        model = GatedSTMamba(
            input_dim=input_dim,
            d_model=128,
            n_layer_time=args.n_layer_time,
            n_layer_space=args.n_layer_space,
            num_classes=num_classes,
            dropout=args.dropout,
            use_cross_sat_context=not args.disable_cross_sat_context,
        ).to(device)
    elif args.model == 'st_mamba_core_residual':
        model = CoreResidualSTMamba(
            input_dim=input_dim,
            d_model=128,
            n_layer_time=args.n_layer_time,
            n_layer_space=args.n_layer_space,
            num_classes=num_classes,
            dropout=args.dropout,
            risk_gate_init=args.risk_gate_init,
            use_cross_sat_context=not args.disable_cross_sat_context,
        ).to(device)
    elif args.model == 'st_mamba_adaptive_core_residual':
        model = AdaptiveCoreResidualSTMamba(
            input_dim=input_dim,
            d_model=128,
            n_layer_time=args.n_layer_time,
            n_layer_space=args.n_layer_space,
            num_classes=num_classes,
            dropout=args.dropout,
            risk_gate_init=args.risk_gate_init,
            use_cross_sat_context=not args.disable_cross_sat_context,
        ).to(device)
    elif args.model == 'transformer':
        # Transformer Configuration
        model = TransformerClassifier(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            num_classes=num_classes
        ).to(device)
    elif args.model == 'cnn':
        # CNN Configuration
        model = CNNClassifier(
            input_dim=input_dim,
            num_classes=num_classes
        ).to(device)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    logging.info(f"Model initialized: {args.model.upper()}")

    # Loss & Optimizer & Scheduler
    # [Improvement] Label Smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # [Aggressive Optimization] AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # [Aggressive Optimization] Warmup + Cosine Decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training Loop
    best_f1 = 0.0
    patience_counter = 0  # Early stopping counter
    class_names = ['Authentic', 'Spoofing']  # Binary classification

    def augment_input(x):
        if args.feature_dropout > 0:
            keep = torch.rand(1, 1, 1, x.size(-1), device=x.device) >= args.feature_dropout
            x = x * keep.float()
        if args.feature_noise_std > 0:
            x = x + torch.randn_like(x) * args.feature_noise_std
        return x
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            if len(batch) >= 5:
                x, mask, y, group_ids = batch[0], batch[1], batch[2], batch[4]
            elif len(batch) >= 4:
                x, mask, y, group_ids = batch[0], batch[1], batch[2], None
            else:
                x, mask, y = batch[0], batch[1], batch[2]
                group_ids = None
            
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            if group_ids is not None:
                group_ids = group_ids.to(device)
            x = augment_input(x)
            
            optimizer.zero_grad()
            logits = model(x, mask)
            
            # Flatten and filter
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            mask_flat = mask.view(-1)
            
            valid_logits = logits_flat[mask_flat]
            valid_targets = y_flat[mask_flat]
            
            if len(valid_targets) == 0:
                continue
                
            if args.group_loss == 'worst_device' and group_ids is not None:
                token_group_ids = group_ids.unsqueeze(1).expand(-1, mask.size(1)).reshape(-1)[mask_flat]
                token_losses = F.cross_entropy(
                    valid_logits,
                    valid_targets,
                    label_smoothing=args.label_smoothing,
                    reduction='none',
                )
                valid_group_ids = token_group_ids[token_group_ids >= 0].unique()
                if len(valid_group_ids) > 0:
                    group_losses = torch.stack([
                        token_losses[token_group_ids == group_id].mean()
                        for group_id in valid_group_ids
                    ])
                    if args.group_dro_eta > 0:
                        weights = torch.softmax(args.group_dro_eta * group_losses.detach(), dim=0)
                        loss = (weights * group_losses).sum()
                    else:
                        loss = group_losses.max()
                else:
                    loss = token_losses.mean()
            else:
                loss = criterion(valid_logits, valid_targets)
            if args.gate_l1 > 0 and hasattr(model, 'gate_values'):
                loss = loss + args.gate_l1 * model.gate_values().mean()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        val_loss, val_f1, val_metrics = evaluate(model, val_loader, device, criterion, class_names)
        
        # [Improvement] Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logging.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}, LR={current_lr:.6f}")
        logging.info(f"  [Dynamic] Static F1: {val_metrics.get('f1_static', 0.0):.4f}, Dynamic F1: {val_metrics.get('f1_dynamic', 0.0):.4f}")
        logging.info(f"  Precision={val_metrics.get('macro_precision', 0.0):.4f}, Recall={val_metrics.get('macro_recall', 0.0):.4f}")
        logging.info(f"Class F1s: { {k: round(v['f1-score'],3) for k,v in val_metrics['per_class'].items()} }")
        
        
        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0  # Reset counter
            
            # [Feature] Configurable Checkpoint Path (for Exp E)
            ckpt_dir = Path(config.get('paths', {}).get('checkpoint_dir', output_dir))
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            save_name = config.get('training', {}).get('save_name', f'best_{args.model}')
            if not save_name.endswith('.pth'): save_name += '.pth'
            
            ckpt_path = ckpt_dir / save_name
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"🔥 New Best F1! Saved to {ckpt_path}")
        else:
            patience_counter += 1
            logging.info(f"No improvement. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping check
        if patience_counter >= args.patience:
            logging.info(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            break

    # --- Final Testing ---
    logging.info("\n>>> Starting Testing Phase <<<")
    
    # Load Best Model
    # [Fix] Use the correct checkpoint directory (might be different from output_dir in Exp E)
    ckpt_dir = Path(config.get('paths', {}).get('checkpoint_dir', output_dir))
    save_name = config.get('training', {}).get('save_name', f'best_{args.model}')
    if not save_name.endswith('.pth'): save_name += '.pth'
    
    ckpt_path = ckpt_dir / save_name
    
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))
        logging.info(f"Loaded best checkpoint from {ckpt_path}")
    else:
        # Fallback to output_dir
        fallback_ckpt = output_dir / save_name
        if fallback_ckpt.exists():
            model.load_state_dict(torch.load(fallback_ckpt))
            logging.info(f"Loaded best checkpoint from {fallback_ckpt}")
        else:
            logging.warning(f"No checkpoint found at {ckpt_path} or {fallback_ckpt}! Testing with last epoch weights.")
    
    # Load Test Data
    test_path = output_dir / 'test.npz'
    if test_path.exists():
        import json
        
        test_dataset = GNSSDataset(test_path)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Count model parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        test_loss, test_f1, test_metrics = evaluate(model, test_loader, device, criterion, class_names)
        
        # Rich Logging
        logging.info(f"=" * 60)
        logging.info(f"📊 Test Set Results for {args.model.upper()}")
        logging.info(f"=" * 60)
        logging.info(f"  Macro F1:        {test_metrics['macro_f1']:.4f}")
        logging.info(f"  Macro Precision: {test_metrics['macro_precision']:.4f}")
        logging.info(f"  Macro Recall:    {test_metrics['macro_recall']:.4f}")
        logging.info(f"  Macro AUC:       {test_metrics['macro_auc']:.4f}")
        logging.info(f"  Inference Speed: {test_metrics['samples_per_sec']:.1f} samples/sec")
        logging.info(f"  [Dynamic] Static F1: {test_metrics.get('f1_static', 0.0):.4f}")
        logging.info(f"  [Dynamic] Dynamic F1: {test_metrics.get('f1_dynamic', 0.0):.4f}")
        logging.info(f"  Model Params:    {num_params / 1e6:.2f}M")
        logging.info(f"-" * 60)
        logging.info(f"  Per-Class F1: {  {k: round(v['f1-score'],3) for k,v in test_metrics['per_class'].items()}  }")
        
        # Add params to metrics
        test_metrics['num_params'] = num_params
        test_metrics['model'] = args.metrics_name or args.model
        test_metrics['base_model'] = args.model
        test_metrics['group_loss'] = args.group_loss
        test_metrics['group_dro_eta'] = args.group_dro_eta
        test_metrics['n_layer_time'] = args.n_layer_time
        test_metrics['n_layer_space'] = args.n_layer_space
        test_metrics['use_cross_sat_context'] = not args.disable_cross_sat_context
        
        # Save to JSON (for radar chart)
        metrics_name = args.metrics_name or args.model
        json_path = output_dir / f'metrics_{metrics_name}.json'
        with open(json_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logging.info(f"📁 Metrics saved to {json_path}")
        
        # Also append to summary txt
        with open(output_dir / 'test_results.txt', 'a') as f:
            f.write(f"Model: {args.model}, F1: {test_f1:.4f}, AUC: {test_metrics['macro_auc']:.4f}\n")
    else:
        logging.warning("test.npz not found, skipping testing.")

if __name__ == '__main__':
    main()
