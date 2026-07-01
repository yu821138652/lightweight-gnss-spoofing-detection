"""
run_labeling.py - Interactive labeling helper for 2022 data

Usage:
    python -m labeling.run_labeling --spoof_type dy_L1 --folder 2022.07.08semicircle
    
This will:
1. Parse all TXT files in the specified folder
2. Display CN0 and AGC plots interactively
3. You manually note the TOW intervals, then update config.py
"""
# Fix import path FIRST
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(_project_root))

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt

from labeling import config
from labeling.visualize import plot_features
from Addition.gnss_plotter.parser import parse_gnss_log
from Addition.gnss_plotter.feature_engineering import process_single_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# Define _THIS_DIR for relative pathing
_THIS_DIR = Path(__file__).parent

def main():
    parser = argparse.ArgumentParser(description="Interactive Labeling Tool")
    parser.add_argument('--spoof_type', required=True, help='Spoofing type (e.g., dy_L1, dy_L5)')
    parser.add_argument('--folder', required=True, help='Folder name (e.g., 2022.07.08semicircle)')
    parser.add_argument('--device', default=None, help='Device folder to filter (e.g., HUAWEI, Xiaomi_MI_8)')
    args = parser.parse_args()
    
    # Update config.ROOT_DATA_DIR and data_dir based on the new structure
    config.ROOT_DATA_DIR = (_THIS_DIR / "../Addition/ConstellationFormer/data_raw").resolve()
    data_dir = config.ROOT_DATA_DIR / args.spoof_type / args.folder
    
    # If device filter specified, narrow down to that subfolder
    if args.device:
        data_dir = data_dir / args.device
        logging.info(f"Filtering by device: {args.device}")
    
    if not data_dir.exists():
        logging.error(f"Directory not found: {data_dir}")
        logging.info(f"Available folders in {args.spoof_type}:")
        parent = config.ROOT_DATA_DIR / args.spoof_type
        if parent.exists():
            for d in parent.iterdir():
                if d.is_dir():
                    logging.info(f"  - {d.name}")
        return
    
    logging.info(f"Processing: {data_dir}")
    
    # Find all TXT files recursively
    txt_files = list(data_dir.rglob("gnss_log_*.txt")) + list(data_dir.rglob("log_mimir_*.txt"))
    
    if not txt_files:
        logging.warning("No TXT files found!")
        return
    
    logging.info(f"Found {len(txt_files)} TXT files")
    
    # Parse and concatenate
    all_dfs = []
    device_name = "Unknown"
    for txt_file in txt_files:
        logging.info(f"Parsing: {txt_file.name}")
        raw_df, model_raw = parse_gnss_log(txt_file)
        if raw_df is None or raw_df.empty:
            continue
        
        # Get device name from first file
        if model_raw and device_name == "Unknown":
            device_name = config.DEVICE_MODEL_MAP.get(model_raw, model_raw)
        
        # Feature engineering - THIS creates TOW column!
        logging.info(f"  Feature engineering...")
        processed_df = process_single_file(raw_df, args.spoof_type)
        if processed_df is not None and not processed_df.empty:
            # Diagnostic: Print TOW range for each file
            if 'TOW' in processed_df.columns:
                tow_min = processed_df['TOW'].min()
                tow_max = processed_df['TOW'].max()
                logging.info(f"  -> TOW range: {tow_min:.1f} ~ {tow_max:.1f} ({tow_max - tow_min:.1f}s)")
            all_dfs.append(processed_df)
    
    if not all_dfs:
        logging.error("No data parsed!")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Total rows: {len(combined_df)}")
    
    # Plot with Label=0 (unknown until you label)
    combined_df['Label'] = 0
    
    file_basename = f"{args.spoof_type}_{args.folder}"
    
    logging.info("Generating CN0/AGC plots (interactive mode)...")
    logging.info("=== LOOK AT THE PLOTS, NOTE THE TOW VALUES WHERE SPOOFING OCCURS ===")
    
    # Temporarily set to only plot CN0 and AGC
    original_features = config.FEATURES_TO_PLOT
    config.FEATURES_TO_PLOT = config.FEATURES_FOR_LABELING
    
    figs = plot_features(
        combined_df, 
        file_basename, 
        spoofing_type=args.spoof_type,
        device_name=device_name,
        interactive_show=True
    )
    
    config.FEATURES_TO_PLOT = original_features
    
    if figs:
        logging.info("")
        logging.info("="*60)
        logging.info("INSTRUCTIONS:")
        logging.info("1. Look at the plots for sudden changes in CN0/AGC")
        logging.info("2. Note down the TOW (x-axis) values where spoofing starts/ends")
        logging.info("3. Close the plots when done")
        logging.info("4. Update labeling/config.py SPOOFING_TOW_INTERVALS with:")
        logging.info(f'   "{args.spoof_type}_{args.folder}": [[START_TOW, END_TOW], ...],')
        logging.info("="*60)
        plt.show()
    else:
        logging.warning("No interactive figures generated")
    
    logging.info("Done!")

if __name__ == '__main__':
    main()
