"""
Training Example Script

Demonstrates how to train the diffusion-based trading system.
"""

import torch
import argparse
import yaml
from pathlib import Path
import wandb

import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategy.generator import DiffusionStrategyGenerator
from training.trainer import StrategyDistillationTrainer
from data.dataset import create_dataloaders, DataProcessor
from risk.constraints import RiskConstraints


def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Trading Model')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Directory containing training data')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(data_dir: str, config: dict):
    """Prepare training data."""
    data_dir = Path(data_dir)
    
    # Check if processed data exists
    train_path = data_dir / 'train_processed.csv'
    val_path = data_dir / 'val_processed.csv'
    
    if not train_path.exists() or not val_path.exists():
        print("Processed data not found. Downloading and processing...")
        
        processor = DataProcessor()
        
        # Download data (example with multiple symbols)
        symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        all_data = []
        
        for symbol in symbols:
            print(f"Downloading {symbol}...")
            df = processor.download_data(
                symbol=symbol,
                start_date=config.get('start_date', '2020-01-01'),
                end_date=config.get('end_date', '2024-01-01'),
                source=config.get('data_source', 'yahoo')
            )
            
            # Process data
            df = processor.process_ohlcv(df)
            df['symbol'] = symbol
            all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Split into train/val
        split_idx = int(len(combined_df) * 0.8)
        train_df = combined_df[:split_idx]
        val_df = combined_df[split_idx:]
        
        # Save processed data
        data_dir.mkdir(exist_ok=True, parents=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"Saved processed data to {data_dir}")
    
    return str(train_path), str(val_path)


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=config.get('project_name', 'alphapolicy'),
            config=config,
            name=config.get('run_name', 'diffusion_trading')
        )
    
    # Prepare data
    print("Preparing data...")
    train_path, val_path = prepare_data(args.data_dir, config)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        seq_length=config.get('seq_length', 100),
        action_seq_length=config.get('action_seq_length', 10)
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create risk constraints
    risk_constraints = RiskConstraints(
        max_position_size=config.get('max_position_size', 1.0),
        max_leverage=config.get('max_leverage', 3.0),
        max_drawdown=config.get('max_drawdown', 0.2),
        max_daily_loss=config.get('max_daily_loss', 0.05)
    )
    
    # Create model
    print("Creating model...")
    model = DiffusionStrategyGenerator(
        price_dim=5,
        indicator_dim=config.get('indicator_dim', 20),
        orderbook_dim=10,
        regime_dim=5,
        seq_length=config.get('seq_length', 100),
        action_dim=4,
        action_seq_length=config.get('action_seq_length', 10),
        hidden_dim=config.get('hidden_dim', 256),
        cond_dim=config.get('cond_dim', 128),
        num_timesteps=config.get('num_timesteps', 1000),
        constraints=risk_constraints
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = StrategyDistillationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5),
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=config.get('num_epochs', 100))
    
    print("Training completed!")
    
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    import pandas as pd
    main()

