"""
Training Script for Diffusion Policy

Trains the diffusion policy model on human trading data.
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys

from models.diffusion_policy import DiffusionPolicy
from data.market_data import MarketDataCollector
from data.strategy_dataset import load_human_trades, create_dataloaders
from data.preprocessor import MarketFeatureEngineer, ActionNormalizer


def setup_logger(log_file: str):
    """Setup logger configuration."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, rotation="1 day", retention="30 days", level="DEBUG")


def train_epoch(
    policy: DiffusionPolicy,
    dataloader,
    optimizer,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    policy.network.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (states, actions) in enumerate(pbar):
        states = states.to(device)
        actions = actions.to(device)
        
        # Compute loss
        loss = policy.compute_loss(actions, states)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.network.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    policy: DiffusionPolicy,
    dataloader,
    device: str
) -> float:
    """Validate the model."""
    policy.network.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            
            loss = policy.compute_loss(actions, states)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    setup_logger(log_dir / "training.log")
    
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize feature engineer and action normalizer
    feature_engineer = MarketFeatureEngineer(
        lookback_window=config['strategy']['lookback_window'],
        technical_indicators=config['strategy']['technical_indicators']
    )
    action_normalizer = ActionNormalizer()
    
    # Load human trading data
    logger.info(f"Loading human trades from {args.data_path}")
    trades_df = load_human_trades(args.data_path)
    
    # Load market data
    logger.info("Loading market data...")
    collector = MarketDataCollector(
        exchange_name=config['trading']['exchanges'][0]['name'],
        testnet=config['trading']['exchanges'][0]['testnet']
    )
    
    # Get market data for all symbols in trades
    symbols = trades_df['symbol'].unique()
    market_data = {}
    
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}")
        df = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=config['trading']['timeframe'],
            start_date=trades_df['timestamp'].min(),
            end_date=trades_df['timestamp'].max()
        )
        
        # Process features
        df = feature_engineer.process_dataframe(df, fit_scaler=(symbol == symbols[0]))
        market_data[symbol] = df
    
    # Combine market data
    combined_market_df = pd.concat(market_data.values(), keys=market_data.keys())
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        trades_df=trades_df,
        market_df=combined_market_df,
        feature_engineer=feature_engineer,
        action_normalizer=action_normalizer,
        batch_size=config['training']['batch_size'],
        train_split=config['training']['train_split'],
        val_split=config['training']['val_split']
    )
    
    # Calculate state dimension
    sample_state, _ = train_loader.dataset[0]
    state_dim = sample_state.shape[0]
    logger.info(f"State dimension: {state_dim}")
    
    # Initialize model
    logger.info("Initializing diffusion policy...")
    policy = DiffusionPolicy(
        state_dim=state_dim,
        action_dim=config['model']['action_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        num_diffusion_steps=config['model']['num_diffusion_steps'],
        beta_schedule=config['model']['beta_schedule'],
        device=device
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(
        policy.network.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Setup tensorboard
    writer = SummaryWriter(config['training']['tensorboard_dir'])
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info("Starting training loop...")
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train
        train_loss = train_epoch(policy, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(policy, val_loader, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(f"Epoch {epoch}/{config['training']['num_epochs']} - "
                   f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            policy.save(checkpoint_dir / "best_model.pt")
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['save_every'] == 0:
            policy.save(checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
            logger.info(f"Saved checkpoint at epoch {epoch}")
        
        # Update learning rate
        scheduler.step()
    
    # Test on test set
    logger.info("Evaluating on test set...")
    test_loss = validate(policy, test_loader, device)
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    writer.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diffusion Policy for Trading")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to human trading data CSV")
    
    args = parser.parse_args()
    
    import pandas as pd
    main(args)

