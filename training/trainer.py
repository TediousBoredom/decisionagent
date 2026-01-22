"""
Training Pipeline for Diffusion-Based Trading System

Implements the complete training loop with strategy distillation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from strategy.generator import DiffusionStrategyGenerator
from risk.constraints import RiskConstraints, RiskMetrics


class StrategyDistillationTrainer:
    """
    Trainer for distilling high-return strategies into diffusion model.
    """
    
    def __init__(
        self,
        model: DiffusionStrategyGenerator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Risk metrics calculator
        self.risk_metrics = RiskMetrics()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_return_loss = 0.0
        total_volatility_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            price = batch['price'].to(self.device)
            indicators = batch['indicators'].to(self.device)
            orderbook = batch['orderbook'].to(self.device)
            regime = batch['regime'].to(self.device)
            expert_actions = batch['actions'].to(self.device)
            returns = batch['returns'].to(self.device)
            volatility = batch['volatility'].to(self.device)
            
            # Forward pass
            losses = self.model.train_step(
                price, indicators, orderbook, regime,
                expert_actions, returns, volatility
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            total_diffusion_loss += losses['diffusion_loss'].item()
            total_return_loss += losses['return_loss'].item()
            total_volatility_loss += losses['volatility_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'diff': losses['diffusion_loss'].item(),
                'ret': losses['return_loss'].item()
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    'train/total_loss': losses['total_loss'].item(),
                    'train/diffusion_loss': losses['diffusion_loss'].item(),
                    'train/return_loss': losses['return_loss'].item(),
                    'train/volatility_loss': losses['volatility_loss'].item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches,
            'return_loss': total_return_loss / num_batches,
            'volatility_loss': total_volatility_loss / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_return_loss = 0.0
        total_volatility_loss = 0.0
        
        # For strategy quality metrics
        all_generated_returns = []
        all_expert_returns = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            price = batch['price'].to(self.device)
            indicators = batch['indicators'].to(self.device)
            orderbook = batch['orderbook'].to(self.device)
            regime = batch['regime'].to(self.device)
            expert_actions = batch['actions'].to(self.device)
            returns = batch['returns'].to(self.device)
            volatility = batch['volatility'].to(self.device)
            
            # Calculate losses
            losses = self.model.train_step(
                price, indicators, orderbook, regime,
                expert_actions, returns, volatility
            )
            
            total_loss += losses['total_loss'].item()
            total_diffusion_loss += losses['diffusion_loss'].item()
            total_return_loss += losses['return_loss'].item()
            total_volatility_loss += losses['volatility_loss'].item()
            
            # Generate strategies and evaluate
            with torch.no_grad():
                market_state = self.model.encode_market_state(
                    price, indicators, orderbook, regime
                )
                generated_actions = self.model.generate_strategy(
                    market_state, num_samples=1, use_ddim=True, ddim_steps=20
                )
                
                # Simulate returns (simplified)
                # In practice, you'd use a proper backtesting framework
                generated_returns = self._simulate_returns(generated_actions, returns)
                expert_returns = self._simulate_returns(expert_actions, returns)
                
                all_generated_returns.append(generated_returns)
                all_expert_returns.append(expert_returns)
        
        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches,
            'return_loss': total_return_loss / num_batches,
            'volatility_loss': total_volatility_loss / num_batches
        }
        
        # Calculate strategy quality metrics
        if all_generated_returns:
            generated_returns = torch.cat(all_generated_returns, dim=0)
            expert_returns = torch.cat(all_expert_returns, dim=0)
            
            metrics['generated_sharpe'] = self.risk_metrics.sharpe_ratio(generated_returns).mean().item()
            metrics['expert_sharpe'] = self.risk_metrics.sharpe_ratio(expert_returns).mean().item()
            metrics['generated_return'] = generated_returns.mean().item()
            metrics['expert_return'] = expert_returns.mean().item()
        
        return metrics
    
    def _simulate_returns(self, actions: torch.Tensor, market_returns: torch.Tensor) -> torch.Tensor:
        """
        Simplified return simulation.
        
        Args:
            actions: [batch, seq_length, action_dim]
            market_returns: [batch, seq_length]
        
        Returns:
            strategy_returns: [batch, seq_length]
        """
        # Use position (first action component) as weight
        positions = actions[:, :, 0]  # [batch, seq_length]
        
        # Strategy returns = position * market_returns
        strategy_returns = positions * market_returns
        
        return strategy_returns
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
            if 'generated_sharpe' in val_metrics:
                print(f"  Generated Sharpe: {val_metrics['generated_sharpe']:.4f}")
                print(f"  Expert Sharpe: {val_metrics['expert_sharpe']:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'val/total_loss': val_metrics['total_loss'],
                    'val/diffusion_loss': val_metrics['diffusion_loss'],
                    'val/return_loss': val_metrics['return_loss'],
                    'val/volatility_loss': val_metrics['volatility_loss'],
                    **{f'val/{k}': v for k, v in val_metrics.items() if k not in ['total_loss', 'diffusion_loss', 'return_loss', 'volatility_loss']}
                })
            
            # Save checkpoint
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint('best_model.pt')
                print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {path}")


class OnlineTrainer:
    """
    Online training for continuous learning from live trading.
    """
    
    def __init__(
        self,
        model: DiffusionStrategyGenerator,
        learning_rate: float = 1e-5,
        buffer_size: int = 10000,
        batch_size: int = 32,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.buffer = {
            'price': [],
            'indicators': [],
            'orderbook': [],
            'regime': [],
            'actions': [],
            'returns': [],
            'volatility': []
        }
        self.buffer_size = buffer_size
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.update_counter = 0
    
    def add_experience(
        self,
        price: np.ndarray,
        indicators: np.ndarray,
        orderbook: np.ndarray,
        regime: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        volatility: np.ndarray
    ):
        """Add experience to replay buffer."""
        self.buffer['price'].append(price)
        self.buffer['indicators'].append(indicators)
        self.buffer['orderbook'].append(orderbook)
        self.buffer['regime'].append(regime)
        self.buffer['actions'].append(actions)
        self.buffer['returns'].append(returns)
        self.buffer['volatility'].append(volatility)
        
        # Remove old experiences
        for key in self.buffer:
            if len(self.buffer[key]) > self.buffer_size:
                self.buffer[key].pop(0)
    
    def update(self, num_updates: int = 1) -> Dict[str, float]:
        """Perform online update."""
        if len(self.buffer['price']) < self.batch_size:
            return {}
        
        total_loss = 0.0
        
        for _ in range(num_updates):
            # Sample batch from buffer
            indices = np.random.choice(len(self.buffer['price']), self.batch_size, replace=False)
            
            batch = {
                key: torch.tensor(np.array([self.buffer[key][i] for i in indices]), 
                                dtype=torch.float32).to(self.device)
                for key in self.buffer
            }
            
            # Forward pass
            losses = self.model.train_step(
                batch['price'],
                batch['indicators'],
                batch['orderbook'],
                batch['regime'],
                batch['actions'],
                batch['returns'],
                batch['volatility']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            self.update_counter += 1
        
        return {'loss': total_loss / num_updates}

