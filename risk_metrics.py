"""
Risk Metrics and Tail Risk Evaluation

This module provides various risk metrics for evaluating trajectory distributions,
with a focus on tail risk assessment for safety-critical decision making.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from diffusion_world_model import TrajectoryDistribution, WorldState


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment of trajectory distribution"""
    expected_cost: torch.Tensor  # [batch]
    cvar: torch.Tensor  # Conditional Value at Risk [batch]
    worst_case: torch.Tensor  # Worst case cost [batch]
    variance: torch.Tensor  # Cost variance [batch]
    tail_probability: torch.Tensor  # Probability of tail events [batch]
    risk_score: torch.Tensor  # Overall risk score [batch]


class RiskMetrics:
    """Compute various risk metrics for trajectory distributions"""
    
    def __init__(
        self,
        cost_function: Optional[Callable] = None,
        cvar_alpha: float = 0.1,
        tail_threshold: float = 0.9
    ):
        """
        Args:
            cost_function: Function to compute cost from trajectory
            cvar_alpha: Quantile for CVaR computation (e.g., 0.1 for worst 10%)
            tail_threshold: Threshold for tail event detection
        """
        self.cost_function = cost_function or self._default_cost_function
        self.cvar_alpha = cvar_alpha
        self.tail_threshold = tail_threshold
    
    def _default_cost_function(self, trajectory: torch.Tensor, world_state: WorldState) -> torch.Tensor:
        """Default cost function based on trajectory deviation"""
        # Simple L2 norm as default cost
        return trajectory.pow(2).sum(dim=-1).mean(dim=-1)
    
    def compute_cvar(self, costs: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute Conditional Value at Risk (CVaR).
        
        CVaR is the expected cost in the worst alpha quantile.
        
        Args:
            costs: [batch, num_samples]
            alpha: Quantile (e.g., 0.1 for worst 10%)
            
        Returns:
            CVaR: [batch]
        """
        batch_size, num_samples = costs.shape
        
        # Sort costs in descending order
        sorted_costs, _ = torch.sort(costs, dim=1, descending=True)
        
        # Number of samples in tail
        n_tail = max(1, int(alpha * num_samples))
        
        # Average of worst cases
        cvar = sorted_costs[:, :n_tail].mean(dim=1)
        
        return cvar
    
    def compute_var(self, costs: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute Value at Risk (VaR).
        
        VaR is the alpha-quantile of the cost distribution.
        
        Args:
            costs: [batch, num_samples]
            alpha: Quantile
            
        Returns:
            VaR: [batch]
        """
        quantile_idx = int(alpha * costs.shape[1])
        sorted_costs, _ = torch.sort(costs, dim=1, descending=True)
        return sorted_costs[:, quantile_idx]
    
    def compute_entropic_risk(
        self,
        costs: torch.Tensor,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute entropic risk measure.
        
        Entropic risk is a coherent risk measure that penalizes uncertainty.
        
        Args:
            costs: [batch, num_samples]
            beta: Risk sensitivity parameter (higher = more risk-averse)
            
        Returns:
            Entropic risk: [batch]
        """
        return (1.0 / beta) * torch.logsumexp(beta * costs, dim=1) - np.log(costs.shape[1]) / beta
    
    def assess_risk(
        self,
        trajectory_dist: TrajectoryDistribution,
        world_state: WorldState,
        custom_cost_fn: Optional[Callable] = None
    ) -> RiskAssessment:
        """
        Comprehensive risk assessment of trajectory distribution.
        
        Args:
            trajectory_dist: Distribution over trajectories
            world_state: Current world state
            custom_cost_fn: Optional custom cost function
            
        Returns:
            RiskAssessment with multiple risk metrics
        """
        batch_size, num_samples, horizon, state_dim = trajectory_dist.samples.shape
        
        # Compute costs for each trajectory sample
        cost_fn = custom_cost_fn or self.cost_function
        costs = torch.zeros(batch_size, num_samples, device=trajectory_dist.samples.device)
        
        for i in range(num_samples):
            costs[:, i] = cost_fn(trajectory_dist.samples[:, i], world_state)
        
        # Expected cost
        expected_cost = costs.mean(dim=1)
        
        # Conditional Value at Risk (CVaR)
        cvar = self.compute_cvar(costs, self.cvar_alpha)
        
        # Worst case cost
        worst_case = costs.max(dim=1)[0]
        
        # Variance
        variance = costs.var(dim=1)
        
        # Tail probability (probability of exceeding threshold)
        threshold = torch.quantile(costs, self.tail_threshold, dim=1, keepdim=True)
        tail_probability = (costs > threshold).float().mean(dim=1)
        
        # Overall risk score (weighted combination)
        risk_score = (
            0.3 * expected_cost +
            0.4 * cvar +
            0.2 * worst_case +
            0.1 * variance.sqrt()
        )
        
        return RiskAssessment(
            expected_cost=expected_cost,
            cvar=cvar,
            worst_case=worst_case,
            variance=variance,
            tail_probability=tail_probability,
            risk_score=risk_score
        )
    
    def compute_safety_margin(
        self,
        trajectory_dist: TrajectoryDistribution,
        safety_constraints: List[Callable],
        confidence_level: float = 0.95
    ) -> torch.Tensor:
        """
        Compute safety margin with respect to constraints.
        
        Args:
            trajectory_dist: Distribution over trajectories
            safety_constraints: List of constraint functions
            confidence_level: Required confidence level
            
        Returns:
            Safety margin: [batch]
        """
        batch_size, num_samples = trajectory_dist.samples.shape[:2]
        device = trajectory_dist.samples.device
        
        margins = []
        
        for constraint_fn in safety_constraints:
            # Evaluate constraint for all samples
            constraint_values = torch.zeros(batch_size, num_samples, device=device)
            
            for i in range(num_samples):
                constraint_values[:, i] = constraint_fn(trajectory_dist.samples[:, i])
            
            # Compute margin at confidence level
            quantile_idx = int((1.0 - confidence_level) * num_samples)
            sorted_values, _ = torch.sort(constraint_values, dim=1)
            margin = sorted_values[:, quantile_idx]
            margins.append(margin)
        
        # Return minimum margin across all constraints
        return torch.stack(margins, dim=1).min(dim=1)[0]


class TailRiskAnalyzer:
    """Specialized analyzer for tail risk events"""
    
    def __init__(self, extreme_quantile: float = 0.05):
        """
        Args:
            extreme_quantile: Quantile for extreme event analysis
        """
        self.extreme_quantile = extreme_quantile
    
    def identify_tail_scenarios(
        self,
        trajectory_dist: TrajectoryDistribution,
        cost_fn: Callable,
        world_state: WorldState,
        num_tail_scenarios: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Identify and analyze worst-case tail scenarios.
        
        Args:
            trajectory_dist: Distribution over trajectories
            cost_fn: Cost function
            world_state: Current world state
            num_tail_scenarios: Number of tail scenarios to return
            
        Returns:
            Dictionary with tail scenario analysis
        """
        batch_size, num_samples = trajectory_dist.samples.shape[:2]
        device = trajectory_dist.samples.device
        
        # Compute costs
        costs = torch.zeros(batch_size, num_samples, device=device)
        for i in range(num_samples):
            costs[:, i] = cost_fn(trajectory_dist.samples[:, i], world_state)
        
        # Find worst scenarios
        _, worst_indices = torch.topk(costs, num_tail_scenarios, dim=1)
        
        # Extract tail trajectories
        tail_trajectories = torch.gather(
            trajectory_dist.samples,
            1,
            worst_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, trajectory_dist.samples.shape[2], trajectory_dist.samples.shape[3]
            )
        )
        
        # Analyze tail characteristics
        tail_costs = torch.gather(costs, 1, worst_indices)
        
        # Compute tail divergence from mean
        mean_trajectory = trajectory_dist.samples.mean(dim=1, keepdim=True)
        tail_divergence = (tail_trajectories - mean_trajectory).pow(2).sum(dim=-1).sqrt().mean(dim=-1)
        
        return {
            'tail_trajectories': tail_trajectories,
            'tail_costs': tail_costs,
            'tail_divergence': tail_divergence,
            'tail_indices': worst_indices
        }
    
    def compute_tail_dependence(
        self,
        trajectory_dist: TrajectoryDistribution,
        risk_factors: List[Callable]
    ) -> torch.Tensor:
        """
        Compute tail dependence between multiple risk factors.
        
        Measures how often multiple risk factors are simultaneously extreme.
        
        Args:
            trajectory_dist: Distribution over trajectories
            risk_factors: List of risk factor functions
            
        Returns:
            Tail dependence coefficient: [batch, num_factors, num_factors]
        """
        batch_size, num_samples = trajectory_dist.samples.shape[:2]
        num_factors = len(risk_factors)
        device = trajectory_dist.samples.device
        
        # Evaluate all risk factors
        factor_values = torch.zeros(batch_size, num_samples, num_factors, device=device)
        
        for f_idx, factor_fn in enumerate(risk_factors):
            for s_idx in range(num_samples):
                factor_values[:, s_idx, f_idx] = factor_fn(trajectory_dist.samples[:, s_idx])
        
        # Compute tail dependence
        threshold_idx = int(self.extreme_quantile * num_samples)
        tail_dependence = torch.zeros(batch_size, num_factors, num_factors, device=device)
        
        for i in range(num_factors):
            for j in range(num_factors):
                # Find extreme events for each factor
                threshold_i = torch.quantile(factor_values[:, :, i], 1.0 - self.extreme_quantile, dim=1, keepdim=True)
                threshold_j = torch.quantile(factor_values[:, :, j], 1.0 - self.extreme_quantile, dim=1, keepdim=True)
                
                extreme_i = factor_values[:, :, i] > threshold_i
                extreme_j = factor_values[:, :, j] > threshold_j
                
                # Compute joint probability
                joint_extreme = (extreme_i & extreme_j).float().mean(dim=1)
                tail_dependence[:, i, j] = joint_extreme / self.extreme_quantile
        
        return tail_dependence
    
    def estimate_extreme_value_distribution(
        self,
        trajectory_dist: TrajectoryDistribution,
        cost_fn: Callable,
        world_state: WorldState
    ) -> Dict[str, torch.Tensor]:
        """
        Fit Generalized Extreme Value (GEV) distribution to tail.
        
        Returns parameters for modeling extreme events.
        
        Args:
            trajectory_dist: Distribution over trajectories
            cost_fn: Cost function
            world_state: Current world state
            
        Returns:
            Dictionary with GEV parameters
        """
        batch_size, num_samples = trajectory_dist.samples.shape[:2]
        device = trajectory_dist.samples.device
        
        # Compute costs
        costs = torch.zeros(batch_size, num_samples, device=device)
        for i in range(num_samples):
            costs[:, i] = cost_fn(trajectory_dist.samples[:, i], world_state)
        
        # Extract tail samples
        n_tail = max(1, int(self.extreme_quantile * num_samples))
        sorted_costs, _ = torch.sort(costs, dim=1, descending=True)
        tail_costs = sorted_costs[:, :n_tail]
        
        # Estimate GEV parameters (simplified method of moments)
        location = tail_costs.mean(dim=1)
        scale = tail_costs.std(dim=1)
        shape = torch.zeros_like(location)  # Gumbel distribution as default
        
        return {
            'location': location,
            'scale': scale,
            'shape': shape,
            'tail_samples': tail_costs
        }

