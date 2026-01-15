"""
Environment Intervention Agent

This module implements the main agent that combines the diffusion world model,
risk assessment, and cultural priors for risk-aware decision making.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from diffusion_world_model import DiffusionWorldModel, WorldState, TrajectoryDistribution
from risk_metrics import RiskMetrics, RiskAssessment, TailRiskAnalyzer
from cultural_priors import (
    CulturalProfile, CulturalConstraintGenerator, AdaptiveCulturalPrior,
    InterpretableActionExplainer, ActionConstraints
)


@dataclass
class InterventionDecision:
    """Represents an intervention decision with full context"""
    action: torch.Tensor  # [action_dim]
    expected_risk: float
    risk_assessment: RiskAssessment
    cultural_compatibility: float
    trajectory_samples: torch.Tensor  # [num_samples, horizon, state_dim]
    explanation: Dict[str, any]
    confidence: float


@dataclass
class AgentConfig:
    """Configuration for the intervention agent"""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 6
    horizon: int = 16
    num_diffusion_steps: int = 100
    num_trajectory_samples: int = 32
    risk_aversion: float = 0.5  # 0 = risk-neutral, 1 = highly risk-averse
    cultural_weight: float = 0.3  # Weight of cultural constraints
    cvar_alpha: float = 0.1  # CVaR quantile
    learning_rate: float = 1e-4


class EnvironmentInterventionAgent(nn.Module):
    """
    Environment Intervention Agent with diffusion-based world model.
    
    This agent:
    1. Predicts distributions over future trajectories using diffusion models
    2. Assesses tail risks and uncertainty
    3. Incorporates cultural priors as soft constraints
    4. Makes interpretable, risk-aware intervention decisions
    """
    
    def __init__(
        self,
        config: AgentConfig,
        cultural_profile: Optional[CulturalProfile] = None
    ):
        super().__init__()
        
        self.config = config
        
        # Core components
        self.world_model = DiffusionWorldModel(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            horizon=config.horizon,
            num_diffusion_steps=config.num_diffusion_steps
        )
        
        self.risk_metrics = RiskMetrics(cvar_alpha=config.cvar_alpha)
        self.tail_risk_analyzer = TailRiskAnalyzer()
        
        # Cultural components
        self.constraint_generator = CulturalConstraintGenerator(
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.adaptive_prior = AdaptiveCulturalPrior(
            action_dim=config.action_dim,
            initial_profile=cultural_profile
        )
        
        self.explainer = InterpretableActionExplainer(config.action_dim)
        
        # Action policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.state_dim + config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Tanh()
        )
        
        # Value network for risk estimation
        self.value_net = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(
        self,
        world_state: WorldState,
        cultural_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: predict action given world state.
        
        Args:
            world_state: Current world state
            cultural_context: Optional cultural context tensor
            
        Returns:
            Action: [batch, action_dim]
        """
        # Get cultural constraints
        cultural_profile = self.adaptive_prior.get_profile()
        constraints = self.constraint_generator(cultural_profile, cultural_context)
        
        # Encode cultural constraints
        cultural_encoding = self.constraint_generator.cultural_encoder(cultural_profile)
        
        # Combine state and cultural encoding
        policy_input = torch.cat([
            world_state.features,
            cultural_encoding.expand(world_state.features.shape[0], -1)
        ], dim=-1)
        
        # Generate base action
        base_action = self.policy_net(policy_input)
        
        # Apply cultural constraints as soft modulation
        action = self._apply_cultural_constraints(base_action, constraints)
        
        return action
    
    def _apply_cultural_constraints(
        self,
        action: torch.Tensor,
        constraints: ActionConstraints
    ) -> torch.Tensor:
        """Apply cultural constraints to action"""
        # Blend with preferred actions
        action = (
            (1 - self.config.cultural_weight) * action +
            self.config.cultural_weight * constraints.preferred_actions.unsqueeze(0)
        )
        
        # Apply soft penalties
        penalty_mask = torch.exp(-constraints.soft_penalties.unsqueeze(0))
        action = action * penalty_mask
        
        # Project out of forbidden regions
        for lower, upper in constraints.forbidden_regions:
            in_forbidden = ((action >= lower) & (action <= upper))
            if in_forbidden.any():
                # Project to nearest boundary
                dist_to_lower = (action - lower).abs()
                dist_to_upper = (action - upper).abs()
                action = torch.where(
                    dist_to_lower < dist_to_upper,
                    lower - 0.1,
                    upper + 0.1
                )
        
        return action
    
    def decide_intervention(
        self,
        world_state: WorldState,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        num_candidates: int = 10,
        cost_function: Optional[Callable] = None
    ) -> InterventionDecision:
        """
        Make a risk-aware intervention decision.
        
        This is the main decision-making method that:
        1. Generates or evaluates candidate actions
        2. Predicts trajectory distributions for each action
        3. Assesses risks including tail risks
        4. Selects action balancing expected outcome and risk
        5. Provides interpretable explanation
        
        Args:
            world_state: Current world state
            candidate_actions: Optional list of candidate actions to evaluate
            num_candidates: Number of candidate actions to generate if not provided
            cost_function: Optional custom cost function
            
        Returns:
            InterventionDecision with selected action and full context
        """
        device = world_state.features.device
        
        # Generate candidate actions if not provided
        if candidate_actions is None:
            candidate_actions = self._generate_candidate_actions(
                world_state, num_candidates
            )
        
        # Evaluate each candidate action
        best_action = None
        best_score = float('-inf')
        best_risk_assessment = None
        best_trajectories = None
        best_compatibility = 0.0
        
        cultural_profile = self.adaptive_prior.get_profile()
        constraints = self.constraint_generator(cultural_profile)
        
        for action in candidate_actions:
            # Predict trajectory distribution
            trajectory_dist = self.world_model.sample_trajectories(
                world_state,
                action.unsqueeze(0) if action.dim() == 1 else action,
                num_samples=self.config.num_trajectory_samples
            )
            
            # Assess risk
            risk_assessment = self.risk_metrics.assess_risk(
                trajectory_dist,
                world_state,
                cost_function
            )
            
            # Compute cultural compatibility
            compatibility = self.adaptive_prior.compute_action_compatibility(
                action.squeeze() if action.dim() > 1 else action,
                constraints
            )
            
            # Compute overall score (lower risk is better)
            risk_score = (
                self.config.risk_aversion * risk_assessment.cvar[0] +
                (1 - self.config.risk_aversion) * risk_assessment.expected_cost[0]
            )
            
            # Incorporate cultural compatibility (higher is better)
            overall_score = -risk_score + 0.5 * compatibility
            
            if overall_score > best_score:
                best_score = overall_score
                best_action = action
                best_risk_assessment = risk_assessment
                best_trajectories = trajectory_dist.samples[0]
                best_compatibility = compatibility.item()
        
        # Generate explanation
        explanation = self.explainer.explain_action(
            best_action.squeeze() if best_action.dim() > 1 else best_action,
            cultural_profile,
            constraints
        )
        
        # Add risk information to explanation
        explanation['risk_metrics'] = {
            'expected_cost': best_risk_assessment.expected_cost[0].item(),
            'cvar': best_risk_assessment.cvar[0].item(),
            'worst_case': best_risk_assessment.worst_case[0].item(),
            'variance': best_risk_assessment.variance[0].item(),
            'tail_probability': best_risk_assessment.tail_probability[0].item()
        }
        
        # Compute confidence based on uncertainty and cultural alignment
        confidence = (
            0.5 * (1.0 - best_risk_assessment.variance[0].sqrt().item()) +
            0.3 * best_compatibility +
            0.2 * cultural_profile.confidence
        )
        
        return InterventionDecision(
            action=best_action.squeeze() if best_action.dim() > 1 else best_action,
            expected_risk=best_risk_assessment.expected_cost[0].item(),
            risk_assessment=best_risk_assessment,
            cultural_compatibility=best_compatibility,
            trajectory_samples=best_trajectories,
            explanation=explanation,
            confidence=confidence
        )
    
    def _generate_candidate_actions(
        self,
        world_state: WorldState,
        num_candidates: int
    ) -> List[torch.Tensor]:
        """Generate diverse candidate actions"""
        device = world_state.features.device
        candidates = []
        
        # Policy-based action
        policy_action = self.forward(world_state)
        candidates.append(policy_action.squeeze(0))
        
        # Add noise-perturbed versions
        for _ in range(num_candidates - 1):
            noise_scale = 0.1 + 0.3 * torch.rand(1).item()
            noisy_action = policy_action + noise_scale * torch.randn_like(policy_action)
            noisy_action = torch.clamp(noisy_action, -1.0, 1.0)
            candidates.append(noisy_action.squeeze(0))
        
        return candidates
    
    def analyze_tail_risks(
        self,
        world_state: WorldState,
        action: torch.Tensor,
        cost_function: Callable
    ) -> Dict[str, any]:
        """
        Perform detailed tail risk analysis for a given action.
        
        Args:
            world_state: Current world state
            action: Action to analyze
            cost_function: Cost function
            
        Returns:
            Dictionary with tail risk analysis
        """
        # Sample trajectories
        trajectory_dist = self.world_model.sample_trajectories(
            world_state,
            action.unsqueeze(0) if action.dim() == 1 else action,
            num_samples=self.config.num_trajectory_samples
        )
        
        # Identify tail scenarios
        tail_analysis = self.tail_risk_analyzer.identify_tail_scenarios(
            trajectory_dist,
            cost_function,
            world_state,
            num_tail_scenarios=5
        )
        
        # Estimate extreme value distribution
        gev_params = self.tail_risk_analyzer.estimate_extreme_value_distribution(
            trajectory_dist,
            cost_function,
            world_state
        )
        
        return {
            'tail_scenarios': tail_analysis,
            'extreme_value_params': gev_params,
            'trajectory_distribution': trajectory_dist
        }
    
    def update_from_outcome(
        self,
        action: torch.Tensor,
        outcome: float,
        cultural_feedback: Optional[Dict[str, float]] = None
    ):
        """
        Update agent based on intervention outcome.
        
        Args:
            action: Action that was taken
            outcome: Observed outcome (higher is better)
            cultural_feedback: Optional feedback on cultural dimensions
        """
        self.adaptive_prior.update(action, outcome, cultural_feedback)
    
    def train_world_model(
        self,
        trajectories: torch.Tensor,
        world_states: List[WorldState],
        actions: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the diffusion world model on historical data.
        
        Args:
            trajectories: Historical trajectories [num_samples, horizon, state_dim]
            world_states: List of world states
            actions: Actions taken [num_samples, action_dim]
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self.config.learning_rate
        )
        
        history = {'loss': []}
        num_samples = trajectories.shape[0]
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_trajectories = trajectories[batch_indices]
                batch_actions = actions[batch_indices]
                
                # Create batch world state
                batch_world_state = WorldState(
                    features=torch.stack([world_states[idx].features for idx in batch_indices])
                )
                
                # Compute loss
                losses = self.world_model.compute_loss(
                    batch_trajectories,
                    batch_world_state,
                    batch_actions
                )
                
                # Optimize
                optimizer.zero_grad()
                losses['total_loss'].backward()
                optimizer.step()
                
                epoch_losses.append(losses['total_loss'].item())
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return history
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'constraint_generator': self.constraint_generator.state_dict(),
            'config': self.config,
            'cultural_profile': self.adaptive_prior.cultural_profile
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.constraint_generator.load_state_dict(checkpoint['constraint_generator'])
        self.adaptive_prior.cultural_profile = checkpoint['cultural_profile']

