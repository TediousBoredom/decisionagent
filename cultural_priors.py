"""
Cultural Priors for Action Space Constraints

This module implements structured cultural priors as soft constraints over the action space,
enabling interpretable and culturally-aware interventions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CulturalDimension(Enum):
    """Cultural dimensions based on cross-cultural psychology"""
    INDIVIDUALISM_COLLECTIVISM = "individualism_collectivism"
    POWER_DISTANCE = "power_distance"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE_RESTRAINT = "indulgence_restraint"
    RISK_TOLERANCE = "risk_tolerance"


@dataclass
class CulturalProfile:
    """Represents a cultural profile with multiple dimensions"""
    dimensions: Dict[CulturalDimension, float]  # Values in [0, 1]
    context: Optional[Dict[str, torch.Tensor]] = None
    confidence: float = 1.0


@dataclass
class ActionConstraints:
    """Soft constraints on actions derived from cultural priors"""
    preferred_actions: torch.Tensor  # [action_dim]
    forbidden_regions: List[Tuple[torch.Tensor, torch.Tensor]]  # List of (lower, upper) bounds
    soft_penalties: torch.Tensor  # [action_dim]
    interpretability_scores: torch.Tensor  # [action_dim]


class CulturalPriorEncoder(nn.Module):
    """Encodes cultural profiles into latent representations"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        num_dimensions = len(CulturalDimension)
        
        self.encoder = nn.Sequential(
            nn.Linear(num_dimensions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, cultural_profile: CulturalProfile) -> torch.Tensor:
        """
        Encode cultural profile to latent representation.
        
        Args:
            cultural_profile: Cultural profile
            
        Returns:
            Latent encoding: [batch, hidden_dim]
        """
        # Convert cultural dimensions to tensor
        dim_values = []
        for dim in CulturalDimension:
            value = cultural_profile.dimensions.get(dim, 0.5)
            dim_values.append(value)
        
        x = torch.tensor(dim_values, dtype=torch.float32).unsqueeze(0)
        return self.encoder(x)


class CulturalConstraintGenerator(nn.Module):
    """Generates action constraints from cultural priors"""
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 128,
        num_constraint_types: int = 5
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.cultural_encoder = CulturalPriorEncoder(hidden_dim)
        
        # Networks for different constraint types
        self.preference_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Preferences in [-1, 1]
        )
        
        self.penalty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()  # Non-negative penalties
        )
        
        self.interpretability_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Interpretability scores in [0, 1]
        )
        
    def forward(
        self,
        cultural_profile: CulturalProfile,
        context: Optional[torch.Tensor] = None
    ) -> ActionConstraints:
        """
        Generate action constraints from cultural profile.
        
        Args:
            cultural_profile: Cultural profile
            context: Optional context tensor
            
        Returns:
            Action constraints
        """
        # Encode cultural profile
        cultural_encoding = self.cultural_encoder(cultural_profile)
        
        # Add context if provided
        if context is not None:
            cultural_encoding = cultural_encoding + context
        
        # Generate constraints
        preferred_actions = self.preference_net(cultural_encoding).squeeze(0)
        soft_penalties = self.penalty_net(cultural_encoding).squeeze(0)
        interpretability_scores = self.interpretability_net(cultural_encoding).squeeze(0)
        
        # Generate forbidden regions based on cultural dimensions
        forbidden_regions = self._generate_forbidden_regions(cultural_profile)
        
        return ActionConstraints(
            preferred_actions=preferred_actions,
            forbidden_regions=forbidden_regions,
            soft_penalties=soft_penalties,
            interpretability_scores=interpretability_scores
        )
    
    def _generate_forbidden_regions(
        self,
        cultural_profile: CulturalProfile
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate forbidden action regions based on cultural norms"""
        forbidden_regions = []
        
        # Example: High uncertainty avoidance -> avoid extreme actions
        uncertainty_avoidance = cultural_profile.dimensions.get(
            CulturalDimension.UNCERTAINTY_AVOIDANCE, 0.5
        )
        
        if uncertainty_avoidance > 0.7:
            # Forbid extreme actions
            lower = torch.full((self.action_dim,), 0.8)
            upper = torch.full((self.action_dim,), 1.0)
            forbidden_regions.append((lower, upper))
            
            lower = torch.full((self.action_dim,), -1.0)
            upper = torch.full((self.action_dim,), -0.8)
            forbidden_regions.append((lower, upper))
        
        # Example: Collectivism -> prefer coordinated actions
        individualism = cultural_profile.dimensions.get(
            CulturalDimension.INDIVIDUALISM_COLLECTIVISM, 0.5
        )
        
        if individualism < 0.3:  # Collectivist
            # Could add constraints for action coordination
            pass
        
        return forbidden_regions


class AdaptiveCulturalPrior:
    """Adaptive cultural prior that learns from interactions"""
    
    def __init__(
        self,
        action_dim: int,
        initial_profile: Optional[CulturalProfile] = None,
        learning_rate: float = 0.01
    ):
        self.action_dim = action_dim
        self.cultural_profile = initial_profile or self._default_profile()
        self.learning_rate = learning_rate
        
        # History of actions and outcomes
        self.action_history: List[torch.Tensor] = []
        self.outcome_history: List[float] = []
        
    def _default_profile(self) -> CulturalProfile:
        """Create default neutral cultural profile"""
        return CulturalProfile(
            dimensions={dim: 0.5 for dim in CulturalDimension},
            confidence=0.5
        )
    
    def update(
        self,
        action: torch.Tensor,
        outcome: float,
        feedback: Optional[Dict[str, float]] = None
    ):
        """
        Update cultural prior based on action outcomes.
        
        Args:
            action: Taken action
            outcome: Outcome value (higher is better)
            feedback: Optional explicit feedback on cultural dimensions
        """
        self.action_history.append(action.detach().clone())
        self.outcome_history.append(outcome)
        
        # Update cultural dimensions based on feedback
        if feedback is not None:
            for dim, value in feedback.items():
                if isinstance(dim, str):
                    dim = CulturalDimension(dim)
                
                current_value = self.cultural_profile.dimensions[dim]
                updated_value = current_value + self.learning_rate * (value - current_value)
                self.cultural_profile.dimensions[dim] = max(0.0, min(1.0, updated_value))
        
        # Increase confidence with more data
        self.cultural_profile.confidence = min(
            1.0,
            self.cultural_profile.confidence + 0.01
        )
    
    def get_profile(self) -> CulturalProfile:
        """Get current cultural profile"""
        return self.cultural_profile
    
    def compute_action_compatibility(
        self,
        action: torch.Tensor,
        constraints: ActionConstraints
    ) -> torch.Tensor:
        """
        Compute how compatible an action is with cultural constraints.
        
        Args:
            action: Proposed action [action_dim]
            constraints: Cultural constraints
            
        Returns:
            Compatibility score in [0, 1]
        """
        # Preference alignment
        preference_score = F.cosine_similarity(
            action.unsqueeze(0),
            constraints.preferred_actions.unsqueeze(0),
            dim=1
        )
        preference_score = (preference_score + 1.0) / 2.0  # Map to [0, 1]
        
        # Penalty score
        penalty_score = torch.exp(-torch.sum(constraints.soft_penalties * action.abs()))
        
        # Forbidden region check
        forbidden_penalty = 0.0
        for lower, upper in constraints.forbidden_regions:
            in_forbidden = ((action >= lower) & (action <= upper)).float()
            forbidden_penalty += in_forbidden.sum()
        
        forbidden_score = torch.exp(-forbidden_penalty)
        
        # Combined compatibility
        compatibility = (
            0.4 * preference_score +
            0.3 * penalty_score +
            0.3 * forbidden_score
        )
        
        return compatibility.squeeze()


class InterpretableActionExplainer:
    """Explains actions in terms of cultural dimensions"""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        
    def explain_action(
        self,
        action: torch.Tensor,
        cultural_profile: CulturalProfile,
        constraints: ActionConstraints
    ) -> Dict[str, any]:
        """
        Generate interpretable explanation for an action.
        
        Args:
            action: Action to explain
            cultural_profile: Cultural profile
            constraints: Cultural constraints
            
        Returns:
            Dictionary with explanation components
        """
        explanation = {
            'action': action.tolist(),
            'cultural_alignment': {},
            'constraint_satisfaction': {},
            'interpretability_score': constraints.interpretability_scores.mean().item()
        }
        
        # Explain alignment with each cultural dimension
        for dim, value in cultural_profile.dimensions.items():
            alignment = self._compute_dimension_alignment(action, dim, value)
            explanation['cultural_alignment'][dim.value] = {
                'dimension_value': value,
                'action_alignment': alignment,
                'interpretation': self._interpret_alignment(dim, value, alignment)
            }
        
        # Explain constraint satisfaction
        preference_alignment = F.cosine_similarity(
            action.unsqueeze(0),
            constraints.preferred_actions.unsqueeze(0)
        ).item()
        
        explanation['constraint_satisfaction'] = {
            'preference_alignment': preference_alignment,
            'penalty_score': torch.sum(constraints.soft_penalties * action.abs()).item(),
            'in_forbidden_region': self._check_forbidden(action, constraints.forbidden_regions)
        }
        
        return explanation
    
    def _compute_dimension_alignment(
        self,
        action: torch.Tensor,
        dimension: CulturalDimension,
        dimension_value: float
    ) -> float:
        """Compute how well action aligns with cultural dimension"""
        
        if dimension == CulturalDimension.UNCERTAINTY_AVOIDANCE:
            # High uncertainty avoidance -> prefer moderate actions
            extremeness = action.abs().mean().item()
            if dimension_value > 0.5:
                return 1.0 - extremeness
            else:
                return extremeness
                
        elif dimension == CulturalDimension.RISK_TOLERANCE:
            # High risk tolerance -> accept larger actions
            action_magnitude = action.norm().item()
            if dimension_value > 0.5:
                return min(1.0, action_magnitude)
            else:
                return max(0.0, 1.0 - action_magnitude)
                
        elif dimension == CulturalDimension.LONG_TERM_ORIENTATION:
            # This would require temporal context
            return 0.5
            
        else:
            return 0.5
    
    def _interpret_alignment(
        self,
        dimension: CulturalDimension,
        dimension_value: float,
        alignment: float
    ) -> str:
        """Generate human-readable interpretation"""
        
        if alignment > 0.7:
            strength = "strongly"
        elif alignment > 0.5:
            strength = "moderately"
        else:
            strength = "weakly"
        
        if dimension == CulturalDimension.UNCERTAINTY_AVOIDANCE:
            if dimension_value > 0.5:
                return f"Action {strength} aligns with high uncertainty avoidance (moderate, cautious approach)"
            else:
                return f"Action {strength} aligns with low uncertainty avoidance (bold, exploratory approach)"
                
        elif dimension == CulturalDimension.RISK_TOLERANCE:
            if dimension_value > 0.5:
                return f"Action {strength} aligns with high risk tolerance (aggressive intervention)"
            else:
                return f"Action {strength} aligns with low risk tolerance (conservative intervention)"
                
        else:
            return f"Action {strength} aligns with {dimension.value} = {dimension_value:.2f}"
    
    def _check_forbidden(
        self,
        action: torch.Tensor,
        forbidden_regions: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> bool:
        """Check if action is in forbidden region"""
        for lower, upper in forbidden_regions:
            if ((action >= lower) & (action <= upper)).any():
                return True
        return False

