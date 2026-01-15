"""
Environment Simulation for Testing Intervention Agent

This module provides simulated environments for testing the intervention agent,
including scenarios with different risk profiles and dynamics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt

from diffusion_world_model import WorldState


@dataclass
class EnvironmentConfig:
    """Configuration for simulated environment"""
    state_dim: int
    action_dim: int
    dynamics_noise: float = 0.1
    risk_level: str = "medium"  # low, medium, high
    has_tail_events: bool = True
    tail_event_probability: float = 0.05


class SimulatedEnvironment:
    """
    Simulated environment with controllable dynamics and risk characteristics.
    
    Supports different risk profiles and tail event scenarios for testing
    the intervention agent's risk-aware decision making.
    """
    
    def __init__(self, config: EnvironmentConfig, seed: Optional[int] = None):
        self.config = config
        self.state = None
        self.time = 0
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize dynamics parameters
        self._init_dynamics()
        
        # History tracking
        self.state_history = []
        self.action_history = []
        self.cost_history = []
        
    def _init_dynamics(self):
        """Initialize environment dynamics"""
        # Linear dynamics matrix
        self.A = torch.randn(self.config.state_dim, self.config.state_dim) * 0.3
        self.A = self.A + torch.eye(self.config.state_dim) * 0.7  # Stable dynamics
        
        # Action influence matrix
        self.B = torch.randn(self.config.state_dim, self.config.action_dim) * 0.5
        
        # Risk parameters based on risk level
        if self.config.risk_level == "low":
            self.base_noise = 0.05
            self.tail_magnitude = 1.5
        elif self.config.risk_level == "medium":
            self.base_noise = 0.1
            self.tail_magnitude = 2.5
        else:  # high
            self.base_noise = 0.2
            self.tail_magnitude = 4.0
    
    def reset(self) -> WorldState:
        """Reset environment to initial state"""
        self.state = torch.randn(self.config.state_dim) * 0.5
        self.time = 0
        self.state_history = [self.state.clone()]
        self.action_history = []
        self.cost_history = []
        
        return WorldState(features=self.state.unsqueeze(0), timestamp=self.time)
    
    def step(self, action: torch.Tensor) -> Tuple[WorldState, float, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take [action_dim]
            
        Returns:
            next_state: Next world state
            cost: Step cost (lower is better)
            done: Whether episode is done
            info: Additional information
        """
        if action.dim() > 1:
            action = action.squeeze(0)
        
        # Deterministic dynamics
        next_state = self.A @ self.state + self.B @ action
        
        # Add noise
        noise = torch.randn_like(next_state) * self.base_noise
        
        # Tail event
        tail_event = False
        if self.config.has_tail_events:
            if np.random.rand() < self.config.tail_event_probability:
                tail_event = True
                tail_noise = torch.randn_like(next_state) * self.tail_magnitude
                noise = noise + tail_noise
        
        next_state = next_state + noise
        
        # Compute cost
        state_cost = torch.sum(next_state ** 2)
        action_cost = torch.sum(action ** 2) * 0.1
        cost = (state_cost + action_cost).item()
        
        # Update state
        self.state = next_state
        self.time += 1
        
        # Track history
        self.state_history.append(self.state.clone())
        self.action_history.append(action.clone())
        self.cost_history.append(cost)
        
        # Check if done
        done = self.time >= 100 or torch.norm(self.state) > 10.0
        
        info = {
            'tail_event': tail_event,
            'state_norm': torch.norm(self.state).item(),
            'cumulative_cost': sum(self.cost_history)
        }
        
        return WorldState(features=self.state.unsqueeze(0), timestamp=self.time), cost, done, info
    
    def get_current_state(self) -> WorldState:
        """Get current world state"""
        return WorldState(features=self.state.unsqueeze(0), timestamp=self.time)
    
    def generate_trajectory(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        include_noise: bool = True
    ) -> torch.Tensor:
        """
        Generate trajectory from initial state and action sequence.
        
        Args:
            initial_state: Initial state [state_dim]
            actions: Action sequence [horizon, action_dim]
            include_noise: Whether to include stochastic noise
            
        Returns:
            Trajectory: [horizon, state_dim]
        """
        horizon = actions.shape[0]
        trajectory = torch.zeros(horizon, self.config.state_dim)
        
        state = initial_state.clone()
        
        for t in range(horizon):
            # Dynamics
            state = self.A @ state + self.B @ actions[t]
            
            # Noise
            if include_noise:
                noise = torch.randn_like(state) * self.base_noise
                
                # Tail event
                if self.config.has_tail_events and np.random.rand() < self.config.tail_event_probability:
                    tail_noise = torch.randn_like(state) * self.tail_magnitude
                    noise = noise + tail_noise
                
                state = state + noise
            
            trajectory[t] = state
        
        return trajectory
    
    def visualize_trajectory(self, save_path: Optional[str] = None):
        """Visualize state and action history"""
        if len(self.state_history) == 0:
            print("No trajectory to visualize")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot states
        states = torch.stack(self.state_history).numpy()
        for i in range(min(5, self.config.state_dim)):
            axes[0].plot(states[:, i], label=f'State {i}')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('State Value')
        axes[0].set_title('State Trajectory')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot actions
        if len(self.action_history) > 0:
            actions = torch.stack(self.action_history).numpy()
            for i in range(min(3, self.config.action_dim)):
                axes[1].plot(actions[:, i], label=f'Action {i}')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Action Value')
            axes[1].set_title('Action Trajectory')
            axes[1].legend()
            axes[1].grid(True)
        
        # Plot costs
        axes[2].plot(self.cost_history, label='Step Cost')
        axes[2].plot(np.cumsum(self.cost_history), label='Cumulative Cost')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Cost')
        axes[2].set_title('Cost Over Time')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class CriticalInfrastructureEnvironment(SimulatedEnvironment):
    """
    Simulated critical infrastructure environment (e.g., power grid, water system).
    
    Features:
    - Multiple interconnected subsystems
    - Cascading failure risks
    - Safety constraints
    """
    
    def __init__(self, config: EnvironmentConfig, seed: Optional[int] = None):
        super().__init__(config, seed)
        
        # Define subsystems
        self.num_subsystems = 3
        self.subsystem_size = config.state_dim // self.num_subsystems
        
        # Safety thresholds
        self.safety_threshold = 3.0
        self.critical_threshold = 5.0
        
    def step(self, action: torch.Tensor) -> Tuple[WorldState, float, bool, Dict]:
        """Step with infrastructure-specific dynamics"""
        next_state, base_cost, done, info = super().step(action)
        
        # Check for subsystem failures
        subsystem_states = self.state.reshape(self.num_subsystems, self.subsystem_size)
        subsystem_norms = torch.norm(subsystem_states, dim=1)
        
        # Cascading failure
        failed_subsystems = (subsystem_norms > self.critical_threshold).sum().item()
        if failed_subsystems > 0:
            # Cascade to connected subsystems
            cascade_penalty = failed_subsystems * 10.0
            base_cost += cascade_penalty
            info['cascading_failure'] = True
            info['failed_subsystems'] = failed_subsystems
        
        # Safety violation penalty
        safety_violations = (subsystem_norms > self.safety_threshold).sum().item()
        if safety_violations > 0:
            base_cost += safety_violations * 2.0
            info['safety_violations'] = safety_violations
        
        return next_state, base_cost, done, info


class EpidemicEnvironment(SimulatedEnvironment):
    """
    Simulated epidemic environment for intervention testing.
    
    Features:
    - Compartmental disease dynamics (SIR-like)
    - Intervention actions (lockdown, vaccination, etc.)
    - Population-level outcomes
    """
    
    def __init__(self, config: EnvironmentConfig, seed: Optional[int] = None):
        # Override state_dim for epidemic model
        config.state_dim = 4  # Susceptible, Infected, Recovered, Deaths
        super().__init__(config, seed)
        
        # Epidemic parameters
        self.beta = 0.3  # Transmission rate
        self.gamma = 0.1  # Recovery rate
        self.mu = 0.01  # Death rate
        
    def reset(self) -> WorldState:
        """Reset with epidemic initial conditions"""
        # Start with mostly susceptible population
        self.state = torch.tensor([0.95, 0.05, 0.0, 0.0])  # S, I, R, D
        self.time = 0
        self.state_history = [self.state.clone()]
        self.action_history = []
        self.cost_history = []
        
        return WorldState(features=self.state.unsqueeze(0), timestamp=self.time)
    
    def step(self, action: torch.Tensor) -> Tuple[WorldState, float, bool, Dict]:
        """Step with epidemic dynamics"""
        if action.dim() > 1:
            action = action.squeeze(0)
        
        S, I, R, D = self.state
        
        # Actions: [social_distancing, vaccination, treatment]
        social_distancing = torch.sigmoid(action[0])  # [0, 1]
        vaccination = torch.sigmoid(action[1]) if self.config.action_dim > 1 else 0.0
        treatment = torch.sigmoid(action[2]) if self.config.action_dim > 2 else 0.0
        
        # Modified transmission rate
        effective_beta = self.beta * (1.0 - 0.7 * social_distancing)
        
        # Modified death rate
        effective_mu = self.mu * (1.0 - 0.5 * treatment)
        
        # SIR dynamics
        dS = -effective_beta * S * I - vaccination * S
        dI = effective_beta * S * I - self.gamma * I - effective_mu * I
        dR = self.gamma * I + vaccination * S
        dD = effective_mu * I
        
        # Update state
        dt = 0.1
        next_state = self.state + dt * torch.tensor([dS, dI, dR, dD])
        next_state = torch.clamp(next_state, 0.0, 1.0)
        
        # Normalize to ensure S + I + R + D = 1
        total = next_state.sum()
        if total > 0:
            next_state = next_state / total
        
        # Compute cost (deaths + economic cost of interventions)
        death_cost = next_state[3] * 100.0
        intervention_cost = (social_distancing * 5.0 + vaccination * 2.0 + treatment * 3.0)
        cost = (death_cost + intervention_cost).item()
        
        # Update
        self.state = next_state
        self.time += 1
        
        self.state_history.append(self.state.clone())
        self.action_history.append(action.clone())
        self.cost_history.append(cost)
        
        # Done if epidemic is over or time limit
        done = self.time >= 200 or I < 0.001
        
        info = {
            'infections': I.item(),
            'deaths': D.item(),
            'cumulative_cost': sum(self.cost_history)
        }
        
        return WorldState(features=self.state.unsqueeze(0), timestamp=self.time), cost, done, info


def create_environment(env_type: str, **kwargs) -> SimulatedEnvironment:
    """
    Factory function to create different environment types.
    
    Args:
        env_type: Type of environment ('basic', 'infrastructure', 'epidemic')
        **kwargs: Additional arguments for environment configuration
        
    Returns:
        Environment instance
    """
    if env_type == 'basic':
        config = EnvironmentConfig(**kwargs)
        return SimulatedEnvironment(config)
    
    elif env_type == 'infrastructure':
        config = EnvironmentConfig(**kwargs)
        return CriticalInfrastructureEnvironment(config)
    
    elif env_type == 'epidemic':
        config = EnvironmentConfig(**kwargs)
        return EpidemicEnvironment(config)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

