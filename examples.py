"""
Example Usage and Demonstration

This script demonstrates how to use the Environment Intervention Agent
for risk-aware decision making with cultural priors.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from intervention_agent import EnvironmentInterventionAgent, AgentConfig, InterventionDecision
from diffusion_world_model import WorldState
from cultural_priors import CulturalProfile, CulturalDimension
from environment import create_environment, EnvironmentConfig
from risk_metrics import RiskMetrics


def example_basic_intervention():
    """Basic example of using the intervention agent"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Intervention Agent")
    print("=" * 80)
    
    # Create agent configuration
    config = AgentConfig(
        state_dim=8,
        action_dim=4,
        hidden_dim=128,
        num_layers=4,
        horizon=16,
        num_diffusion_steps=50,
        num_trajectory_samples=16,
        risk_aversion=0.7,
        cultural_weight=0.3
    )
    
    # Create cultural profile (risk-averse, uncertainty-avoiding)
    cultural_profile = CulturalProfile(
        dimensions={
            CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.8,
            CulturalDimension.RISK_TOLERANCE: 0.3,
            CulturalDimension.LONG_TERM_ORIENTATION: 0.7,
            CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.4,
            CulturalDimension.POWER_DISTANCE: 0.5,
            CulturalDimension.INDULGENCE_RESTRAINT: 0.4
        },
        confidence=0.8
    )
    
    # Create agent
    agent = EnvironmentInterventionAgent(config, cultural_profile)
    
    # Create a world state
    world_state = WorldState(
        features=torch.randn(1, config.state_dim) * 0.5,
        timestamp=0.0
    )
    
    print(f"\nInitial world state: {world_state.features[0, :4].tolist()}")
    
    # Make intervention decision
    print("\nMaking intervention decision...")
    decision = agent.decide_intervention(world_state, num_candidates=5)
    
    print(f"\nSelected action: {decision.action.tolist()}")
    print(f"Expected risk: {decision.expected_risk:.4f}")
    print(f"Cultural compatibility: {decision.cultural_compatibility:.4f}")
    print(f"Decision confidence: {decision.confidence:.4f}")
    
    print("\nRisk Assessment:")
    print(f"  Expected cost: {decision.risk_assessment.expected_cost[0]:.4f}")
    print(f"  CVaR (worst 10%): {decision.risk_assessment.cvar[0]:.4f}")
    print(f"  Worst case: {decision.risk_assessment.worst_case[0]:.4f}")
    print(f"  Variance: {decision.risk_assessment.variance[0]:.4f}")
    print(f"  Tail probability: {decision.risk_assessment.tail_probability[0]:.4f}")
    
    print("\nCultural Alignment:")
    for dim, info in decision.explanation['cultural_alignment'].items():
        print(f"  {dim}:")
        print(f"    Dimension value: {info['dimension_value']:.2f}")
        print(f"    Action alignment: {info['action_alignment']:.2f}")
        print(f"    Interpretation: {info['interpretation']}")
    
    return agent, decision


def example_training_world_model():
    """Example of training the world model on synthetic data"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Training World Model")
    print("=" * 80)
    
    # Create environment
    env_config = EnvironmentConfig(
        state_dim=8,
        action_dim=4,
        dynamics_noise=0.1,
        risk_level="medium",
        has_tail_events=True
    )
    env = create_environment('basic', **env_config.__dict__)
    
    # Generate training data
    print("\nGenerating training data...")
    num_episodes = 50
    horizon = 16
    
    trajectories = []
    world_states = []
    actions = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_trajectory = []
        
        for t in range(horizon):
            # Random action
            action = torch.randn(env_config.action_dim) * 0.5
            
            # Generate trajectory segment
            traj = env.generate_trajectory(
                env.state,
                action.unsqueeze(0).expand(horizon, -1),
                include_noise=True
            )
            
            trajectories.append(traj)
            world_states.append(WorldState(features=env.state.unsqueeze(0)))
            actions.append(action)
            
            # Step environment
            state, _, done, _ = env.step(action)
            if done:
                break
    
    trajectories = torch.stack(trajectories)
    actions = torch.stack(actions)
    
    print(f"Generated {len(trajectories)} trajectory samples")
    
    # Create and train agent
    config = AgentConfig(
        state_dim=env_config.state_dim,
        action_dim=env_config.action_dim,
        hidden_dim=128,
        num_layers=4,
        horizon=horizon,
        num_diffusion_steps=50
    )
    
    agent = EnvironmentInterventionAgent(config)
    
    print("\nTraining world model...")
    history = agent.train_world_model(
        trajectories,
        world_states,
        actions,
        num_epochs=20,
        batch_size=16
    )
    
    print(f"\nFinal training loss: {history['loss'][-1]:.4f}")
    
    return agent, env, history


def example_risk_comparison():
    """Compare different risk profiles"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Risk Profile Comparison")
    print("=" * 80)
    
    config = AgentConfig(
        state_dim=6,
        action_dim=3,
        hidden_dim=128,
        horizon=12,
        num_trajectory_samples=32
    )
    
    # Create agents with different risk profiles
    risk_profiles = {
        'Risk-Neutral': 0.0,
        'Moderately Risk-Averse': 0.5,
        'Highly Risk-Averse': 0.9
    }
    
    world_state = WorldState(
        features=torch.randn(1, config.state_dim) * 0.8,
        timestamp=0.0
    )
    
    print(f"\nWorld state: {world_state.features[0, :3].tolist()}")
    print("\nComparing intervention decisions across risk profiles:\n")
    
    results = {}
    
    for profile_name, risk_aversion in risk_profiles.items():
        config.risk_aversion = risk_aversion
        agent = EnvironmentInterventionAgent(config)
        
        decision = agent.decide_intervention(world_state, num_candidates=8)
        
        results[profile_name] = decision
        
        print(f"{profile_name} (risk_aversion={risk_aversion}):")
        print(f"  Action magnitude: {decision.action.norm().item():.4f}")
        print(f"  Expected risk: {decision.expected_risk:.4f}")
        print(f"  CVaR: {decision.risk_assessment.cvar[0]:.4f}")
        print(f"  Confidence: {decision.confidence:.4f}")
        print()
    
    return results


def example_cultural_adaptation():
    """Example of cultural adaptation over time"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Cultural Adaptation")
    print("=" * 80)
    
    config = AgentConfig(
        state_dim=6,
        action_dim=3,
        hidden_dim=128,
        cultural_weight=0.4
    )
    
    # Start with neutral cultural profile
    agent = EnvironmentInterventionAgent(config)
    
    print("\nSimulating interventions with feedback...\n")
    
    # Simulate multiple interventions with outcomes
    num_interventions = 10
    
    for i in range(num_interventions):
        world_state = WorldState(
            features=torch.randn(1, config.state_dim) * 0.5,
            timestamp=float(i)
        )
        
        decision = agent.decide_intervention(world_state, num_candidates=5)
        
        # Simulate outcome (better outcomes for moderate actions)
        action_magnitude = decision.action.norm().item()
        outcome = 1.0 - 0.5 * action_magnitude + 0.1 * np.random.randn()
        
        # Provide cultural feedback
        cultural_feedback = {
            'uncertainty_avoidance': 0.7 if action_magnitude < 0.5 else 0.3,
            'risk_tolerance': 0.3 if action_magnitude < 0.5 else 0.7
        }
        
        # Update agent
        agent.update_from_outcome(decision.action, outcome, cultural_feedback)
        
        print(f"Intervention {i+1}:")
        print(f"  Action magnitude: {action_magnitude:.4f}")
        print(f"  Outcome: {outcome:.4f}")
        print(f"  Cultural confidence: {agent.adaptive_prior.cultural_profile.confidence:.4f}")
    
    print("\nFinal cultural profile:")
    for dim, value in agent.adaptive_prior.cultural_profile.dimensions.items():
        print(f"  {dim.value}: {value:.3f}")
    
    return agent


def example_tail_risk_analysis():
    """Detailed tail risk analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Tail Risk Analysis")
    print("=" * 80)
    
    config = AgentConfig(
        state_dim=8,
        action_dim=4,
        hidden_dim=128,
        num_trajectory_samples=64,  # More samples for better tail estimation
        risk_aversion=0.8
    )
    
    agent = EnvironmentInterventionAgent(config)
    
    world_state = WorldState(
        features=torch.randn(1, config.state_dim) * 0.7,
        timestamp=0.0
    )
    
    # Define a cost function
    def cost_function(trajectory: torch.Tensor, world_state: WorldState) -> torch.Tensor:
        # Cost based on state deviation and volatility
        state_cost = trajectory.pow(2).sum(dim=-1).mean(dim=-1)
        volatility = trajectory.diff(dim=0).pow(2).sum(dim=-1).mean(dim=0)
        return state_cost + 0.5 * volatility
    
    # Get intervention decision
    decision = agent.decide_intervention(
        world_state,
        num_candidates=10,
        cost_function=cost_function
    )
    
    print(f"\nSelected action: {decision.action.tolist()}")
    
    # Perform detailed tail risk analysis
    print("\nPerforming tail risk analysis...")
    tail_analysis = agent.analyze_tail_risks(
        world_state,
        decision.action,
        cost_function
    )
    
    print("\nTail Risk Metrics:")
    tail_scenarios = tail_analysis['tail_scenarios']
    print(f"  Number of tail scenarios identified: {tail_scenarios['tail_trajectories'].shape[0]}")
    print(f"  Tail costs: {tail_scenarios['tail_costs'][0].tolist()}")
    print(f"  Average tail divergence: {tail_scenarios['tail_divergence'][0].mean().item():.4f}")
    
    gev_params = tail_analysis['extreme_value_params']
    print("\nExtreme Value Distribution Parameters:")
    print(f"  Location: {gev_params['location'][0].item():.4f}")
    print(f"  Scale: {gev_params['scale'][0].item():.4f}")
    print(f"  Shape: {gev_params['shape'][0].item():.4f}")
    
    return agent, tail_analysis


def example_epidemic_intervention():
    """Example with epidemic environment"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Epidemic Intervention")
    print("=" * 80)
    
    # Create epidemic environment
    env_config = EnvironmentConfig(
        state_dim=4,  # Will be overridden by epidemic env
        action_dim=3,  # social_distancing, vaccination, treatment
        risk_level="high",
        has_tail_events=True
    )
    
    env = create_environment('epidemic', **env_config.__dict__)
    
    # Create agent
    config = AgentConfig(
        state_dim=4,
        action_dim=3,
        hidden_dim=128,
        horizon=20,
        risk_aversion=0.8,  # High risk aversion for epidemic
        cultural_weight=0.4
    )
    
    # Cultural profile emphasizing caution and collective action
    cultural_profile = CulturalProfile(
        dimensions={
            CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.9,
            CulturalDimension.RISK_TOLERANCE: 0.2,
            CulturalDimension.INDIVIDUALISM_COLLECTIVISM: 0.3,  # Collectivist
            CulturalDimension.LONG_TERM_ORIENTATION: 0.8,
            CulturalDimension.POWER_DISTANCE: 0.5,
            CulturalDimension.INDULGENCE_RESTRAINT: 0.3
        },
        confidence=0.9
    )
    
    agent = EnvironmentInterventionAgent(config, cultural_profile)
    
    # Run simulation
    print("\nRunning epidemic simulation with intervention agent...\n")
    
    state = env.reset()
    total_cost = 0
    max_steps = 50
    
    for step in range(max_steps):
        # Agent decides intervention
        decision = agent.decide_intervention(state, num_candidates=8)
        
        # Execute action in environment
        next_state, cost, done, info = env.step(decision.action)
        total_cost += cost
        
        if step % 10 == 0:
            S, I, R, D = state.features[0].tolist()
            print(f"Step {step}:")
            print(f"  Population: S={S:.3f}, I={I:.3f}, R={R:.3f}, D={D:.3f}")
            print(f"  Action: {decision.action.tolist()}")
            print(f"  Step cost: {cost:.2f}")
            print(f"  Cumulative cost: {total_cost:.2f}")
            print()
        
        state = next_state
        
        if done:
            print(f"Epidemic ended at step {step}")
            break
    
    print(f"\nFinal statistics:")
    print(f"  Total deaths: {state.features[0, 3].item():.4f}")
    print(f"  Total cost: {total_cost:.2f}")
    print(f"  Final infections: {state.features[0, 1].item():.4f}")
    
    return agent, env


def visualize_trajectory_distribution(decision: InterventionDecision, save_path: str = None):
    """Visualize the distribution of predicted trajectories"""
    trajectories = decision.trajectory_samples.cpu().numpy()
    num_samples, horizon, state_dim = trajectories.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot first 4 state dimensions
    for dim in range(min(4, state_dim)):
        ax = axes[dim]
        
        # Plot all trajectory samples
        for i in range(num_samples):
            ax.plot(trajectories[i, :, dim], alpha=0.1, color='blue')
        
        # Plot mean and confidence intervals
        mean_traj = trajectories[:, :, dim].mean(axis=0)
        std_traj = trajectories[:, :, dim].std(axis=0)
        
        ax.plot(mean_traj, color='red', linewidth=2, label='Mean')
        ax.fill_between(
            range(horizon),
            mean_traj - 2*std_traj,
            mean_traj + 2*std_traj,
            alpha=0.3,
            color='red',
            label='95% CI'
        )
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'State Dimension {dim}')
        ax.set_title(f'Trajectory Distribution - Dimension {dim}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTrajectory visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("ENVIRONMENT INTERVENTION AGENT - EXAMPLES")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run examples
    agent1, decision1 = example_basic_intervention()
    
    agent2, env2, history2 = example_training_world_model()
    
    results3 = example_risk_comparison()
    
    agent4 = example_cultural_adaptation()
    
    agent5, tail_analysis5 = example_tail_risk_analysis()
    
    agent6, env6 = example_epidemic_intervention()
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    
    # Visualize trajectory distribution from first example
    print("\nGenerating trajectory visualization...")
    visualize_trajectory_distribution(
        decision1,
        save_path='/inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/diffworld/trajectory_distribution.png'
    )


if __name__ == "__main__":
    main()

