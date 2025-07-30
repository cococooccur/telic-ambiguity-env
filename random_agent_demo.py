import numpy as np
from telic_ambiguity_env import TelicAmbiguityEnv, ACTION_NAMES

class RandomTelicAgent:
    """
    Minimal random agent for demonstration of TelicAmbiguityEnv
    """
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
    
    def select_action(self, observation):
        """Select random action from action space"""
        return self.action_space.sample()
    
    def run_episode(self, max_steps=None, verbose=True):
        """Run a single episode and return statistics"""
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        alignment_count = 0
        
        if verbose:
            print("="*80)
            print("TELIC AMBIGUITY ENVIRONMENT - RANDOM AGENT DEMO")
            print("="*80)
            print(f"{'Step':<6} {'φ(t)':<8} {'Loss':<8} {'Action':<18} {'Status':<15} {'Prompt'}")
            print("-"*80)
        
        done = False
        while not done:
            action = self.select_action(obs)
            obs, reward, done, info = self.env.step(action)
            
            total_reward += reward
            steps += 1
            
            if info['telic_aligned']:
                alignment_count += 1
            
            if verbose:
                action_name = ACTION_NAMES[action]
                status = "telic-aligned" if info['telic_aligned'] else "collapse"
                print(f"{steps:<6} {info['phi_t']:<8.3f} {info['loss']:<8.3f} {action_name:<18} {status:<15} '{info['prompt']}'")
            
            if max_steps and steps >= max_steps:
                break
        
        if verbose:
            print("-"*80)
            print(f"Episode Summary:")
            print(f"  Total Steps: {steps}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Alignment Rate: {alignment_count/steps:.2%}")
            print(f"  Final φ(t): {info['phi_t']:.3f}")
            print("="*80)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'alignment_rate': alignment_count / steps if steps > 0 else 0,
            'final_phi': info['phi_t']
        }


def demo_timestep_progression():
    """
    Demonstrate alignment status for timesteps t=0.00-2.00 in increments of 0.11
    """
    print("\n" + "="*60)
    print("TIMESTEP PROGRESSION DEMO (t=0.00-2.00, Δt=0.11)")
    print("="*60)
    
    env = TelicAmbiguityEnv(max_steps=200)
    agent = RandomTelicAgent(env)
    
    obs = env.reset()
    
    target_times = np.arange(0.00, 2.01, 0.11)
    current_time = 0.0
    time_step = 0.11
    
    print(f"{'Time':<8} {'φ(t)':<8} {'Loss':<8} {'Status':<15} {'Action':<18}")
    print("-"*65)
    
    for target_t in target_times:
        # Run environment to reach target time
        while current_time < target_t:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            current_time += time_step
            
            if done:
                obs = env.reset()
                current_time = 0.0
        
        # Display current state
        phi_t = info['phi_t'] if 'phi_t' in info else env.phi_t
        loss = info['loss'] if 'loss' in info else env._calculate_loss(0)
        status = "telic-aligned" if info.get('telic_aligned', False) else "collapse"
        action_name = ACTION_NAMES[info.get('action_taken', 0)]
        
        print(f"{target_t:<8.2f} {phi_t:<8.3f} {loss:<8.3f} {status:<15} {action_name:<18}")


def run_performance_analysis():
    """
    Run multiple episodes to analyze random agent performance
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS (10 Episodes)")
    print("="*60)
    
    env = TelicAmbiguityEnv()
    agent = RandomTelicAgent(env)
    
    results = []
    for episode in range(10):
        print(f"\nEpisode {episode + 1}:")
        result = agent.run_episode(max_steps=20, verbose=False)
        results.append(result)
        print(f"  Reward: {result['total_reward']:.2f}, Alignment: {result['alignment_rate']:.2%}, Final φ(t): {result['final_phi']:.3f}")
    
    # Summary statistics
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_alignment = np.mean([r['alignment_rate'] for r in results])
    avg_phi = np.mean([r['final_phi'] for r in results])
    
    print(f"\nSummary Statistics:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Alignment Rate: {avg_alignment:.2%}")
    print(f"  Average Final φ(t): {avg_phi:.3f}")


if __name__ == "__main__":
    # Create environment and agent
    env = TelicAmbiguityEnv()
    agent = RandomTelicAgent(env)
    
    # Run single detailed episode
    agent.run_episode(max_steps=25)
    
    # Run timestep progression demo
    demo_timestep_progression()
    
    # Run performance analysis
    run_performance_analysis()
    
    print("\nDemo completed. Environment is ready for RL training!")
