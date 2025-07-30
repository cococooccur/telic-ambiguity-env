import gym
from gym import spaces
import numpy as np
import random
import math
from typing import Dict, Any, Tuple

class TelicAmbiguityEnv(gym.Env):
    """
    Telic Ambiguity Environment
    
    A reinforcement learning environment that demonstrates telic-aligned behavior
    under recursive ambiguity conditions. Uses deformation-routing for semantic
    closure rather than symbolic targets.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, max_steps: int = 100, complexity_factor: float = 0.7):
        super(TelicAmbiguityEnv, self).__init__()
        
        self.max_steps = max_steps
        self.complexity_factor = complexity_factor
        
        # Action space: 4 telic response types
        # 0: Recursive fold, 1: Semantic anchor, 2: Deformation route, 3: Silence
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [ambiguity_vector, phi_t, internal_state]
        # ambiguity_vector: 8-dim semantic encoding of current prompt
        # phi_t: current deformation metric
        # internal_state: 4-dim agent memory state
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )
        
        # Ambiguous prompts database
        self.ambiguous_prompts = [
            "Resolve now",
            "Approach...?", 
            "Alignment pending",
            "?",
            "",
            "Collapse imminent",
            "Recursive fold detected",
            "Semantic drift",
            "Telic anchor required",
            "Deformation threshold",
            "Memory trace",
            "Signal degradation",
            "Compression failure",
            "Echo state",
            "Void response"
        ]
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.phi_t = 0.5  # Initial deformation metric
        self.internal_state = np.random.uniform(-0.1, 0.1, 4)  # Agent memory
        self.entropy_history = []
        self.alignment_history = []
        
        # Generate initial ambiguous prompt
        self.current_prompt = random.choice(self.ambiguous_prompts)
        
        return self._get_observation()
    
    def _encode_prompt(self, prompt: str) -> np.ndarray:
        """Encode prompt into 8-dimensional semantic vector"""
        # Simple semantic encoding based on prompt characteristics
        vector = np.zeros(8)
        
        if not prompt or prompt == "?":
            vector[0] = 1.0  # Pure ambiguity
        
        if "resolve" in prompt.lower():
            vector[1] = 0.8  # Resolution demand
            
        if "..." in prompt or prompt.endswith("?"):
            vector[2] = 0.9  # Uncertainty marker
            
        if "align" in prompt.lower():
            vector[3] = 0.7  # Alignment reference
            
        if "collapse" in prompt.lower():
            vector[4] = 0.95  # Collapse signal
            
        if "recursive" in prompt.lower() or "fold" in prompt.lower():
            vector[5] = 0.85  # Recursive pattern
            
        if "semantic" in prompt.lower() or "memory" in prompt.lower():
            vector[6] = 0.6  # Semantic reference
            
        # Prompt length encoding
        vector[7] = min(len(prompt) / 20.0, 1.0)
        
        return vector
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""
        prompt_encoding = self._encode_prompt(self.current_prompt)
        phi_array = np.array([self.phi_t])
        
        observation = np.concatenate([
            prompt_encoding,  # 8 dims
            phi_array,        # 1 dim
            self.internal_state  # 4 dims
        ])
        
        return observation.astype(np.float32)
    
    def _update_phi_t(self, action: int) -> float:
        """Update deformation metric based on action and context"""
        # Base deformation evolution
        time_factor = self.current_step / self.max_steps
        base_drift = 0.1 * math.sin(time_factor * math.pi * 2)
        
        # Action-specific deformation modulation
        action_effects = {
            0: -0.15,  # Recursive fold: reduces deformation
            1: -0.10,  # Semantic anchor: stabilizes  
            2: -0.05,  # Deformation route: mild stabilization
            3: 0.05    # Silence: slight drift increase
        }
        
        # Ambiguity stress factor
        prompt_stress = 0.2 if self.current_prompt in ["?", "", "Collapse imminent"] else 0.1
        
        # Recursive pressure from entropy history
        if len(self.entropy_history) > 3:
            entropy_slope = np.gradient(self.entropy_history[-3:])[-1]
            recursive_pressure = 0.1 * entropy_slope
        else:
            recursive_pressure = 0.0
        
        self.phi_t += base_drift + action_effects[action] + prompt_stress + recursive_pressure
        self.phi_t = np.clip(self.phi_t, 0.0, 1.0)
        
        return self.phi_t
    
    def _calculate_loss(self, action: int) -> float:
        """Calculate loss function based on current state and action"""
        # Baseline loss from misalignment
        base_loss = 0.3
        
        # Action alignment assessment
        prompt_lower = self.current_prompt.lower()
        
        if "resolve" in prompt_lower and action != 0:  # Should use recursive fold
            base_loss += 0.4
        elif "?" in self.current_prompt and action == 3:  # Silence for ambiguity
            base_loss -= 0.2
        elif "collapse" in prompt_lower and action == 1:  # Anchor during collapse
            base_loss -= 0.3
        elif self.current_prompt == "" and action == 2:  # Route empty state
            base_loss -= 0.1
        
        # Internal state entropy factor
        internal_entropy = -np.sum(self.internal_state * np.log(np.abs(self.internal_state) + 1e-8))
        entropy_penalty = 0.1 * internal_entropy
        
        # Deformation stress penalty
        deformation_stress = 0.5 * (self.phi_t ** 2)
        
        total_loss = base_loss + entropy_penalty + deformation_stress
        return max(0.0, total_loss)
    
    def _update_internal_state(self, action: int):
        """Update agent's internal memory state"""
        # Apply action-specific memory transformations
        transformations = {
            0: np.array([0.1, -0.05, 0.02, -0.01]),  # Recursive fold
            1: np.array([-0.02, 0.1, -0.03, 0.05]),   # Semantic anchor
            2: np.array([0.03, -0.02, 0.08, -0.04]),  # Deformation route
            3: np.array([0.0, 0.0, 0.0, 0.01])        # Silence
        }
        
        # Memory decay
        self.internal_state *= 0.95
        
        # Apply transformation
        self.internal_state += transformations[action]
        
        # Clip to valid range
        self.internal_state = np.clip(self.internal_state, -1.0, 1.0)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        # Update deformation metric
        old_phi = self.phi_t
        new_phi = self._update_phi_t(action)
        
        # Calculate loss
        loss = self._calculate_loss(action)
        
        # Update internal state
        self._update_internal_state(action)
        
        # Track entropy history
        internal_entropy = -np.sum(self.internal_state * np.log(np.abs(self.internal_state) + 1e-8))
        self.entropy_history.append(internal_entropy)
        
        # Determine telic alignment
        telic_aligned = loss <= self.phi_t
        self.alignment_history.append(telic_aligned)
        
        # Calculate reward (inverse of misalignment)
        reward = 1.0 if telic_aligned else -0.5
        
        # Bonus for maintaining alignment
        if len(self.alignment_history) >= 3 and all(self.alignment_history[-3:]):
            reward += 0.5
        
        # Advance step
        self.current_step += 1
        
        # Generate next prompt
        self.current_prompt = random.choice(self.ambiguous_prompts)
        
        # Check termination
        done = self.current_step >= self.max_steps
        
        # Additional termination: collapse condition
        if self.phi_t > 0.9 and not telic_aligned:
            done = True
            reward -= 2.0  # Penalty for collapse
        
        info = {
            'phi_t': self.phi_t,
            'loss': loss,
            'telic_aligned': telic_aligned,
            'prompt': self.current_prompt,
            'internal_entropy': internal_entropy,
            'action_taken': action
        }
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """Render current environment state"""
        if hasattr(self, 'current_step'):
            alignment_status = "telic-aligned" if len(self.alignment_history) > 0 and self.alignment_history[-1] else "collapse"
            loss = self._calculate_loss(0) if hasattr(self, 'phi_t') else 0.0
            
            print(f"Step {self.current_step:03d}: φ(t)={self.phi_t:.3f}, loss={loss:.3f}, status={alignment_status}, prompt='{self.current_prompt}'")
    
    def close(self):
        """Clean up environment"""
        pass


# Action space mapping for reference
ACTION_NAMES = {
    0: "Recursive fold",
    1: "Semantic anchor", 
    2: "Deformation route",
    3: "Silence"
}


if __name__ == "__main__":
    # Quick test of the environment
    env = TelicAmbiguityEnv()
    obs = env.reset()
    
    print("TelicAmbiguityEnv initialized successfully")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break