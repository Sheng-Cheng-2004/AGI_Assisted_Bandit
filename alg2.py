import numpy as np
import math

class AGIAssistedBandit:
    def __init__(self, K, T, c, scenario_A_means, scenario_B_means, true_scenario):
        """
        Initialize the AGI-Assisted Bandit algorithm.
        
        Parameters:
        - K: Number of arms
        - T: Total time horizon
        - c: Cost of each AGI query
        - scenario_A_means: True means for each arm in scenario A
        - scenario_B_means: True means for each arm in scenario B
        - true_scenario: The actual scenario ('A' or 'B')
        """
        self.K = K
        self.T = T
        self.c = c
        self.scenario_A_means = scenario_A_means
        self.scenario_B_means = scenario_B_means
        self.true_scenario = true_scenario
        
        # Tracking variables
        self.empirical_means = np.zeros(K)
        self.empirical_means_A = np.zeros(K)
        self.empirical_means_B = np.zeros(K)
        self.pull_counts = np.zeros(K)
        self.pull_counts_A = np.zeros(K)
        self.pull_counts_B = np.zeros(K)
        self.total_AGI_queries = 0
        self.total_regret = 0
        self.optimal_mean = max(scenario_A_means if true_scenario == 'A' else scenario_B_means)
        
        # Algorithm state
        self.phase = 1  # Start with phase 1 (scenario identification)
        self.active_arms = list(range(K))
        self.identified_scenario = None
        self.current_time = 0
    
    def pull_real_arm(self, arm):
        """Simulate pulling a real arm and getting a reward."""
        true_mean = self.scenario_A_means[arm] if self.true_scenario == 'A' else self.scenario_B_means[arm]
        reward = np.random.binomial(1, true_mean)  # Bernoulli rewards
        self._update_real_stats(arm, reward)
        self.current_time += 1
        self.total_regret += self.optimal_mean - true_mean
        return reward
    
    def query_AGI(self, scenario, arm):
        """Simulate querying AGI for a scenario-arm pair."""
        self.total_AGI_queries += 1
        self.total_regret += self.c  # Add AGI query cost to regret
        
        if scenario == 'A':
            reward = np.random.binomial(1, self.scenario_A_means[arm])
            self._update_AGI_A_stats(arm, reward)
        else:
            reward = np.random.binomial(1, self.scenario_B_means[arm])
            self._update_AGI_B_stats(arm, reward)
        return reward
    
    def _update_real_stats(self, arm, reward):
        """Update statistics for real arm pulls."""
        n = self.pull_counts[arm]
        self.empirical_means[arm] = (self.empirical_means[arm] * n + reward) / (n + 1)
        self.pull_counts[arm] += 1
    
    def _update_AGI_A_stats(self, arm, reward):
        """Update statistics for AGI queries in scenario A."""
        n = self.pull_counts_A[arm]
        self.empirical_means_A[arm] = (self.empirical_means_A[arm] * n + reward) / (n + 1)
        self.pull_counts_A[arm] += 1
    
    def _update_AGI_B_stats(self, arm, reward):
        """Update statistics for AGI queries in scenario B."""
        n = self.pull_counts_B[arm]
        self.empirical_means_B[arm] = (self.empirical_means_B[arm] * n + reward) / (n + 1)
        self.pull_counts_B[arm] += 1
    
    def compute_UCB(self, arm, counts, means):
        """Compute Upper Confidence Bound for an arm."""
        if counts[arm] == 0:
            return float('inf')
        return means[arm] + math.sqrt(2 * math.log(self.T) / counts[arm])
    
    def compute_LCB(self, arm, counts, means):
        """Compute Lower Confidence Bound for an arm."""
        if counts[arm] == 0:
            return float('-inf')
        return means[arm] - math.sqrt(2 * math.log(self.T) / counts[arm])
    
    def phase1_identify_scenario(self):
        """Execute Phase 1: Identify the true scenario."""
        # Calculate m as per the algorithm description
        delta = sum((a - b)**2 for a, b in zip(self.scenario_A_means, self.scenario_B_means))
        m = int(math.sqrt(self.K * self.T * math.log(self.T) / delta)) if delta > 0 else 1
        m = max(m, 1)  # Ensure at least 1 pull
        
        # Pull each arm m times in real world
        for arm in range(self.K):
            for _ in range(m):
                self.pull_real_arm(arm)
        
        # Query AGI for each scenario m times
        for arm in range(self.K):
            for _ in range(m):
                self.query_AGI('A', arm)
                self.query_AGI('B', arm)
        
        # Calculate variances
        sigma_A = sum((self.empirical_means[arm] - self.empirical_means_A[arm])**2 for arm in range(self.K))
        sigma_B = sum((self.empirical_means[arm] - self.empirical_means_B[arm])**2 for arm in range(self.K))
        
        # Identify scenario
        self.identified_scenario = 'A' if sigma_A < sigma_B else 'B'
        self.phase = 2
    
    def phase2_AGI_elimination(self):
        """Execute Phase 2: Use AGI to eliminate suboptimal arms."""
        while True:
            # Calculate current deltas
            current_means = self.empirical_means_A if self.identified_scenario == 'A' else self.empirical_means_B
            max_mean = max(current_means[arm] for arm in self.active_arms)
            deltas = [max_mean - current_means[arm] for arm in self.active_arms]
            
            # Check if we should continue with AGI queries
            if all(delta < self.c for delta in deltas):
                break
            
            # Query AGI for each active arm
            for arm in self.active_arms:
                self.query_AGI(self.identified_scenario, arm)
            
            # Perform elimination
            new_active_arms = []
            for i, arm in enumerate(self.active_arms):
                keep_arm = True
                for j, other_arm in enumerate(self.active_arms):
                    if i == j:
                        continue
                    # Get UCB and LCB using AGI statistics
                    counts = self.pull_counts_A if self.identified_scenario == 'A' else self.pull_counts_B
                    means = self.empirical_means_A if self.identified_scenario == 'A' else self.empirical_means_B
                    ucb_i = self.compute_UCB(arm, counts, means)
                    lcb_j = self.compute_LCB(other_arm, counts, means)
                    if ucb_i < lcb_j:
                        keep_arm = False
                        break
                if keep_arm:
                    new_active_arms.append(arm)
            
            self.active_arms = new_active_arms
            
            if len(self.active_arms) == 1:
                break
        
        self.phase = 3
    
    def phase3_real_pulls(self):
        """Execute Phase 3: Real pulls to identify the best arm."""
        while len(self.active_arms) > 1 and self.current_time < self.T:
            # Pull each active arm
            for arm in self.active_arms:
                if self.current_time >= self.T:
                    break
                self.pull_real_arm(arm)
            
            # Perform elimination
            new_active_arms = []
            for i, arm in enumerate(self.active_arms):
                keep_arm = True
                for j, other_arm in enumerate(self.active_arms):
                    if i == j:
                        continue
                    ucb_i = self.compute_UCB(arm, self.pull_counts, self.empirical_means)
                    lcb_j = self.compute_LCB(other_arm, self.pull_counts, self.empirical_means)
                    if ucb_i < lcb_j:
                        keep_arm = False
                        break
                if keep_arm:
                    new_active_arms.append(arm)
            
            self.active_arms = new_active_arms
        
        # Commit to the remaining arm
        if len(self.active_arms) == 1:
            best_arm = self.active_arms[0]
            while self.current_time < self.T:
                self.pull_real_arm(best_arm)
    
    def run(self):
        """Run the complete algorithm."""
        # Phase 1: Identify scenario
        self.phase1_identify_scenario()
        
        # Phase 2: AGI-assisted elimination
        self.phase2_AGI_elimination()
        
        # Phase 3: Real pulls
        self.phase3_real_pulls()
        
        return self.total_regret, self.total_AGI_queries, self.active_arms[0] if len(self.active_arms) == 1 else None

# Example usage
if __name__ == "__main__":
    # Parameters
    K = 5  # Number of arms
    T = 10000  # Time horizon
    c = 0.1  # Cost of AGI query
    
    # Define scenarios
    scenario_A_means = [0.1, 0.3, 0.5, 0.7, 0.9]  # Means for arms in scenario A
    scenario_B_means = [0.9, 0.7, 0.5, 0.3, 0.1]  # Means for arms in scenario B
    true_scenario = 'A'  # The actual scenario
    
    # Initialize and run algorithm
    bandit = AGIAssistedBandit(K, T, c, scenario_A_means, scenario_B_means, true_scenario)
    total_regret, AGI_queries, best_arm = bandit.run()
    
    print(f"True scenario: {true_scenario}")
    print(f"Identified scenario: {bandit.identified_scenario}")
    print(f"Best arm identified: {best_arm}")
    print(f"Total regret: {total_regret}")
    print(f"Total AGI queries: {AGI_queries}")