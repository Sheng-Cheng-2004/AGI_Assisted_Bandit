import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from alg2 import AGIAssistedBandit

# Fixed parameters
K = 5
c = 0.1
scenario_A_means = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
scenario_B_means = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
true_scenario = 'A'

# # Range of T values to test
# T_values = np.logspace(2, 5, num=20, dtype=int)  # From 100 to 100,000
# num_trials = 20  # Number of trials for each T to average results

# # Storage for results
# avg_regrets = []
# std_regrets = []

# for T in tqdm(T_values):
#     regrets = []
#     for _ in range(num_trials):
#         bandit = AGIAssistedBandit(K, T, c, scenario_A_means, scenario_B_means, true_scenario)
#         total_regret, _, _ = bandit.run()
#         regrets.append(total_regret)
    
#     avg_regrets.append(np.mean(regrets))
#     std_regrets.append(np.std(regrets))

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.errorbar(T_values, avg_regrets, yerr=std_regrets, fmt='-o', capsize=5, label='Empirical Regret')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Time Horizon (T)', fontsize=12)
# plt.ylabel('Total Regret', fontsize=12)
# plt.title('Regret vs Time Horizon for AGI-Assisted Bandit', fontsize=14)

# # Plot reference lines for comparison
# for exponent in [0.5, 1.0]:
#     ref_line = T_values**exponent
#     plt.plot(T_values, ref_line, '--', label=f'T^{exponent} reference')

# plt.legend(fontsize=10)
# plt.grid(True, which="both", ls="--")
# plt.tight_layout()
# plt.show()

# # Plot the ratio of regret to sqrt(T) to check sublinearity
# plt.figure(figsize=(10, 6))
# plt.plot(T_values, np.array(avg_regrets) / np.sqrt(T_values), '-o', label='Regret/sqrt(T)')
# plt.xscale('log')
# plt.xlabel('Time Horizon (T)', fontsize=12)
# plt.ylabel('Regret / sqrt(T)', fontsize=12)
# plt.title('Normalized Regret to Check Sublinearity', fontsize=14)
# plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
# plt.legend(fontsize=10)
# plt.grid(True, which="both", ls="--")
# plt.tight_layout()
# plt.show()


# num_trials = 100
# T_values = [100, 500, 1000, 5000]
# correct_identification = []

# for T in T_values:
#     correct = 0
#     for _ in range(num_trials):
#         bandit = AGIAssistedBandit(K, T, c, scenario_A_means, scenario_B_means, true_scenario)
#         bandit.phase1_identify_scenario()
#         if bandit.identified_scenario == true_scenario:
#             correct += 1
#     correct_identification.append(correct/num_trials)

# plt.figure(figsize=(10, 6))
# plt.plot(T_values, correct_identification, 'o-')
# plt.xlabel('Time Horizon (T)', fontsize=12)
# plt.ylabel('Probability of Correct Identification', fontsize=12)
# plt.title('Scenario Identification Accuracy', fontsize=14)
# plt.grid(True)
# plt.show()

# num_trials = 100
# T_values = [100, 500, 1000, 5000]
# accuracy = []

# true_best_arm = np.argmax(scenario_A_means if true_scenario == 'A' else scenario_B_means)

# for T in T_values:
#     correct = 0
#     for _ in range(num_trials):
#         bandit = AGIAssistedBandit(K, T, c, scenario_A_means, scenario_B_means, true_scenario)
#         _, _, best_arm = bandit.run()
#         if best_arm == true_best_arm:
#             correct += 1
#     accuracy.append(correct/num_trials)

# plt.figure(figsize=(10, 6))
# plt.plot(T_values, accuracy, 'o-')
# plt.xlabel('Time Horizon (T)', fontsize=12)
# plt.ylabel('Probability of Correct Best Arm', fontsize=12)
# plt.title('Final Arm Selection Accuracy', fontsize=14)
# plt.grid(True)
# plt.show()

# class AGIAssistedBanditDetailed(AGIAssistedBandit):
#     def run(self):
#         # Phase 1
#         self.phase1_identify_scenario()
#         phase1_regret = self.total_regret
        
#         # Phase 2
#         self.phase2_AGI_elimination()
#         phase2_regret = self.total_regret - phase1_regret
        
#         # Phase 3
#         self.phase3_real_pulls()
#         phase3_regret = self.total_regret - phase2_regret - phase1_regret
        
#         return phase1_regret, phase2_regret, phase3_regret

# T_values = [100, 500, 1000, 1500 ,2000,3000, 5000,8000,10000,13000,15000,20000]
# phase1 = []
# phase2 = []
# phase3 = []

# for T in T_values:
#     p1, p2, p3 = AGIAssistedBanditDetailed(K, T, c, scenario_A_means, scenario_B_means, true_scenario).run()
#     phase1.append(p1)
#     phase2.append(p2)
#     phase3.append(p3)

# plt.figure(figsize=(12, 6))
# plt.stackplot(T_values, [phase1, phase2, phase3], 
#               labels=['Phase 1 (Scenario ID)', 'Phase 2 (AGI Elimination)', 'Phase 3 (Real Pulls)'])
# plt.xlabel('Time Horizon (T)', fontsize=12)
# plt.ylabel('Regret Contribution', fontsize=12)
# plt.title('Regret Decomposition by Algorithm Phase', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.show()





class AGIAssistedBanditPhaseAccurate(AGIAssistedBandit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase3_real_pulls = 0
    
    def pull_real_arm(self, arm):
        if self.phase == 3:
            self.phase3_real_pulls += 1
        return super().pull_real_arm(arm)

# Parameters
K = 5
T = 1000  # Fixed time horizon
num_trials = 100
c_values = np.linspace(0.01, 0.5, 20)  # Test a range of AGI costs
scenario_A_means = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
scenario_B_means = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
true_scenario = 'A'

# Storage
# phase3_real_pulls = []
total_agi_queries = []

for c in c_values:
    AGIquery = []
    for _ in range(num_trials):
        bandit = AGIAssistedBandit(K, T, c, scenario_A_means, scenario_B_means, true_scenario)
        bandit.run()
        # phase3_real_pulls.append(bandit.phase3_real_pulls)
        AGIquery.append(bandit.total_AGI_queries)
    total_agi_queries.append(np.mean(AGIquery))

# Plot
plt.figure(figsize=(12, 6))
# plt.plot(c_values, phase3_real_pulls, 'r-o', label='Phase 3 Real Pulls', linewidth=2)
plt.plot(c_values, total_agi_queries, 'b-o', label='Total AGI Queries', linewidth=2)
plt.xlabel('AGI Cost (c)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(f'Phase 3 AGI queries vs AGI Cost (T={T})', fontsize=14)
plt.legend()
plt.grid(True)

# Annotate key regions
# plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
# plt.text(0.1, max(phase3_real_pulls)*0.7, 'Low cost: AGI dominates\n(Phase 3 pulls suppressed)', 
#          ha='center', va='center', backgroundcolor='white')

# plt.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
# plt.text(0.3, max(phase3_real_pulls)*0.3, 'High cost: Real pulls dominate\n(AGI used sparingly)', 
#          ha='center', va='center', backgroundcolor='white')

plt.tight_layout()
plt.show()