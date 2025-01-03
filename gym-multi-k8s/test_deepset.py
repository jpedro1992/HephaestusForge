import logging

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from tqdm import tqdm
from envs.karmada_scheduling_env import KarmadaSchedulingEnv
from envs.dqn_deepset import DQN_DeepSets
from envs.ppo_deepset import PPO_DeepSets

SEED = 2
env_kwargs = {"n_nodes": 10, "arrival_rate_r": 100, "call_duration_r": 1, "episode_length": 100}
MONITOR_PATH = f"./results/test/ppo_deepset_{SEED}_n{env_kwargs['n_nodes']}_lam{env_kwargs['arrival_rate_r']}_mu{env_kwargs['call_duration_r']}.monitor.csv"

# Logging
logging.basicConfig(filename='run_test.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

if __name__ == "__main__":
    # Define here variables for testing
    num_clusters = [4]  # 4, 8, 12, 16, 32
    reward_function = 'multi'
    alg = 'dqn'
    strategy = "inequality/"
    latency_weight = 0.0
    cost_weight = 0.0  # 0.0
    gini_weight = 1.0

    episodes = 2000
    episode_length = 100
    call_duration_r = 1

    replicas = [4, 8, 12, 16, 24, 32]  # 4, 8, 12, 16, 24, 32

    i = 0
    for c in num_clusters:
        for r in replicas:
            # min = c
            # max = 4 * c
            env = KarmadaSchedulingEnv(num_clusters=c, arrival_rate_r=episode_length, call_duration_r=call_duration_r,
                                       episode_length=episode_length,
                                       latency_weight=latency_weight, cost_weight=cost_weight, gini_weight=gini_weight,
                                       min_replicas=r, max_replicas=r,
                                       reward_function=reward_function)
            env.reset()
            _, _, _, info = env.step(0)
            info_keywords = tuple(info.keys())

            envs = DummyVecEnv([lambda: KarmadaSchedulingEnv(
                num_clusters=c, arrival_rate_r=episode_length, call_duration_r=call_duration_r,
                episode_length=episode_length,
                latency_weight=latency_weight, cost_weight=cost_weight, gini_weight=gini_weight,
                reward_function=reward_function,
                min_replicas=r, max_replicas=r,
                file_results_name=str(i) + '_karmada_gym_results_num_clusters_' + str(c) + '_replicas_' + str(r))
                                ])
            envs = VecMonitor(envs, MONITOR_PATH, info_keywords=info_keywords)

            # PPO or DQN
            agent = None
            if alg == "ppo":
                agent = PPO_DeepSets(envs, seed=SEED, tensorboard_log=None)
            elif alg == 'dqn':
                agent = DQN_DeepSets(envs, seed=SEED, tensorboard_log=None)
            else:
                print('Invalid algorithm!')

            # Adapt the path accordingly
            agent.load(f"./results/karmada/"
                       + reward_function + "/" + strategy + alg + "_deepsets_env_karmada_num_clusters_4_reward_"
                       + reward_function + "_totalSteps_200000_run_1/"
                       + alg + "_deepsets_env_karmada_num_clusters_4_reward_"
                       + reward_function + "_totalSteps_200000")

            # Test the agent for 100 episodes
            for _ in tqdm(range(episodes)):
                obs = envs.reset()
                action_mask = np.array(envs.env_method("action_masks"))
                done = False
                while not done:
                    action = agent.predict(obs, action_mask)
                    obs, reward, dones, info = envs.step(action)
                    action_mask = np.array(envs.env_method("action_masks"))
                    done = dones[0]

            i += 1
