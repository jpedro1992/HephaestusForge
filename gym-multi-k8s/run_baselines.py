import logging

import numpy as np
from tqdm import tqdm
from envs.karmada_scheduling_env import KarmadaSchedulingEnv
from envs.utils import cpu_greedy_policy, latency_greedy_policy, cost_greedy_policy, binpack_greedy_policy, \
    karmada_greedy_policy

MONITOR_PATH = "./results/baselines.csv"

# Logging
logging.basicConfig(filename='run_baselines.log', filemode='w', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

SEED = 2
CPU_GREEDY = 'cpu'
BINPACK_GREEDY = 'binpack'
LATENCY_GREEDY = 'latency'
COST_GREEDY = 'cost'
KARMADA_GREEDY = 'karmada'

if __name__ == "__main__":
    policy = LATENCY_GREEDY
    num_clusters = [4] # 4, 8, 12, 16, 32

    reward_function = 'multi'
    latency_weight = 0.0
    cost_weight = 1.0  # 0.0
    gini_weight = 0.0

    episodes = 100
    num_episodes = 100
    call_duration_r = 1

    replicas = [4, 8, 12, 16, 24, 32] # 4, 8, 12, 16, 24, 32
    test_replicas = True

    i = 0
    for c in num_clusters:
        if test_replicas:
            for r in replicas:
                # print("Initiating run for {} with {} clusters and {} replicas".format(policy, c, r))
                min = r
                max = r
                env = KarmadaSchedulingEnv(num_clusters=c, arrival_rate_r=100, call_duration_r=1,
                                           episode_length=100, latency_weight=latency_weight, cost_weight=cost_weight,
                                           gini_weight=gini_weight, reward_function='multi',
                                           min_replicas=min, max_replicas=max,
                                           seed=SEED,
                                           file_results_name=str(
                                               i) + "_" + policy + '_baselines_gym_num_clusters_' + str(
                                               c) + '_replicas_' + str(r))
                env.reset()
                _, _, _, info = env.step(0)
                info_keywords = tuple(info.keys())
                env = KarmadaSchedulingEnv(num_clusters=c, arrival_rate_r=100, call_duration_r=1,
                                           episode_length=100, latency_weight=latency_weight, cost_weight=cost_weight,
                                           gini_weight=gini_weight, reward_function='multi',
                                           min_replicas=min, max_replicas=max,
                                           seed=SEED,
                                           file_results_name=str(
                                               i) + "_" + policy + '_baselines_gym_num_clusters_' + str(
                                               c) + '_replicas_' + str(r))

                returns = []
                for _ in tqdm(range(num_episodes)):
                    obs = env.reset()
                    action_mask = env.action_masks()
                    num_actions = c + 1
                    return_ = 0.0
                    done = False
                    while not done:
                        if policy == CPU_GREEDY:
                            action = cpu_greedy_policy(num_actions, env, action_mask)
                        elif policy == LATENCY_GREEDY:
                            action = latency_greedy_policy(num_actions, action_mask, env.latency,
                                                           env.deployment_request.latency_threshold)
                        elif policy == COST_GREEDY:
                            action = cost_greedy_policy(num_actions, env, action_mask)
                        elif policy == BINPACK_GREEDY:
                            action = binpack_greedy_policy(num_actions, env, action_mask)
                        elif policy == KARMADA_GREEDY:
                            action = karmada_greedy_policy(num_actions, env, action_mask)
                        else:
                            print("unrecognized policy!")

                        obs, reward, done, info = env.step(action)
                        action_mask = env.action_masks()
                        return_ += reward
                    returns.append(return_)

                # print(f"{np.mean(returns)} +/- {1.96 * np.std(returns) / np.sqrt(len(returns))}")
                i += 1
        else:
            # print("Initiating run for {} with {} clusters".format(policy, c))
            min = c
            max = 4 * c
            env = KarmadaSchedulingEnv(num_clusters=c, arrival_rate_r=100, call_duration_r=1,
                                       episode_length=100, latency_weight=latency_weight, cost_weight=cost_weight,
                                       gini_weight=gini_weight, reward_function='multi',
                                       min_replicas=min, max_replicas=max,
                                       seed=SEED,
                                       file_results_name=str(i) + "_" + policy + '_baselines_gym_num_clusters_' + str(
                                           c))  # + '_replicas_' + str(r))
            env.reset()
            _, _, _, info = env.step(0)
            info_keywords = tuple(info.keys())
            env = KarmadaSchedulingEnv(num_clusters=c, arrival_rate_r=100, call_duration_r=1,
                                       episode_length=100, latency_weight=latency_weight, cost_weight=cost_weight,
                                       gini_weight=gini_weight, reward_function='multi',
                                       min_replicas=min, max_replicas=max,
                                       seed=SEED,
                                       file_results_name=str(i) + "_" + policy + '_baselines_gym_num_clusters_' + str(
                                           c))  # + '_replicas_' + str(r))

            returns = []
            for _ in tqdm(range(num_episodes)):
                obs = env.reset()
                action_mask = env.action_masks()
                num_actions = c + 1
                return_ = 0.0
                done = False
                while not done:
                    if policy == CPU_GREEDY:
                        action = cpu_greedy_policy(num_actions, env, action_mask)
                    elif policy == LATENCY_GREEDY:
                        action = latency_greedy_policy(num_actions, action_mask, env.latency,
                                                       env.deployment_request.latency_threshold)
                    elif policy == COST_GREEDY:
                        action = cost_greedy_policy(num_actions, env, action_mask)
                    elif policy == BINPACK_GREEDY:
                        action = binpack_greedy_policy(num_actions, env, action_mask)
                    elif policy == KARMADA_GREEDY:
                        action = karmada_greedy_policy(num_actions, env, action_mask)
                    else:
                        print("unrecognized policy!")

                    obs, reward, done, info = env.step(action)
                    action_mask = env.action_masks()
                    return_ += reward
                returns.append(return_)


        # print(f"{np.mean(returns)} +/- {1.96 * np.std(returns) / np.sqrt(len(returns))}")
        i += 1