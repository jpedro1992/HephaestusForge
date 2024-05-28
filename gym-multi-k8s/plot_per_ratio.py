import glob
import logging
import os
from collections import namedtuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def get_statistics(df, alg_name, avg_reward, ci_avg_reward, avg_latency, ci_avg_latency,
                   avg_cost, ci_avg_cost, avg_ep_block_prob, ci_avg_ep_block_prob,
                   avg_executionTime, ci_avg_executionTime, avg_gini, ci_avg_gini,
                   avg_ep_deploy_all, ci_ep_deploy_all, avg_ep_ffd, ci_ep_ffd, avg_ep_ffi, ci_ep_ffi,
                   avg_ep_bf1b1, ci_ep_bf1b1):
    '''
    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward Std: {}".format(alg_name, np.std(df["reward"])))

    print("{} latency Mean: {}".format(alg_name, np.mean(df["avg_latency"])))
    print("{} latency Std: {}".format(alg_name, np.std(df["avg_latency"])))

    print("{} cost Mean: {}".format(alg_name, np.mean(df["avg_cost"])))
    print("{} cost Std: {}".format(alg_name, np.std(df["avg_cost"])))

    print("{} ep block prob Mean: {}".format(alg_name, np.mean(df["ep_block_prob"])))
    print("{} ep block prob Std: {}".format(alg_name, np.std(df["ep_block_prob"])))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime Std: {}".format(alg_name, np.std(df["executionTime"])))
    '''
    check_ep_ffi = True

    avg_reward.append(np.mean(df["reward"]))
    ci_avg_reward.append(1.96 * np.std(df["reward"]) / np.sqrt(len(df["reward"])))
    avg_latency.append(np.mean(df["avg_latency"]))
    ci_avg_latency.append(1.96 * np.std(df["avg_latency"]) / np.sqrt(len(df["avg_latency"])))
    avg_cost.append(np.mean(df["avg_cost"]))
    ci_avg_cost.append(1.96 * np.std(df["avg_cost"]) / np.sqrt(len(df["avg_cost"])))
    avg_ep_block_prob.append(np.mean(df["ep_block_prob"]))
    ci_avg_ep_block_prob.append(1.96 * np.std(df["ep_block_prob"]) / np.sqrt(len(df["ep_block_prob"])))
    avg_gini.append(np.mean(df["gini"]))
    ci_avg_gini.append(1.96 * np.std(df["gini"]) / np.sqrt(len(df["gini"])))
    avg_executionTime.append(np.mean(df["executionTime"]))
    ci_avg_executionTime.append(1.96 * np.std(df["executionTime"]) / np.sqrt(len(df["executionTime"])))
    avg_ep_deploy_all.append(np.mean(df["ep_deploy_all"]))
    ci_ep_deploy_all.append(1.96 * np.std(df["ep_deploy_all"]) / np.sqrt(len(df["ep_deploy_all"])))
    avg_ep_ffd.append(np.mean(df["ep_ffd"]))
    ci_ep_ffd.append(1.96 * np.std(df["ep_ffd"]) / np.sqrt(len(df["ep_ffd"])))
    if alg_name != 'baseline' and check_ep_ffi:
        avg_ep_ffi.append(np.mean(df["ep_ffi"]))
        ci_ep_ffi.append(1.96 * np.std(df["ep_ffi"]) / np.sqrt(len(df["ep_ffi"])))
        avg_ep_bf1b1.append(np.mean(df["ep_bf1b1"]))
        ci_ep_bf1b1.append(1.96 * np.std(df["ep_bf1b1"]) / np.sqrt(len(df["ep_bf1b1"])))


if __name__ == "__main__":
    reward = 'latency'  # cost, risk or latency
    max_reward = 100  # cost= 1500, risk and latency= 100
    ylim = 120  # 1700 for cost and 120 for rest


    # testing
    path_ppo_cost = "results/karmada/v3/multi/cost/ppo_deepsets_env_karmada_num_clusters_4_reward_multi_totalSteps_200000_run_1/testing/ratio/"
    path_ppo_latency = "results/karmada/v3/multi/latency/ppo_deepsets_env_karmada_num_clusters_4_reward_multi_totalSteps_200000_run_1/testing/ratio/"
    path_ppo_inequality = "results/karmada/v3/multi/inequality/ppo_deepsets_env_karmada_num_clusters_4_reward_multi_totalSteps_200000_run_1/testing/ratio/"

    path_dqn_cost = "results/karmada/v3/multi/cost/dqn_deepsets_env_karmada_num_clusters_4_reward_multi_totalSteps_200000_run_1/testing/ratio/"
    path_dqn_latency = "results/karmada/v3/multi/latency/dqn_deepsets_env_karmada_num_clusters_4_reward_multi_totalSteps_200000_run_1/testing/ratio/"
    path_dqn_inequality = "results/karmada/v3/multi/inequality/dqn_deepsets_env_karmada_num_clusters_4_reward_multi_totalSteps_200000_run_1/testing/ratio/"

    avg_reward_ppo_cost = []
    ci_avg_reward_ppo_cost = []
    avg_latency_ppo_cost = []
    ci_avg_latency_ppo_cost = []
    avg_cost_ppo_cost = []
    ci_avg_cost_ppo_cost = []
    avg_ep_block_prob_ppo_cost = []
    ci_avg_ep_block_prob_ppo_cost = []
    avg_gini_ppo_cost = []
    ci_avg_gini_ppo_cost = []
    avg_executionTime_ppo_cost = []
    ci_executionTime_ppo_cost = []
    avg_ep_deploy_all_ppo_cost = []
    ci_ep_deploy_all_ppo_cost = []
    avg_ep_ffd_ppo_cost = []
    ci_ep_ffd_ppo_cost = []
    avg_ep_ffi_ppo_cost = []
    ci_ep_ffi_ppo_cost = []
    avg_ep_bf1b1_ppo_cost = []
    ci_ep_bf1b1_ppo_cost = []

    avg_reward_dqn_cost = []
    ci_avg_reward_dqn_cost = []
    avg_latency_dqn_cost = []
    ci_avg_latency_dqn_cost = []
    avg_cost_dqn_cost = []
    ci_avg_cost_dqn_cost = []
    avg_ep_block_prob_dqn_cost = []
    ci_avg_ep_block_prob_dqn_cost = []
    avg_gini_dqn_cost = []
    ci_avg_gini_dqn_cost = []
    avg_executionTime_dqn_cost = []
    ci_executionTime_dqn_cost = []
    avg_ep_deploy_all_dqn_cost = []
    ci_ep_deploy_all_dqn_cost = []
    avg_ep_ffd_dqn_cost = []
    ci_ep_ffd_dqn_cost = []
    avg_ep_ffi_dqn_cost = []
    ci_ep_ffi_dqn_cost = []
    avg_ep_bf1b1_dqn_cost = []
    ci_ep_bf1b1_dqn_cost = []

    avg_reward_ppo_latency = []
    ci_avg_reward_ppo_latency = []
    avg_latency_ppo_latency = []
    ci_avg_latency_ppo_latency = []
    avg_cost_ppo_latency = []
    ci_avg_cost_ppo_latency = []
    avg_ep_block_prob_ppo_latency = []
    ci_avg_ep_block_prob_ppo_latency = []
    avg_gini_ppo_latency = []
    ci_avg_gini_ppo_latency = []
    avg_executionTime_ppo_latency = []
    ci_executionTime_ppo_latency = []
    avg_ep_deploy_all_ppo_latency = []
    ci_ep_deploy_all_ppo_latency = []
    avg_ep_ffd_ppo_latency = []
    ci_ep_ffd_ppo_latency = []
    avg_ep_ffi_ppo_latency = []
    ci_ep_ffi_ppo_latency = []
    avg_ep_bf1b1_ppo_latency = []
    ci_ep_bf1b1_ppo_latency = []

    avg_reward_dqn_latency = []
    ci_avg_reward_dqn_latency = []
    avg_latency_dqn_latency = []
    ci_avg_latency_dqn_latency = []
    avg_cost_dqn_latency = []
    ci_avg_cost_dqn_latency = []
    avg_ep_block_prob_dqn_latency = []
    ci_avg_ep_block_prob_dqn_latency = []
    avg_executionTime_dqn_latency = []
    avg_gini_dqn_latency = []
    ci_avg_gini_dqn_latency = []
    ci_avg_executionTime_dqn_latency = []
    ci_executionTime_dqn_latency = []
    avg_ep_deploy_all_dqn_latency = []
    ci_ep_deploy_all_dqn_latency = []
    avg_ep_ffd_dqn_latency = []
    ci_ep_ffd_dqn_latency = []
    avg_ep_ffi_dqn_latency = []
    ci_ep_ffi_dqn_latency = []
    avg_ep_bf1b1_dqn_latency = []
    ci_ep_bf1b1_dqn_latency = []

    avg_reward_ppo_inequality = []
    ci_avg_reward_ppo_inequality = []
    avg_latency_ppo_inequality = []
    ci_avg_latency_ppo_inequality = []
    avg_cost_ppo_inequality = []
    ci_avg_cost_ppo_inequality = []
    avg_ep_block_prob_ppo_inequality = []
    ci_avg_ep_block_prob_ppo_inequality = []
    avg_gini_ppo_inequality = []
    ci_avg_gini_ppo_inequality = []
    avg_executionTime_ppo_inequality = []
    ci_executionTime_ppo_inequality = []
    avg_ep_deploy_all_ppo_inequality = []
    ci_ep_deploy_all_ppo_inequality = []
    avg_ep_ffd_ppo_inequality = []
    ci_ep_ffd_ppo_inequality = []
    avg_ep_ffi_ppo_inequality = []
    ci_ep_ffi_ppo_inequality = []
    avg_ep_bf1b1_ppo_inequality = []
    ci_ep_bf1b1_ppo_inequality = []

    avg_reward_dqn_inequality = []
    ci_avg_reward_dqn_inequality = []
    avg_latency_dqn_inequality = []
    ci_avg_latency_dqn_inequality = []
    avg_cost_dqn_inequality = []
    ci_avg_cost_dqn_inequality = []
    avg_ep_block_prob_dqn_inequality = []
    ci_avg_ep_block_prob_dqn_inequality = []
    avg_gini_dqn_inequality = []
    ci_avg_gini_dqn_inequality = []
    avg_executionTime_dqn_inequality = []
    ci_avg_executionTime_dqn_inequality = []
    avg_ep_deploy_all_dqn_inequality = []
    ci_ep_deploy_all_dqn_inequality = []
    avg_ep_ffd_dqn_inequality = []
    ci_ep_ffd_dqn_inequality = []
    avg_ep_ffi_dqn_inequality = []
    ci_ep_ffi_dqn_inequality = []
    avg_ep_bf1b1_dqn_inequality = []
    ci_ep_bf1b1_dqn_inequality = []

    # Baselines: for cpu, latency, cost, binpack, karmada
    path_cpu = "results/karmada/baselines/cpu/ratio/"
    path_binpack = "results/karmada/baselines/binpack/ratio/"
    path_latency = "results/karmada/baselines/latency/ratio/"
    path_karmada = "results/karmada/baselines/karmada/ratio/"
    baseline = 'baseline'

    avg_reward_cpu = []
    ci_avg_reward_cpu = []
    avg_latency_cpu = []
    ci_avg_latency_cpu = []
    avg_cost_cpu = []
    ci_avg_cost_cpu = []
    avg_ep_block_prob_cpu = []
    ci_avg_ep_block_prob_cpu = []
    avg_executionTime_cpu = []
    ci_executionTime_cpu = []
    avg_gini_cpu = []
    ci_avg_gini_cpu = []
    avg_ep_deploy_all_cpu = []
    ci_ep_deploy_all_cpu = []
    avg_ep_ffd_cpu = []
    ci_ep_ffd_cpu = []
    avg_ep_ffi_cpu = []
    ci_ep_ffi_cpu = []
    avg_ep_bf1b1_cpu = []
    ci_ep_bf1b1_cpu = []

    avg_reward_binpack = []
    ci_avg_reward_binpack = []
    avg_latency_binpack = []
    ci_avg_latency_binpack = []
    avg_cost_binpack = []
    ci_avg_cost_binpack = []
    avg_ep_block_prob_binpack = []
    ci_avg_ep_block_prob_binpack = []
    avg_executionTime_binpack = []
    ci_executionTime_binpack = []
    avg_gini_binpack = []
    ci_avg_gini_binpack = []
    avg_ep_deploy_all_binpack = []
    ci_ep_deploy_all_binpack = []
    avg_ep_ffd_binpack = []
    ci_ep_ffd_binpack = []
    avg_ep_ffi_binpack = []
    ci_ep_ffi_binpack = []
    avg_ep_bf1b1_binpack = []
    ci_ep_bf1b1_binpack = []

    avg_reward_latency = []
    ci_avg_reward_latency = []
    avg_latency_latency = []
    ci_avg_latency_latency = []
    avg_cost_latency = []
    ci_avg_cost_latency = []
    avg_ep_block_prob_latency = []
    ci_avg_ep_block_prob_latency = []
    avg_executionTime_latency = []
    ci_executionTime_latency = []
    avg_gini_latency = []
    ci_avg_gini_latency = []
    avg_ep_deploy_all_latency = []
    ci_ep_deploy_all_latency = []
    avg_ep_ffd_latency = []
    ci_ep_ffd_latency = []
    avg_ep_ffi_latency = []
    ci_ep_ffi_latency = []
    avg_ep_bf1b1_latency = []
    ci_ep_bf1b1_latency = []

    avg_reward_karmada = []
    ci_avg_reward_karmada = []
    avg_latency_karmada = []
    ci_avg_latency_karmada = []
    avg_cost_karmada = []
    ci_avg_cost_karmada = []
    avg_ep_block_prob_karmada = []
    ci_avg_ep_block_prob_karmada = []
    avg_executionTime_karmada = []
    ci_executionTime_karmada = []
    avg_gini_karmada = []
    ci_avg_gini_karmada = []
    avg_ep_deploy_all_karmada = []
    ci_ep_deploy_all_karmada = []
    avg_ep_ffd_karmada = []
    ci_ep_ffd_karmada = []
    avg_ep_ffi_karmada = []
    ci_ep_ffi_karmada = []
    avg_ep_bf1b1_karmada = []
    ci_ep_bf1b1_karmada = []

    if os.path.exists(path_ppo_cost):
        for file in glob.glob(f"{path_ppo_cost}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_ppo_cost, ci_avg_reward_ppo_cost,
                           avg_latency_ppo_cost, ci_avg_latency_ppo_cost,
                           avg_cost_ppo_cost, ci_avg_cost_ppo_cost,
                           avg_ep_block_prob_ppo_cost, ci_avg_ep_block_prob_ppo_cost,
                           avg_executionTime_ppo_cost, ci_executionTime_ppo_cost,
                           avg_gini_ppo_cost, ci_avg_gini_ppo_cost,
                           avg_ep_deploy_all_ppo_cost, ci_ep_deploy_all_ppo_cost,
                           avg_ep_ffd_ppo_cost, ci_ep_ffd_ppo_cost, avg_ep_ffi_ppo_cost, ci_ep_ffi_ppo_cost,
                           avg_ep_bf1b1_ppo_cost, ci_ep_bf1b1_ppo_cost)

    if os.path.exists(path_dqn_cost):
        for file in glob.glob(f"{path_dqn_cost}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_dqn_cost, ci_avg_reward_dqn_cost,
                           avg_latency_dqn_cost, ci_avg_latency_dqn_cost,
                           avg_cost_dqn_cost, ci_avg_cost_dqn_cost,
                           avg_ep_block_prob_dqn_cost, ci_avg_ep_block_prob_dqn_cost,
                           avg_executionTime_dqn_cost, ci_executionTime_dqn_cost,
                           avg_gini_dqn_cost, ci_avg_gini_dqn_cost,
                           avg_ep_deploy_all_dqn_cost, ci_ep_deploy_all_dqn_cost,
                           avg_ep_ffd_dqn_cost, ci_ep_ffd_dqn_cost, avg_ep_ffi_dqn_cost, ci_ep_ffi_dqn_cost,
                           avg_ep_bf1b1_dqn_cost, ci_ep_bf1b1_dqn_cost)

    if os.path.exists(path_ppo_latency):
        for file in glob.glob(f"{path_ppo_latency}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_ppo_latency, ci_avg_reward_ppo_latency,
                           avg_latency_ppo_latency, ci_avg_latency_ppo_latency,
                           avg_cost_ppo_latency, ci_avg_cost_ppo_latency,
                           avg_ep_block_prob_ppo_latency, ci_avg_ep_block_prob_ppo_latency,
                           avg_executionTime_ppo_latency, ci_executionTime_ppo_latency,
                           avg_gini_ppo_latency, ci_avg_gini_ppo_latency,
                           avg_ep_deploy_all_ppo_latency, ci_ep_deploy_all_ppo_latency,
                           avg_ep_ffd_ppo_latency, ci_ep_ffd_ppo_latency, avg_ep_ffi_ppo_latency, ci_ep_ffi_ppo_latency,
                           avg_ep_bf1b1_ppo_latency, ci_ep_bf1b1_ppo_latency)

    if os.path.exists(path_dqn_latency):
        for file in glob.glob(f"{path_dqn_latency}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_dqn_latency, ci_avg_reward_dqn_latency,
                           avg_latency_dqn_latency, ci_avg_latency_dqn_latency,
                           avg_cost_dqn_latency, ci_avg_cost_dqn_latency,
                           avg_ep_block_prob_dqn_latency, ci_avg_ep_block_prob_dqn_latency,
                           avg_executionTime_dqn_latency, ci_executionTime_dqn_latency,
                           avg_gini_dqn_latency, ci_avg_gini_dqn_latency,
                           avg_ep_deploy_all_dqn_latency, ci_ep_deploy_all_dqn_latency,
                           avg_ep_ffd_dqn_latency, ci_ep_ffd_dqn_latency, avg_ep_ffi_dqn_latency, ci_ep_ffi_dqn_latency,
                           avg_ep_bf1b1_dqn_latency, ci_ep_bf1b1_dqn_latency)

    if os.path.exists(path_ppo_inequality):
        for file in glob.glob(f"{path_ppo_inequality}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_ppo_inequality, ci_avg_reward_ppo_inequality,
                           avg_latency_ppo_inequality, ci_avg_latency_ppo_inequality,
                           avg_cost_ppo_inequality, ci_avg_cost_ppo_inequality,
                           avg_ep_block_prob_ppo_inequality, ci_avg_ep_block_prob_ppo_inequality,
                           avg_executionTime_ppo_inequality, ci_executionTime_ppo_inequality,
                           avg_gini_ppo_inequality, ci_avg_gini_ppo_inequality,
                           avg_ep_deploy_all_ppo_inequality, ci_ep_deploy_all_ppo_inequality,
                           avg_ep_ffd_ppo_inequality, ci_ep_ffd_ppo_inequality, avg_ep_ffi_ppo_inequality, ci_ep_ffi_ppo_inequality,
                           avg_ep_bf1b1_ppo_inequality, ci_ep_bf1b1_ppo_inequality)

    if os.path.exists(path_dqn_inequality):
        for file in glob.glob(f"{path_dqn_inequality}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_dqn_inequality, ci_avg_reward_dqn_inequality,
                           avg_latency_dqn_inequality, ci_avg_latency_dqn_inequality,
                           avg_cost_dqn_inequality, ci_avg_cost_dqn_inequality,
                           avg_ep_block_prob_dqn_inequality, ci_avg_ep_block_prob_dqn_inequality,
                           avg_executionTime_dqn_inequality, ci_avg_executionTime_dqn_inequality,
                           avg_gini_dqn_inequality, ci_avg_gini_dqn_inequality,
                           avg_ep_deploy_all_dqn_inequality, ci_ep_deploy_all_dqn_inequality,
                           avg_ep_ffd_dqn_inequality, ci_ep_ffd_dqn_inequality, avg_ep_ffi_dqn_inequality, ci_ep_ffi_dqn_inequality,
                           avg_ep_bf1b1_dqn_inequality, ci_ep_bf1b1_dqn_inequality)

    # Baselines
    if os.path.exists(path_cpu):
        for file in glob.glob(f"{path_cpu}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                           avg_reward_cpu, ci_avg_reward_cpu,
                           avg_latency_cpu, ci_avg_latency_cpu,
                           avg_cost_cpu, ci_avg_cost_cpu,
                           avg_ep_block_prob_cpu, ci_avg_ep_block_prob_cpu,
                           avg_executionTime_cpu, ci_executionTime_cpu,
                           avg_gini_cpu, ci_avg_gini_cpu,
                           avg_ep_deploy_all_cpu, ci_ep_deploy_all_cpu,
                           avg_ep_ffd_cpu, ci_ep_ffd_cpu, avg_ep_ffi_cpu, ci_ep_ffi_cpu,
                           avg_ep_bf1b1_cpu, ci_ep_bf1b1_cpu)

    if os.path.exists(path_binpack):
        for file in glob.glob(f"{path_binpack}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                           avg_reward_binpack, ci_avg_reward_binpack,
                           avg_latency_binpack, ci_avg_latency_binpack,
                           avg_cost_binpack, ci_avg_cost_binpack,
                           avg_ep_block_prob_binpack, ci_avg_ep_block_prob_binpack,
                           avg_executionTime_binpack, ci_executionTime_binpack,
                           avg_gini_binpack, ci_avg_gini_binpack,
                           avg_ep_deploy_all_binpack, ci_ep_deploy_all_binpack,
                           avg_ep_ffd_binpack, ci_ep_ffd_binpack, avg_ep_ffi_binpack, ci_ep_ffi_binpack,
                           avg_ep_bf1b1_binpack, ci_ep_bf1b1_binpack)

    if os.path.exists(path_latency):
        for file in glob.glob(f"{path_latency}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                           avg_reward_latency, ci_avg_reward_latency,
                           avg_latency_latency, ci_avg_latency_latency,
                           avg_cost_latency, ci_avg_cost_latency,
                           avg_ep_block_prob_latency, ci_avg_ep_block_prob_latency,
                           avg_executionTime_latency, ci_executionTime_latency,
                           avg_gini_latency, ci_avg_gini_latency,
                           avg_ep_deploy_all_latency, ci_ep_deploy_all_latency,
                           avg_ep_ffd_latency, ci_ep_ffd_latency, avg_ep_ffi_latency, ci_ep_ffi_latency,
                           avg_ep_bf1b1_latency, ci_ep_bf1b1_latency)

    if os.path.exists(path_karmada):
        for file in glob.glob(f"{path_karmada}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, baseline,
                           avg_reward_karmada, ci_avg_reward_karmada,
                           avg_latency_karmada, ci_avg_latency_karmada,
                           avg_cost_karmada, ci_avg_cost_karmada,
                           avg_ep_block_prob_karmada, ci_avg_ep_block_prob_karmada,
                           avg_executionTime_karmada, ci_executionTime_karmada,
                           avg_gini_karmada, ci_avg_gini_karmada,
                           avg_ep_deploy_all_karmada, ci_ep_deploy_all_karmada,
                           avg_ep_ffd_karmada, ci_ep_ffd_karmada, avg_ep_ffi_karmada, ci_ep_ffi_karmada,
                           avg_ep_bf1b1_karmada, ci_ep_bf1b1_karmada)

    # Accumulated Reward
    fig = plt.figure()
    x = [1, 2, 3, 4, 6, 8]

    plt.errorbar(x, avg_reward_ppo_cost, yerr=ci_avg_reward_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_reward_ppo_latency, yerr=ci_avg_reward_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_reward_ppo_inequality, yerr=ci_avg_reward_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_reward_dqn_cost, yerr=ci_avg_reward_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DS-DQN (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_reward_dqn_latency, yerr=ci_avg_reward_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_reward_dqn_inequality, yerr=ci_avg_reward_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=max_reward, color='black', linestyle='--', label="max reward= " + str(max_reward))
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 100)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Accumulated Reward", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_reward.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Avg. Cost
    plt.errorbar(x, avg_cost_ppo_cost, yerr=ci_avg_cost_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_cost_ppo_latency, yerr=ci_avg_cost_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_cost_ppo_inequality, yerr=ci_avg_cost_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_cost_dqn_cost, yerr=ci_avg_cost_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DS-DQN (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_cost_dqn_latency, yerr=ci_avg_cost_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_cost_dqn_inequality, yerr=ci_avg_cost_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # Baselines

    plt.errorbar(x, avg_cost_cpu, yerr=ci_avg_cost_cpu,
                 linestyle=None,
                 marker="s", color='#E897E8', label='CPU Greedy',
                 markersize=6)

    plt.errorbar(x, avg_cost_binpack, yerr=ci_avg_cost_binpack,
                 marker='o', linestyle='dashed',
                 color='#BCCE61', label='Binpack Greedy', markersize=6)

    plt.errorbar(x, avg_cost_latency, yerr=ci_avg_cost_latency,
                 marker='^', linestyle='dotted',
                 color='#DAB9AA', label='Latency Greedy', markersize=6)

    plt.errorbar(x, avg_cost_karmada, yerr=ci_avg_cost_karmada,
                 marker='x', color='#221F1E',
                 linestyle='-.', label='Karmada Greedy',
                 markersize=6)

    # plt.errorbar(x, avg_cost_ppo, yerr=ci_avg_cost_ppo, linestyle=None, marker="s", color='#3399FF',
    #             label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_cost_dqn, yerr=ci_avg_cost_dqn, color='#EDB120',
    #             linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 12)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Deployment Cost", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_cost.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Avg latency
    plt.errorbar(x, avg_latency_ppo_cost, yerr=ci_avg_latency_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_latency_ppo_latency, yerr=ci_avg_latency_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_latency_ppo_inequality, yerr=ci_avg_latency_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_latency_dqn_cost, yerr=ci_avg_latency_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DS-DQN (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_latency_dqn_latency, yerr=ci_avg_latency_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_latency_dqn_inequality, yerr=ci_avg_latency_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # Baselines

    plt.errorbar(x, avg_latency_cpu, yerr=ci_avg_latency_cpu,
                 linestyle=None,
                 marker="s", color='#E897E8', label='CPU Greedy',
                 markersize=6)

    plt.errorbar(x, avg_latency_binpack, yerr=ci_avg_latency_binpack,
                 marker='o', linestyle='dashed',
                 color='#BCCE61', label='Binpack Greedy', markersize=6)

    plt.errorbar(x, avg_latency_latency, yerr=ci_avg_latency_latency,
                 marker='^', linestyle='dotted',
                 color='#DAB9AA', label='Latency Greedy', markersize=6)

    plt.errorbar(x, avg_latency_karmada, yerr=ci_avg_latency_karmada,
                 marker='x', color='#221F1E',
                 linestyle='-.', label='Karmada Greedy',
                 markersize=6)

    # plt.errorbar(x, avg_latency_ppo, yerr=ci_avg_latency_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)

    # plt.errorbar(x, avg_latency_dqn, yerr=ci_avg_latency_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 700)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Avg. Latency (in ms)", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_latency.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Episode Block Prob
    plt.errorbar(x, avg_ep_block_prob_ppo_cost, yerr=ci_avg_ep_block_prob_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_ep_block_prob_ppo_latency, yerr=ci_avg_ep_block_prob_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_ppo_inequality, yerr=ci_avg_ep_block_prob_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_dqn_cost, yerr=ci_avg_ep_block_prob_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DS-DQN (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_ep_block_prob_dqn_latency, yerr=ci_avg_ep_block_prob_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_dqn_inequality, yerr=ci_avg_ep_block_prob_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # Baselines
    plt.errorbar(x, avg_ep_block_prob_cpu, yerr=ci_avg_ep_block_prob_cpu,
                 linestyle=None,
                 marker="s", color='#E897E8', label='CPU Greedy',
                 markersize=6)

    plt.errorbar(x, avg_ep_block_prob_binpack, yerr=ci_avg_ep_block_prob_binpack,
                 marker='o', linestyle='dashed',
                 color='#BCCE61', label='Binpack Greedy', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_latency, yerr=ci_avg_ep_block_prob_latency,
                 marker='^', linestyle='dotted',
                 color='#DAB9AA', label='Latency Greedy', markersize=6)

    plt.errorbar(x, avg_ep_block_prob_karmada, yerr=ci_avg_ep_block_prob_karmada,
                 marker='x', color='#221F1E',
                 linestyle='-.', label='Karmada Greedy',
                 markersize=6)

    # plt.errorbar(x, avg_ep_block_prob_ppo, yerr=ci_avg_ep_block_prob_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)
    # plt.errorbar(x, avg_ep_block_prob_dqn, yerr=ci_avg_ep_block_prob_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 1.0)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Percentage of Rejected Requests", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_rejected_requests.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Gini Coefficient
    plt.errorbar(x, avg_gini_ppo_cost, yerr=ci_avg_gini_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_gini_ppo_latency, yerr=ci_avg_gini_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_gini_ppo_inequality, yerr=ci_avg_gini_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_gini_dqn_cost, yerr=ci_avg_gini_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DS-DQN (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_gini_dqn_latency, yerr=ci_avg_gini_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_gini_dqn_inequality, yerr=ci_avg_gini_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # Baselines

    plt.errorbar(x, avg_gini_cpu, yerr=ci_avg_gini_cpu,
                 linestyle=None,
                 marker="s", color='#E897E8', label='CPU Greedy',
                 markersize=6)

    plt.errorbar(x, avg_gini_binpack, yerr=ci_avg_gini_binpack,
                 marker='o', linestyle='dashed',
                 color='#BCCE61', label='Binpack Greedy', markersize=6)

    plt.errorbar(x, avg_gini_latency, yerr=ci_avg_gini_latency,
                 marker='^', linestyle='dotted',
                 color='#DAB9AA', label='Latency Greedy', markersize=6)

    plt.errorbar(x, avg_gini_karmada, yerr=ci_avg_gini_karmada,
                 marker='x', color='#221F1E',
                 linestyle='-.', label='Karmada Greedy',
                 markersize=6)


    # plt.errorbar(x, avg_ep_block_prob_ppo, yerr=ci_avg_ep_block_prob_ppo, linestyle=None, marker="s", color='#3399FF',
    #              label='Deepsets PPO', markersize=6)
    # plt.errorbar(x, avg_ep_block_prob_dqn, yerr=ci_avg_ep_block_prob_dqn, color='#EDB120',
    #              linestyle='dotted', marker="s", label='Deepsets DQN', markersize=6)

    # specifying horizontal line type
    # plt.axhline(y=1800, color='black', linestyle='--', label="max reward= 1500 ")
    # plt.yscale('log')

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 0.8)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Gini Coefficient", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_gini.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Plot deploy all
    plt.errorbar(x, avg_ep_deploy_all_ppo_cost, yerr=ci_ep_deploy_all_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_ppo_latency, yerr=ci_ep_deploy_all_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_ppo_inequality, yerr=ci_ep_deploy_all_ppo_inequality,
                    marker='^', linestyle='dotted',
                    color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_dqn_cost, yerr=ci_ep_deploy_all_dqn_cost,
                    marker='x', color='#94E827',
                    linestyle='-.', label='DS-DQN (Cost)',
                    markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_dqn_latency, yerr=ci_ep_deploy_all_dqn_latency,
                    marker='o', color='#F5520C',
                    linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_dqn_inequality, yerr=ci_ep_deploy_all_dqn_inequality,
                    marker='^', color='#0481FD',
                    linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # Baselines
    '''
    plt.errorbar(x, avg_ep_deploy_all_cpu, yerr=ci_ep_deploy_all_cpu,
                    linestyle=None,
                    marker="s", color='#E897E8', label='CPU Greedy - Deploy-All',
                    markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_binpack, yerr=ci_ep_deploy_all_binpack,
                    marker='o', linestyle='dashed',
                    color='#BCCE61', label='Binpack Greedy - Deploy-All', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_latency, yerr=ci_ep_deploy_all_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy - Deploy-All', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_karmada, yerr=ci_ep_deploy_all_karmada,
                    marker='x', color='#221F1E',
                    linestyle='-.', label='Karmada Greedy - Deploy-All',

                    markersize=6)
    '''
    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 100)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Number of Deploy-All Actions", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_deploy_all.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Plot FFD
    plt.errorbar(x, avg_ep_ffd_ppo_cost, yerr=ci_ep_ffd_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='DS-PPO (Cost)',
                 markersize=6)

    plt.errorbar(x, avg_ep_ffd_ppo_latency, yerr=ci_ep_ffd_ppo_latency,
                    marker='o', linestyle='dashed',
                    color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_ffd_ppo_inequality, yerr=ci_ep_ffd_ppo_inequality,
                    marker='^', linestyle='dotted',
                    color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ep_ffd_dqn_cost, yerr=ci_ep_ffd_dqn_cost,
                    marker='x', color='#94E827',
                    linestyle='-.', label='DS-DQN (Cost)',
                    markersize=6)

    plt.errorbar(x, avg_ep_ffd_dqn_latency, yerr=ci_ep_ffd_dqn_latency,
                    marker='o', color='#F5520C',
                    linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_ffd_dqn_inequality, yerr=ci_ep_ffd_dqn_inequality,
                    marker='^', color='#0481FD',
                    linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # Baselines
    '''
    plt.errorbar(x, avg_ep_ffd_cpu, yerr=ci_ep_ffd_cpu,
                    linestyle=None,
                    marker="s", color='#E897E8', label='CPU Greedy - FFD',
                    markersize=6)

    plt.errorbar(x, avg_ep_ffd_binpack, yerr=ci_ep_ffd_binpack,
                    marker='o', linestyle='dashed',
                    color='#BCCE61', label='Binpack Greedy - FFD', markersize=6)

    plt.errorbar(x, avg_ep_ffd_latency, yerr=ci_ep_ffd_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy - FFD', markersize=6)

    plt.errorbar(x, avg_ep_ffd_karmada, yerr=ci_ep_ffd_karmada,
                    marker='x', color='#221F1E',
                    linestyle='-.', label='Karmada Greedy - FFD',

                    markersize=6)
    '''
    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 100)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Number of FFD Actions", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_ffd.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    '''
    # Plot FFI
    plt.errorbar(x, avg_ep_ffi_ppo_cost, yerr=ci_ep_ffi_ppo_cost,
                    linestyle=None,
                    marker="s", color='#77AC30', label='DS-PPO (Cost)',
                    markersize=6)

    plt.errorbar(x, avg_ep_ffi_ppo_latency, yerr=ci_ep_ffi_ppo_latency,
                    marker='o', linestyle='dashed',
                    color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_ffi_ppo_inequality, yerr=ci_ep_ffi_ppo_inequality,
                    marker='^', linestyle='dotted',
                    color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ep_ffi_dqn_cost, yerr=ci_ep_ffi_dqn_cost,
                    marker='x', color='#94E827',
                    linestyle='-.', label='DS-DQN (Cost)',
                    markersize=6)

    plt.errorbar(x, avg_ep_ffi_dqn_latency, yerr=ci_ep_ffi_dqn_latency,
                    marker='o', color='#F5520C',
                    linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_ffi_dqn_inequality, yerr=ci_ep_ffi_dqn_inequality,
                    marker='^', color='#0481FD',
                    linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)
    '''
    # Baselines
    '''
    plt.errorbar(x, avg_ep_ffi_cpu, yerr=ci_ep_ffi_cpu,
                    linestyle=None,
                    marker="s", color='#E897E8', label='CPU Greedy - FFI',
                    markersize=6)
                    
    plt.errorbar(x, avg_ep_ffi_binpack, yerr=ci_ep_ffi_binpack,
                    marker='o', linestyle='dashed',
                    color='#BCCE61', label='Binpack Greedy - FFI', markersize=6)
                    
    plt.errorbar(x, avg_ep_ffi_latency, yerr=ci_ep_ffi_latency,
                    marker='^', linestyle='dotted',
                    color='#DAB9AA', label='Latency Greedy - FFI', markersize=6)
                    
    plt.errorbar(x, avg_ep_ffi_karmada, yerr=ci_ep_ffi_karmada,
                    marker='x', color='#221F1E',
                    linestyle='-.', label='Karmada Greedy - FFI',
                    
                    markersize=6)
    '''
    '''
    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 100)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Number of FFI Actions", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_ffi.pdf', dpi=250, bbox_inches='tight')
    plt.close()
    '''

    # Plot BF1B1
    fig = plt.figure()

    plt.errorbar(x, avg_ep_bf1b1_ppo_cost, yerr=ci_ep_bf1b1_ppo_cost,
                    linestyle=None,
                    marker="s", color='#77AC30', label='DS-PPO (Cost)',
                    markersize=6)

    plt.errorbar(x, avg_ep_bf1b1_ppo_latency, yerr=ci_ep_bf1b1_ppo_latency,
                    marker='o', linestyle='dashed',
                    color='#D95319', label='DS-PPO (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_bf1b1_ppo_inequality, yerr=ci_ep_bf1b1_ppo_inequality,
                    marker='^', linestyle='dotted',
                    color='#3399FF', label='DS-PPO (Inequality)', markersize=6)

    plt.errorbar(x, avg_ep_bf1b1_dqn_cost, yerr=ci_ep_bf1b1_dqn_cost,
                    marker='x', color='#94E827',
                    linestyle='-.', label='DS-DQN (Cost)',
                    markersize=6)

    plt.errorbar(x, avg_ep_bf1b1_dqn_latency, yerr=ci_ep_bf1b1_dqn_latency,
                    marker='o', color='#F5520C',
                    linestyle='dashed', label='DS-DQN (Latency)', markersize=6)

    plt.errorbar(x, avg_ep_bf1b1_dqn_inequality, yerr=ci_ep_bf1b1_dqn_inequality,
                    marker='^', color='#0481FD',
                    linestyle='dotted', label='DS-DQN (Inequality)', markersize=6)

    # set x and y limits
    # plt.xlim(0, 129)
    plt.ylim(0, 100)

    # set x-axis label
    plt.xlabel("Ratio", fontsize=14)

    # set y-axis label
    plt.ylabel("Number of BF1B1 Actions", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_ratio_bf1b1.pdf', dpi=250, bbox_inches='tight')
    plt.close()
