import glob
import logging
import os
from collections import namedtuple
import seaborn as sns
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from envs.utils import set_box_color

matplotlib.use('TkAgg')


def get_statistics(df, alg_name, avg_reward, ci_avg_reward, avg_latency, ci_avg_latency,
                   avg_cost, ci_avg_cost, avg_ep_block_prob, ci_avg_ep_block_prob,
                   avg_executionTime, ci_avg_executionTime, avg_gini, ci_avg_gini,
                   avg_ep_deploy_all, ci_ep_deploy_all, avg_ep_ffd, ci_ep_ffd):

    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward 95% CI: {}".format(alg_name, 1.96 * np.std(df["reward"]) / np.sqrt(len(df["reward"]))))

    print("{} latency Mean: {}".format(alg_name, np.mean(df["avg_latency"])))
    print("{} latency 95% CI: {}".format(alg_name, 1.96 * np.std(df["avg_latency"]) / np.sqrt(len(df["avg_latency"]))))

    print("{} cost Mean: {}".format(alg_name, np.mean(df["avg_cost"])))
    print("{} cost 95% CI: {}".format(alg_name, 1.96 * np.std(df["avg_cost"]) / np.sqrt(len(df["avg_cost"]))))

    print("{} ep block prob Mean: {}".format(alg_name, np.mean(df["ep_block_prob"])))
    print("{} ep block prob 95% CI: {}".format(alg_name, 1.96 * np.std(df["ep_block_prob"]) / np.sqrt(len(df["ep_block_prob"]))))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime 95% CI: {}".format(alg_name, 1.96 * np.std(df["executionTime"]) / np.sqrt(len(df["executionTime"]))))


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


if __name__ == "__main__":
    reward = 'latency'  # cost, risk or latency
    max_reward = 100  # cost= 1500, risk and latency= 100
    ylim = 120  # 1700 for cost and 120 for rest

    # testing
    path_ppo_cost = "results/karmada/v1/per_cluster/ppo/cost/"
    path_ppo_latency = "results/karmada/v1/per_cluster/ppo/latency/"
    path_ppo_inequality = "results/karmada/v1/per_cluster/ppo/inequality/"

    path_dqn_cost = "results/karmada/v1/per_cluster/dqn/cost/"
    path_dqn_latency = "results/karmada/v1/per_cluster/dqn/latency/"
    path_dqn_inequality = "results/karmada/v1/per_cluster/dqn/inequality/"

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
    ci_executionTime_dqn_inequality = []
    avg_ep_deploy_all_dqn_inequality = []
    ci_ep_deploy_all_dqn_inequality = []
    avg_ep_ffd_dqn_inequality = []
    ci_ep_ffd_dqn_inequality = []

    # Baselines: for cpu, latency, cost, binpack, karmada
    path_cpu = "results/karmada/v1/per_cluster/baselines/cpu/"
    path_binpack = "results/karmada/v1/per_cluster/baselines/binpack/"
    path_latency = "results/karmada/v1/per_cluster/baselines/latency/"
    path_karmada = "results/karmada/v1/per_cluster/baselines/karmada/"

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

    df_ppo_cost = pd.DataFrame()
    df_ppo_latency = pd.DataFrame()
    df_ppo_inequality = pd.DataFrame()
    df_dqn_cost = pd.DataFrame()
    df_dqn_latency = pd.DataFrame()
    df_dqn_inequality = pd.DataFrame()

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
                           avg_ep_ffd_ppo_cost, ci_ep_ffd_ppo_cost)

            # aggregate all df in one
            df_ppo_cost = pd.concat([df_ppo_cost, df], ignore_index=True)


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
                           avg_ep_ffd_dqn_cost, ci_ep_ffd_dqn_cost)

            # aggregate all df in one
            df_dqn_cost = pd.concat([df_dqn_cost, df], ignore_index=True)

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
                            avg_ep_ffd_ppo_latency, ci_ep_ffd_ppo_latency)

            # aggregate all df in one
            df_ppo_latency = pd.concat([df_ppo_latency, df], ignore_index=True)

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
                           avg_ep_ffd_dqn_latency, ci_ep_ffd_dqn_latency)

            # aggregate all df in one
            df_dqn_latency = pd.concat([df_dqn_latency, df], ignore_index=True)

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
                           avg_ep_ffd_ppo_inequality, ci_ep_ffd_ppo_inequality)

            # aggregate all df in one
            df_ppo_inequality = pd.concat([df_ppo_inequality, df], ignore_index=True)

    if os.path.exists(path_dqn_inequality):
        for file in glob.glob(f"{path_dqn_inequality}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_dqn_inequality, ci_avg_reward_dqn_inequality,
                           avg_latency_dqn_inequality, ci_avg_latency_dqn_inequality,
                           avg_cost_dqn_inequality, ci_avg_cost_dqn_inequality,
                           avg_ep_block_prob_dqn_inequality, ci_avg_ep_block_prob_dqn_inequality,
                           avg_executionTime_dqn_inequality, ci_executionTime_dqn_inequality,
                           avg_gini_dqn_inequality, ci_avg_gini_dqn_inequality,
                           avg_ep_deploy_all_dqn_inequality, ci_ep_deploy_all_dqn_inequality,
                           avg_ep_ffd_dqn_inequality, ci_ep_ffd_dqn_inequality)

            # aggregate all df in one
            df_dqn_inequality = pd.concat([df_dqn_inequality, df], ignore_index=True)


    # Baselines
    df_cpu = pd.DataFrame()
    df_binpack = pd.DataFrame()
    df_latency = pd.DataFrame()
    df_karmada = pd.DataFrame()

    if os.path.exists(path_cpu):
        for file in glob.glob(f"{path_cpu}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_cpu, ci_avg_reward_cpu,
                           avg_latency_cpu, ci_avg_latency_cpu,
                           avg_cost_cpu, ci_avg_cost_cpu,
                           avg_ep_block_prob_cpu, ci_avg_ep_block_prob_cpu,
                           avg_executionTime_cpu, ci_executionTime_cpu,
                           avg_gini_cpu, ci_avg_gini_cpu,
                           avg_ep_deploy_all_cpu, ci_ep_deploy_all_cpu,
                           avg_ep_ffd_cpu, ci_ep_ffd_cpu)

            # aggregate all df in one
            df_cpu = pd.concat([df_cpu, df], ignore_index=True)

    if os.path.exists(path_binpack):
        for file in glob.glob(f"{path_binpack}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_binpack, ci_avg_reward_binpack,
                           avg_latency_binpack, ci_avg_latency_binpack,
                           avg_cost_binpack, ci_avg_cost_binpack,
                           avg_ep_block_prob_binpack, ci_avg_ep_block_prob_binpack,
                           avg_executionTime_binpack, ci_executionTime_binpack,
                           avg_gini_binpack, ci_avg_gini_binpack,
                            avg_ep_deploy_all_binpack, ci_ep_deploy_all_binpack,
                            avg_ep_ffd_binpack, ci_ep_ffd_binpack)

            # aggregate all df in one
            df_binpack = pd.concat([df_binpack, df], ignore_index=True)


    if os.path.exists(path_latency):
        for file in glob.glob(f"{path_latency}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_latency, ci_avg_reward_latency,
                           avg_latency_latency, ci_avg_latency_latency,
                           avg_cost_latency, ci_avg_cost_latency,
                           avg_ep_block_prob_latency, ci_avg_ep_block_prob_latency,
                           avg_executionTime_latency, ci_executionTime_latency,
                           avg_gini_latency, ci_avg_gini_latency,
                            avg_ep_deploy_all_latency, ci_ep_deploy_all_latency,
                            avg_ep_ffd_latency, ci_ep_ffd_latency)

            # aggregate all df in one
            df_latency = pd.concat([df_latency, df], ignore_index=True)

    if os.path.exists(path_karmada):
        for file in glob.glob(f"{path_karmada}/*_gym_*.csv"):
            print(f"\n######### Opening {file} #########")
            df = pd.read_csv(file)
            get_statistics(df, file,
                           avg_reward_karmada, ci_avg_reward_karmada,
                           avg_latency_karmada, ci_avg_latency_karmada,
                           avg_cost_karmada, ci_avg_cost_karmada,
                           avg_ep_block_prob_karmada, ci_avg_ep_block_prob_karmada,
                           avg_executionTime_karmada, ci_executionTime_karmada,
                           avg_gini_karmada, ci_avg_gini_karmada,
                            avg_ep_deploy_all_karmada, ci_ep_deploy_all_karmada,
                            avg_ep_ffd_karmada, ci_ep_ffd_karmada)

            # aggregate all df in one
            df_karmada = pd.concat([df_karmada, df], ignore_index=True)

    # Accumulated Reward
    fig = plt.figure()
    x = [4, 8, 12, 16, 32]

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
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Accumulated Reward", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_reward.pdf', dpi=250, bbox_inches='tight')
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
    plt.xlim(0, 33)
    plt.ylim(0, 14)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Deployment Cost", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_cost.pdf', dpi=250, bbox_inches='tight')
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
    plt.xlim(0, 33)
    plt.ylim(0, 800)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Avg. Latency (in ms)", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_latency.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Episode Block Prob
    # sns.pointplot(data=df_ppo_cost["ep_block_prob"], errorbar='ci')

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
    plt.xlim(0, 33)
    plt.ylim(0, 1.0)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Percentage of Rejected Requests", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_rejected_requests.pdf', dpi=250, bbox_inches='tight')
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
    plt.xlim(0, 33)
    plt.ylim(0, 1)

    # set x-axis label
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Gini Coefficient", fontsize=14)

    # show and save figure
    plt.legend(ncols=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_gini.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Plot deploy all
    plt.errorbar(x, avg_ep_deploy_all_ppo_cost, yerr=ci_ep_deploy_all_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='PPO (Cost) - Deploy-All',
                 markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_ppo_latency, yerr=ci_ep_deploy_all_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='PPO (Latency) - Deploy-All', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_ppo_inequality, yerr=ci_ep_deploy_all_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='PPO (Inequality) - Deploy-All', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_dqn_cost, yerr=ci_ep_deploy_all_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DQN (Cost) - Deploy-All',
                 markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_dqn_latency, yerr=ci_ep_deploy_all_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DQN (Latency) - Deploy-All', markersize=6)

    plt.errorbar(x, avg_ep_deploy_all_dqn_inequality, yerr=ci_ep_deploy_all_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DQN (Inequality) - Deploy-All', markersize=6)

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
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Number of Deploy-All Actions", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_deploy_all.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Plot FFD
    plt.errorbar(x, avg_ep_ffd_ppo_cost, yerr=ci_ep_ffd_ppo_cost,
                 linestyle=None,
                 marker="s", color='#77AC30', label='PPO (Cost) - FFD',
                 markersize=6)

    plt.errorbar(x, avg_ep_ffd_ppo_latency, yerr=ci_ep_ffd_ppo_latency,
                 marker='o', linestyle='dashed',
                 color='#D95319', label='PPO (Latency) - FFD', markersize=6)

    plt.errorbar(x, avg_ep_ffd_ppo_inequality, yerr=ci_ep_ffd_ppo_inequality,
                 marker='^', linestyle='dotted',
                 color='#3399FF', label='PPO (Inequality) - FFD', markersize=6)

    plt.errorbar(x, avg_ep_ffd_dqn_cost, yerr=ci_ep_ffd_dqn_cost,
                 marker='x', color='#94E827',
                 linestyle='-.', label='DQN (Cost) - FFD',
                 markersize=6)

    plt.errorbar(x, avg_ep_ffd_dqn_latency, yerr=ci_ep_ffd_dqn_latency,
                 marker='o', color='#F5520C',
                 linestyle='dashed', label='DQN (Latency) - FFD', markersize=6)

    plt.errorbar(x, avg_ep_ffd_dqn_inequality, yerr=ci_ep_ffd_dqn_inequality,
                 marker='^', color='#0481FD',
                 linestyle='dotted', label='DQN (Inequality) - FFD', markersize=6)

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
    plt.xlabel("Total Number of clusters", fontsize=14)

    # set y-axis label
    plt.ylabel("Number of FFD Actions", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot_per_cluster_ffd.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    # Plot CDF Latency
    fig = plt.figure()
    sns.ecdfplot(data=df_ppo_cost['avg_latency'], color='#77AC30', label='Deepsets PPO (Cost)')
    sns.ecdfplot(data=df_ppo_latency['avg_latency'], color='#D95319', label='Deepsets PPO (Latency)')
    sns.ecdfplot(data=df_ppo_inequality['avg_latency'], color='#3399FF', label='Deepsets PPO (Inequality)')

    sns.ecdfplot(data=df_dqn_cost['avg_latency'], color='#94E827', label='Deepsets DQN (Cost)')
    sns.ecdfplot(data=df_dqn_latency['avg_latency'], color='#F5520C', label='Deepsets DQN (Latency)')
    sns.ecdfplot(data=df_dqn_inequality['avg_latency'], color='#0481FD', label='Deepsets DQN (Inequality)')

    sns.ecdfplot(data=df_cpu['avg_latency'], color='#E897E8', label='CPU-Greedy')
    sns.ecdfplot(data=df_binpack['avg_latency'], color='#BCCE61', label='Binpack-Greedy')
    sns.ecdfplot(data=df_latency['avg_latency'], color='#DAB9AA', label='Latency-Greedy')
    sns.ecdfplot(data=df_karmada['avg_latency'], color='#221F1E', label='Karmada-Greedy')

    plt.xlabel("Latency (in ms)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()
    plt.savefig('plot_per_cluster_cdf_seaborn_latency.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['PPO (Cost)', 'PPO (Latency)', 'PPO (Inequality)',
             'DQN (Cost)', 'DQN (Latency)', 'DQN (Inequality)',
             'CPU-Greedy', 'Binpack-Greedy', 'Latency-Greedy', 'Karmada-Greedy']

    data_ppo_cost = [df_ppo_cost['avg_cost'].tolist()]
    data_ppo_latency = [df_ppo_latency['avg_cost'].tolist()]
    data_ppo_inequality = [df_ppo_inequality['avg_cost'].tolist()]
    data_dqn_cost = [df_dqn_cost['avg_cost'].tolist()]
    data_dqn_latency = [df_dqn_latency['avg_cost'].tolist()]
    data_dqn_inequality = [df_dqn_inequality['avg_cost'].tolist()]
    data_cpu_greedy = [df_cpu['avg_cost'].tolist()]
    data_binpack_greedy = [df_binpack['avg_cost'].tolist()]
    data_latency_greedy = [df_latency['avg_cost'].tolist()]
    data_karmada_greedy = [df_karmada['avg_cost'].tolist()]

    a = plt.boxplot(data_ppo_cost, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_latency, positions=[5], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality, positions=[10], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    d = plt.boxplot(data_dqn_cost, positions=[15], whiskerprops=dict(ls='dotted'), widths=width, flierprops=red_square)
    e = plt.boxplot(data_dqn_latency, positions=[20], whiskerprops=dict(ls='-.'), widths=width, flierprops=red_square)
    f = plt.boxplot(data_dqn_inequality, positions=[25], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    g = plt.boxplot(data_cpu_greedy, positions=[30], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)
    h = plt.boxplot(data_binpack_greedy, positions=[35], whiskerprops=dict(ls='-.'), widths=width,
                    flierprops=red_square)
    i = plt.boxplot(data_latency_greedy, positions=[40], whiskerprops=dict(ls='--'), widths=width,
                    flierprops=red_square)
    j = plt.boxplot(data_karmada_greedy, positions=[45], whiskerprops=dict(ls='dotted'), widths=width,
                    flierprops=red_square)

    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#94E827')
    set_box_color(e, '#F5520C')
    set_box_color(f, '#0481FD')
    set_box_color(g, '#E897E8')
    set_box_color(h, '#BCCE61')
    set_box_color(i, '#DAB9AA')
    set_box_color(j, '#221F1E')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='Deepsets PPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='Deepsets PPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='Deepsets PPO (Inequality)')
    plt.plot([], c='#94E827', ls='dotted', label='Deepsets DQN (Cost)')
    plt.plot([], c='#F5520C', ls='-.', label='Deepsets DQN (Latency)')
    plt.plot([], c='#0481FD', ls='--', label='Deepsets DQN (Inequality)')
    plt.plot([], c='#E897E8', ls='dotted', label='CPU-Greedy')
    plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 30)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Deployment Cost (in units)")
    plt.legend()

    plt.savefig('plot_per_cluster_box_plot_cost.pdf', dpi=250, bbox_inches='tight')

