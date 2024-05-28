import logging
from collections import namedtuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')

stats = namedtuple("episode_stats",
                   ["ppo_cost_rewards", "ppo_latency_rewards", "ppo_inequality_rewards", "ppo_latcost_rewards",
                    "ppo_latineq_rewards", "ppo_costineq_rewards", "ppo_balanced_rewards", "ppo_favorlat_rewards",
                    # block probability
                    "ppo_cost_ep_block_prob", "ppo_latency_ep_block_prob", "ppo_inequality_ep_block_prob",
                    "ppo_latcost_ep_block_prob",
                    "ppo_latineq_ep_block_prob", "ppo_costineq_ep_block_prob", "ppo_balanced_ep_block_prob",
                    "ppo_favorlat_ep_block_prob",
                    # latency
                    "ppo_cost_latency", "ppo_latency_latency", "ppo_inequality_latency", "ppo_latcost_latency",
                    "ppo_latineq_latency", "ppo_costineq_latency", "ppo_balanced_latency", "ppo_favorlat_latency",
                    # cost
                    "ppo_cost_cost", "ppo_latency_cost", "ppo_inequality_cost", "ppo_latcost_cost", "ppo_latineq_cost",
                    "ppo_costineq_cost", "ppo_balanced_cost", "ppo_favorlat_cost",
                    ])


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_stats(figName, stats, max_reward, smoothing_window=10):
    # latency greedy: C521EE
    # resource greedy: 7A21EE

    # Plot the episode reward over time
    ppo_cost = pd.Series(stats.ppo_cost_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_rewards).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_balanced = pd.Series(stats.ppo_balanced_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_favorlat = pd.Series(stats.ppo_favorlat_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    fig = plt.figure()
    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='DS-PPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='DS-PPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='DS-PPO (Inequality)')
    plt.plot(ppo_latcost,
             linestyle='-.', color='#EDB120', label='DS-PPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='dashdot', color='#7A21EE', label='DS-PPO (LatIneq)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='DS-PPO (CostIneq)')
    plt.plot(ppo_balanced,
             linestyle='dashed', color='#D74281', label='DS-PPO (Balanced)')
    plt.plot(ppo_favorlat,
             linestyle='-.', color='#434DD7', label='DS-PPO (FavorLat)')

    # specifying horizontal line type
    plt.axhline(y=max_reward, color='black', linestyle='--', label="max reward= " + str(max_reward))
    # plt.yscale('log')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.xlim(smoothing_window, 2000)
    plt.ylim(-50, 101)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_reward.pdf', dpi=250, bbox_inches='tight')

    ppo_cost = pd.Series(stats.ppo_cost_ep_block_prob).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_ep_block_prob).rolling(smoothing_window,
                                                                           min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_ep_block_prob).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_ep_block_prob).rolling(smoothing_window,
                                                                       min_periods=smoothing_window).mean()
    ppo_balanced = pd.Series(stats.ppo_balanced_ep_block_prob).rolling(smoothing_window,
                                                                       min_periods=smoothing_window).mean()
    ppo_favorlat = pd.Series(stats.ppo_favorlat_ep_block_prob).rolling(smoothing_window,
                                                                       min_periods=smoothing_window).mean()

    fig = plt.figure()
    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='DS-PPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='DS-PPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='DS-PPO (Inequality)')

    plt.plot(ppo_latcost,
             linestyle='-.', color='#EDB120', label='DS-PPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='dashdot', color='#7A21EE', label='DS-PPO (LatIneq)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='DS-PPO (CostIneq)')
    plt.plot(ppo_balanced,
             linestyle='dashed', color='#D74281', label='DS-PPO (Balanced)')
    plt.plot(ppo_favorlat,
             linestyle='-.', color='#434DD7', label='DS-PPO (FavorLat)')

    plt.xlabel("Episode")
    plt.ylabel("Percentage of Rejected Requests")
    plt.xlim(smoothing_window, 2000)
    plt.ylim(0, 0.6)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_block_probability.pdf', dpi=250, bbox_inches='tight')

    # Avg latency
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_latency).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_latency).rolling(smoothing_window,
                                                                     min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_latency).rolling(smoothing_window,
                                                               min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    ppo_balanced = pd.Series(stats.ppo_balanced_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()
    ppo_favorlat = pd.Series(stats.ppo_favorlat_latency).rolling(smoothing_window,
                                                                 min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='DS-PPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='DS-PPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='DS-PPO (Inequality)')
    plt.plot(ppo_latcost,
             linestyle='-.', color='#EDB120', label='DS-PPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='dashdot', color='#7A21EE', label='DS-PPO (LatIneq)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='DS-PPO (CostIneq)')
    plt.plot(ppo_balanced,
             linestyle='dashed', color='#D74281', label='DS-PPO (Balanced)')
    plt.plot(ppo_favorlat,
             linestyle='-.', color='#434DD7', label='DS-PPO (FavorLat)')

    plt.xlabel("Episode")
    plt.ylabel("Avg. Latency (in ms)")
    plt.xlim(smoothing_window, 2000)
    plt.ylim(0, 700)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_latency.pdf', dpi=250, bbox_inches='tight')

    # Avg cost
    fig = plt.figure()
    ppo_cost = pd.Series(stats.ppo_cost_cost).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ppo_latency = pd.Series(stats.ppo_latency_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_inequality = pd.Series(stats.ppo_inequality_cost).rolling(smoothing_window,
                                                                  min_periods=smoothing_window).mean()
    ppo_latcost = pd.Series(stats.ppo_latcost_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_latineq = pd.Series(stats.ppo_latineq_cost).rolling(smoothing_window,
                                                            min_periods=smoothing_window).mean()
    ppo_costineq = pd.Series(stats.ppo_costineq_cost).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_balanced = pd.Series(stats.ppo_balanced_cost).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()
    ppo_favorlat = pd.Series(stats.ppo_favorlat_cost).rolling(smoothing_window,
                                                              min_periods=smoothing_window).mean()

    plt.plot(ppo_cost,
             linestyle=None, color='#77AC30', label='DS-PPO (Cost)')
    plt.plot(ppo_latency,
             linestyle='dotted', color='#D95319', label='DS-PPO (Latency)')
    plt.plot(ppo_inequality,
             linestyle='dashed', color='#3399FF', label='DS-PPO (Inequality)')
    plt.plot(ppo_latcost,
             linestyle='-.', color='#EDB120', label='DS-PPO (LatCost)')
    plt.plot(ppo_latineq,
             linestyle='dashdot', color='#7A21EE', label='DS-PPO (LatIneq)')
    plt.plot(ppo_costineq,
             linestyle='dotted', color='#C521EE', label='DS-PPO (CostIneq)')
    plt.plot(ppo_balanced,
             linestyle='dashed', color='#D74281', label='DS-PPO (Balanced)')
    plt.plot(ppo_favorlat,
             linestyle='-.', color='#434DD7', label='DS-PPO (FavorLat)')

    plt.xlabel("Episode")
    plt.ylabel("Avg. Cost (in units)")
    plt.xlim(smoothing_window, 2000)
    plt.ylim(0, 12)
    plt.legend(ncol=2)
    # plt.title("Episode Reward (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(figName + '_cost.pdf', dpi=250, bbox_inches='tight')


def remove_duplicates(df, column_name):
    modified = df.drop_duplicates(subset=[column_name])
    modified.to_csv('karmada_gym_results.csv', index=False)
    return modified


def remove_empty_lines(df):
    print(df.isnull().sum())
    # Droping the empty rows
    modified = df.dropna()
    # Saving it to the csv file
    modified.to_csv('karmada_gym_results.csv', index=False)
    return modified


def print_statistics(df, alg_name):
    print("{} reward Mean: {}".format(alg_name, np.mean(df["reward"])))
    print("{} reward Std: {}".format(alg_name, 1.96 * np.std(df["reward"]) / np.sqrt(
        len(df["reward"]))))

    print("{} rejected requests Mean: {}".format(alg_name, 100 * np.mean(df["ep_block_prob"])))
    print("{} rejected requests Std: {}".format(alg_name, 100 * 1.96 * np.std(df["ep_block_prob"]) / np.sqrt(
        len(df["ep_block_prob"]))))

    print("{} cost Mean: {}".format(alg_name, np.mean(df["avg_cost"])))
    print("{} cost Std: {}".format(alg_name, 1.96 * np.std(df["avg_cost"]) / np.sqrt(
        len(df["avg_cost"]))))

    print("{} latency Mean: {}".format(alg_name, np.mean(df["avg_latency"])))
    print("{} latency Std: {}".format(alg_name, 1.96 * np.std(df["avg_latency"]) / np.sqrt(len(df["avg_latency"]))))

    print("{} avg_cpu_cluster_selected Mean: {}".format(alg_name, np.mean(df["avg_cpu_cluster_selected"])))
    print("{} avg_cpu_cluster_selected Std: {}".format(alg_name,
                                                       1.96 * np.std(df["avg_cpu_cluster_selected"]) / np.sqrt(
                                                           len(df["avg_cpu_cluster_selected"]))))

    print("{} gini Mean: {}".format(alg_name, np.mean(df["gini"])))
    print("{} gini Std: {}".format(alg_name, 1.96 * np.std(df["gini"]) / np.sqrt(len(df["gini"]))))

    print("{} executionTime Mean: {}".format(alg_name, np.mean(df["executionTime"])))
    print("{} executionTime Std: {}".format(alg_name,
                                            1.96 * np.std(df["executionTime"]) / np.sqrt(len(df["executionTime"]))))


if __name__ == "__main__":
    reward = 'multi'  # cost, risk or latency
    # test = '/lat_0.6_cost_0.2_gini_0.2'
    path = "results/karmada/v1/"
    path_model = "_env_karmada_num_clusters_4_reward_"
    testing_path = "testing/"  # "testing/"
    alg = 'ppo'  # ppo

    file_results = "vec_karmada_gym_results_monitor.csv"
    file_results_testing = "0_karmada_gym_results_num_clusters_4.csv"

    window = 200
    max_reward = 100

    # Training
    file_latency = path + reward + "/latency/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_cost = path + reward + "/cost/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_inequality = path + reward + "/inequality/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_latcost = path + reward + "/latcost/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_latineq = path + reward + "/latineq/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_costineq = path + reward + "/costineq/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_balanced = path + reward + "/balanced/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + file_results
    file_favorlat = path + reward + "/favorlat/" + alg + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + file_results

    file_dqn_latency = path + reward + "/latency/" + "dqn_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_cost = path + reward + "/cost/" + "dqn_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_inequality = path + reward + "/inequality/" + "dqn_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_cpu_greedy = "results/karmada/baselines/cpu/karmada_gym_results.csv"
    file_binpack_greedy = "results/karmada/baselines/binpack/karmada_gym_results.csv"
    file_latency_greedy = "results/karmada/baselines/latency/karmada_gym_results.csv"
    file_karmada_greedy = "results/karmada/baselines/karmada/karmada_gym_results.csv"

    '''
    # testing
    file_a2c = "results/testing/run_1/" + reward + "/a2c/karmada_gym_results.csv"
    file_mask_ppo = "results/testing/run_1/" + reward + "/mask_ppo/karmada_gym_results.csv"
    file_deepsets_ppo = "results/testing/run_1/" + reward + "/ppo_deepsets/0_karmada_gym_results_num_clusters_4.csv"
    file_deepsets_dqn = "results/testing/run_1/" + reward + "/dqn_deepsets/0_karmada_gym_results_num_clusters_4.csv"
    '''

    df_latency = pd.read_csv(file_latency)
    df_cost = pd.read_csv(file_cost)
    df_inequality = pd.read_csv(file_inequality)
    df_latcost = pd.read_csv(file_latcost)
    df_latineq = pd.read_csv(file_latineq)
    df_costineq = pd.read_csv(file_costineq)
    df_balanced = pd.read_csv(file_balanced)
    df_favorlat = pd.read_csv(file_favorlat)

    df_dqn_cost = pd.read_csv(file_dqn_cost)
    df_dqn_latency = pd.read_csv(file_dqn_latency)
    df_dqn_inequality = pd.read_csv(file_dqn_inequality)

    df_cpu_greedy = pd.read_csv(file_cpu_greedy)
    df_binpack_greedy = pd.read_csv(file_binpack_greedy)
    df_latency_greedy = pd.read_csv(file_latency_greedy)
    df_karmada_greedy = pd.read_csv(file_karmada_greedy)

    # df_mask_ppo = pd.read_csv(file_mask_ppo)
    # df_deepsets_ppo = pd.read_csv(file_deepsets_ppo)
    # df_deepsets_dqn = pd.read_csv(file_deepsets_dqn)
    # df_greedy = pd.read_csv(file_greedy)

    # remove_empty_lines(df_a2c)
    # remove_empty_lines(df_mask_ppo)
    # remove_empty_lines(df_deepsets_ppo)
    # remove_empty_lines(df_deepsets_dqn)

    # remove_duplicates(df_deepsets_dqn, 'episode')

    stats = stats(
        ppo_cost_rewards=df_cost['reward'],
        ppo_latency_rewards=df_latency['reward'],
        ppo_inequality_rewards=df_inequality['reward'],
        ppo_latcost_rewards=df_latcost['reward'],
        ppo_latineq_rewards=df_latineq['reward'],
        ppo_costineq_rewards=df_costineq['reward'],
        ppo_balanced_rewards=df_balanced['reward'],
        ppo_favorlat_rewards=df_favorlat['reward'],

        ppo_cost_ep_block_prob=df_cost['ep_block_prob'],
        ppo_latency_ep_block_prob=df_latency['ep_block_prob'],
        ppo_inequality_ep_block_prob=df_inequality['ep_block_prob'],
        ppo_latcost_ep_block_prob=df_latcost['ep_block_prob'],
        ppo_latineq_ep_block_prob=df_latineq['ep_block_prob'],
        ppo_costineq_ep_block_prob=df_costineq['ep_block_prob'],
        ppo_balanced_ep_block_prob=df_balanced['ep_block_prob'],
        ppo_favorlat_ep_block_prob=df_favorlat['ep_block_prob'],

        ppo_cost_latency=df_cost['avg_latency'],
        ppo_latency_latency=df_latency['avg_latency'],
        ppo_inequality_latency=df_inequality['avg_latency'],
        ppo_latcost_latency=df_latcost['avg_latency'],
        ppo_latineq_latency=df_latineq['avg_latency'],
        ppo_costineq_latency=df_costineq['avg_latency'],
        ppo_balanced_latency=df_balanced['avg_latency'],
        ppo_favorlat_latency=df_favorlat['avg_latency'],

        ppo_cost_cost=df_cost['avg_cost'],
        ppo_latency_cost=df_latency['avg_cost'],
        ppo_inequality_cost=df_inequality['avg_cost'],
        ppo_latcost_cost=df_latcost['avg_cost'],
        ppo_latineq_cost=df_latineq['avg_cost'],
        ppo_costineq_cost=df_costineq['avg_cost'],
        ppo_balanced_cost=df_balanced['avg_cost'],
        ppo_favorlat_cost=df_favorlat['avg_cost'],
    )

    plot_stats("karmada_training_" + reward, stats, max_reward=max_reward, smoothing_window=window)

    print_statistics(df_cpu_greedy, "cpu_greedy")
    print_statistics(df_binpack_greedy, "binpack_greedy")
    print_statistics(df_latency_greedy, "latency_greedy")
    print_statistics(df_karmada_greedy, "karmada_greedy")

    # print_statistics(df_ppo_cost, "ppo_cost")
    # print_statistics(df_latency, "ppo_latency")
    # print_statistics(df_ppo_inequality, "ppo_inequality")
    # print_statistics(df_ppo_latcost, "ppo_latcost")
    # print_statistics(df_ppo_latineq, "ppo_latineq")
    # print_statistics(df_ppo_costineq, "ppo_costineq")
    # print_statistics(df_ppo_balanced, "ppo_balanced")
    # print_statistics(df_ppo_favorlat, "ppo_favorlat")

    fig = plt.figure()

    sns.ecdfplot(data=df_cost['avg_latency'], color='#77AC30', label='DS-PPO (Cost)')
    sns.ecdfplot(data=df_latency['avg_latency'], color='#D95319', label='DS-PPO (Latency)')
    sns.ecdfplot(data=df_inequality['avg_latency'], color='#3399FF', label='DS-PPO (Inequality)')

    sns.ecdfplot(data=df_dqn_cost['avg_latency'], color='#94E827', label='DS-DQN (Cost)')
    sns.ecdfplot(data=df_dqn_latency['avg_latency'], color='#F5520C', label='DS-DQN (Latency)')
    sns.ecdfplot(data=df_dqn_inequality['avg_latency'], color='#0481FD', label='DS-DQN (Inequality)')

    sns.ecdfplot(data=df_cpu_greedy['avg_latency'], color='#E897E8', label='CPU-Greedy')
    sns.ecdfplot(data=df_binpack_greedy['avg_latency'], color='#BCCE61', label='Binpack-Greedy')
    sns.ecdfplot(data=df_latency_greedy['avg_latency'], color='#DAB9AA', label='Latency-Greedy')
    sns.ecdfplot(data=df_karmada_greedy['avg_latency'], color='#221F1E', label='Karmada-Greedy')

    plt.xlabel("Latency (in ms)")
    plt.ylabel("Cumulative Distribution Function (CDF)")
    plt.legend()

    plt.savefig('cdf_seaborn_latency.pdf', dpi=250, bbox_inches='tight')

    fig = plt.figure()
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['PPO (Cost)', 'PPO (Latency)', 'PPO (Inequality)',
             'DQN (Cost)', 'DQN (Latency)', 'DQN (Inequality)',
             'CPU-Greedy', 'Binpack-Greedy', 'Latency-Greedy', 'Karmada-Greedy']

    data_ppo_cost = [df_cost['avg_cost'].tolist()]
    data_ppo_latency = [df_latency['avg_cost'].tolist()]
    data_ppo_inequality = [df_inequality['avg_cost'].tolist()]
    data_dqn_cost = [df_dqn_cost['avg_cost'].tolist()]
    data_dqn_latency = [df_dqn_latency['avg_cost'].tolist()]
    data_dqn_inequality = [df_dqn_inequality['avg_cost'].tolist()]
    data_cpu_greedy = [df_cpu_greedy['avg_cost'].tolist()]
    data_binpack_greedy = [df_binpack_greedy['avg_cost'].tolist()]
    data_latency_greedy = [df_latency_greedy['avg_cost'].tolist()]
    data_karmada_greedy = [df_karmada_greedy['avg_cost'].tolist()]

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
    plt.plot([], c='#77AC30', label='DS-PPO (Cost)')
    plt.plot([], c='#D95319', ls='-.', label='DS-PPO (Latency)')
    plt.plot([], c='#3399FF', ls='--', label='DS-PPO (Inequality)')
    plt.plot([], c='#94E827', ls='dotted', label='DS-DQN (Cost)')
    plt.plot([], c='#F5520C', ls='-.', label='DS-DQN (Latency)')
    plt.plot([], c='#0481FD', ls='--', label='DS-DQN (Inequality)')
    plt.plot([], c='#E897E8', ls='dotted', label='CPU-Greedy')
    plt.plot([], c='#BCCE61', ls='-.', label='Binpack-Greedy')
    plt.plot([], c='#DAB9AA', ls='--', label='Latency-Greedy')
    plt.plot([], c='#221F1E', ls='dotted', label='Karmada-Greedy')

    plt.xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45], ticks, fontsize=4)
    # plt.xlim(0, 80)
    plt.ylim(0, 24)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel("Deployment Cost (in units)")
    plt.legend(ncol=2)

    plt.savefig('box_plot_cost.pdf', dpi=250, bbox_inches='tight')
