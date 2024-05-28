import logging
from collections import namedtuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.use('TkAgg')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


if __name__ == "__main__":
    reward = 'multi'  # cost, risk or latency
    # test = '/lat_0.6_cost_0.2_gini_0.2'
    path_v1 = "results/karmada/v1/"
    path_v2 = "results/karmada/v2/"
    path_v3 = "results/karmada/v3/"
    path_v4 = "results/karmada/v4/"

    path_model = "_env_karmada_num_clusters_4_reward_"
    testing_path = "" #"testing/"  # "testing/"
    alg_ppo = 'ppo'  # ppo
    alg_dqn = 'dqn'  # dqn
    file_results = "vec_karmada_gym_results_monitor.csv"
    file_results_testing = "0_karmada_gym_results_num_clusters_4.csv"

    variable = 'reward'
    ylim = 100
    smoothing_window = 50

    # Training
    file_ppo_latency_v1 = path_v1 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_ppo_cost_v1 = path_v1 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_ppo_inequality_v1 = path_v1 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results

    file_ppo_latency_v2 = path_v2 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_ppo_cost_v2 = path_v2 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_ppo_inequality_v2 = path_v2 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results

    file_ppo_latency_v3 = path_v3 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_ppo_cost_v3 = path_v3 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_ppo_inequality_v3 = path_v3 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results

    file_ppo_latency_v4 = path_v4 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_300000_run_1/" + testing_path + file_results
    file_ppo_cost_v4 = path_v4 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_300000_run_1/" + testing_path + file_results
    file_ppo_inequality_v4 = path_v4 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_300000_run_1/" + testing_path + file_results

    file_dqn_latency_v1 = path_v1 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_dqn_cost_v1 = path_v1 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_dqn_inequality_v1 = path_v1 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results

    file_dqn_latency_v2 = path_v2 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_dqn_cost_v2 = path_v2 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_dqn_inequality_v2 = path_v2 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results

    file_dqn_latency_v3 = path_v3 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_dqn_cost_v3 = path_v3 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results
    file_dqn_inequality_v3 = path_v3 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results

    file_dqn_latency_v4 = path_v4 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_300000_run_1/" + testing_path + file_results
    file_dqn_cost_v4 = path_v4 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_300000_run_1/" + testing_path + file_results
    file_dqn_inequality_v4 = path_v4 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_300000_run_1/" + testing_path + file_results

    df_ppo_latency_v1 = pd.read_csv(file_ppo_latency_v1)
    df_ppo_cost_v1 = pd.read_csv(file_ppo_cost_v1)
    df_ppo_inequality_v1 = pd.read_csv(file_ppo_inequality_v1)

    df_ppo_latency_v2 = pd.read_csv(file_ppo_latency_v2)
    df_ppo_cost_v2 = pd.read_csv(file_ppo_cost_v2)
    df_ppo_inequality_v2 = pd.read_csv(file_ppo_inequality_v2)

    df_ppo_latency_v3 = pd.read_csv(file_ppo_latency_v3)
    df_ppo_cost_v3 = pd.read_csv(file_ppo_cost_v3)
    df_ppo_inequality_v3 = pd.read_csv(file_ppo_inequality_v3)

    df_ppo_latency_v4 = pd.read_csv(file_ppo_latency_v4)
    df_ppo_cost_v4 = pd.read_csv(file_ppo_cost_v4)
    df_ppo_inequality_v4 = pd.read_csv(file_ppo_inequality_v4)

    df_dqn_cost_v1 = pd.read_csv(file_dqn_cost_v1)
    df_dqn_latency_v1 = pd.read_csv(file_dqn_latency_v1)
    df_dqn_inequality_v1 = pd.read_csv(file_dqn_inequality_v1)

    df_dqn_cost_v2 = pd.read_csv(file_dqn_cost_v2)
    df_dqn_latency_v2 = pd.read_csv(file_dqn_latency_v2)
    df_dqn_inequality_v2 = pd.read_csv(file_dqn_inequality_v2)

    df_dqn_cost_v3 = pd.read_csv(file_dqn_cost_v3)
    df_dqn_latency_v3 = pd.read_csv(file_dqn_latency_v3)
    df_dqn_inequality_v3 = pd.read_csv(file_dqn_inequality_v3)

    df_dqn_cost_v4 = pd.read_csv(file_dqn_cost_v4)
    df_dqn_latency_v4 = pd.read_csv(file_dqn_latency_v4)
    df_dqn_inequality_v4 = pd.read_csv(file_dqn_inequality_v4)

    # Plotting
    fig = plt.figure()

    data_ppo_latency_v1 = pd.Series(df_ppo_latency_v1[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_cost_v1 = pd.Series(df_ppo_cost_v1[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_inequality_v1 = pd.Series(df_ppo_inequality_v1[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_dqn_cost_v1 = pd.Series(df_dqn_cost_v1[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_latency_v1 = pd.Series(df_dqn_latency_v1[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_inequality_v1 = pd.Series(df_dqn_inequality_v1[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_ppo_latency_v2 = pd.Series(df_ppo_latency_v2[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_cost_v2 = pd.Series(df_ppo_cost_v2[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_inequality_v2 = pd.Series(df_ppo_inequality_v2[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_dqn_cost_v2 = pd.Series(df_dqn_cost_v2[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_latency_v2 = pd.Series(df_dqn_latency_v2[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_inequality_v2 = pd.Series(df_dqn_inequality_v2[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_ppo_latency_v3 = pd.Series(df_ppo_latency_v3[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_cost_v3 = pd.Series(df_ppo_cost_v3[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_inequality_v3 = pd.Series(df_ppo_inequality_v3[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_dqn_cost_v3 = pd.Series(df_dqn_cost_v3[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_latency_v3 = pd.Series(df_dqn_latency_v3[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_inequality_v3 = pd.Series(df_dqn_inequality_v3[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_ppo_latency_v4 = pd.Series(df_ppo_latency_v4[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_cost_v4 = pd.Series(df_ppo_cost_v4[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_ppo_inequality_v4 = pd.Series(df_ppo_inequality_v4[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    data_dqn_cost_v4 = pd.Series(df_dqn_cost_v4[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_latency_v4 = pd.Series(df_dqn_latency_v4[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()
    data_dqn_inequality_v4 = pd.Series(df_dqn_inequality_v4[variable]).rolling(smoothing_window, min_periods=smoothing_window).mean()

    # Plotting
    fig = plt.figure()
    #plt.plot(data_ppo_latency_v4,linestyle=None, color='#77AC30', label='PPO (Latency) - V1')
    #plt.plot(data_ppo_cost_v4, linestyle=None, color='#D95319', label='PPO (Cost) - V2')
    #plt.plot(data_ppo_inequality_v4, linestyle=None, color='#3399FF', label='PPO (Inequality) - V3')
    plt.plot(data_dqn_cost_v4, linestyle=None, color='#77AC30', label='DQN (Cost) - V4')
    plt.plot(data_dqn_latency_v4, linestyle=None, color='#D95319', label='DQN (Latency) - V4')
    plt.plot(data_dqn_inequality_v4, linestyle=None, color='#3399FF', label='DQN (Inequality) - V4')

    plt.xlabel("Episode")
    plt.ylabel(variable)
    plt.xlim(smoothing_window, 3000)
    # plt.ylim(-50, 101)
    plt.legend(fontsize=10)

    plt.savefig(variable + '_v4.pdf', dpi=250, bbox_inches='tight')

