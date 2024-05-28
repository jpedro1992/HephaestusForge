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
    testing_path = "testing/"  # "testing/"
    alg_ppo = 'ppo'  # ppo
    alg_dqn = 'dqn'  # dqn
    file_results = "vec_karmada_gym_results_monitor.csv"
    file_results_testing = "0_karmada_gym_results_num_clusters_4.csv"

    variable = 'reward'
    ylim = 100

    # Training
    file_ppo_latency_v1 = path_v1 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_cost_v1 = path_v1 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_inequality_v1 = path_v1 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_ppo_latency_v2 = path_v2 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_cost_v2 = path_v2 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_inequality_v2 = path_v2 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_ppo_latency_v3 = path_v3 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_cost_v3 = path_v3 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_inequality_v3 = path_v3 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_ppo_latency_v4 = path_v4 + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_cost_v4 = path_v4 + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_inequality_v4 = path_v4 + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_dqn_latency_v1 = path_v1 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_cost_v1 = path_v1 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_inequality_v1 = path_v1 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_dqn_latency_v2 = path_v2 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_cost_v2 = path_v2 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_inequality_v2 = path_v2 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_dqn_latency_v3 = path_v3 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_cost_v3 = path_v3 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_inequality_v3 = path_v3 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_dqn_latency_v4 = path_v4 + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_cost_v4 = path_v4 + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_inequality_v4 = path_v4 + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

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
    width = 0.4
    red_square = dict(markerfacecolor='r', marker='s')

    ticks = ['v1', 'v2', 'v3', 'v4']
    '''
    ['PPO (Cost) V1', 'PPO (Latency) V1', 'PPO (Inequality) V1',
    'DQN (Cost) V1', 'DQN (Latency) V1', 'DQN (Inequality)V1 ',
    'PPO (Cost) V2', 'PPO (Latency) V2', 'PPO (Inequality) V2',
    'DQN (Cost) V2', 'DQN (Latency) V2', 'DQN (Inequality)V2 ',
    'PPO (Cost) V3', 'PPO (Latency) V3', 'PPO (Inequality) V3',
    'DQN (Cost) V3', 'DQN (Latency) V3', 'DQN (Inequality)V3 ',
    ]
    '''

    data_ppo_latency_v1 = [df_ppo_latency_v1[variable].tolist()]
    data_ppo_cost_v1 = [df_ppo_cost_v1[variable].tolist()]
    data_ppo_inequality_v1 = [df_ppo_inequality_v1[variable].tolist()]

    data_dqn_cost_v1 = [df_dqn_cost_v1[variable].tolist()]
    data_dqn_latency_v1 = [df_dqn_latency_v1[variable].tolist()]
    data_dqn_inequality_v1 = [df_dqn_inequality_v1[variable].tolist()]

    data_ppo_latency_v2 = [df_ppo_latency_v2[variable].tolist()]
    data_ppo_cost_v2 = [df_ppo_cost_v2[variable].tolist()]
    data_ppo_inequality_v2 = [df_ppo_inequality_v2[variable].tolist()]

    data_dqn_cost_v2 = [df_dqn_cost_v2[variable].tolist()]
    data_dqn_latency_v2 = [df_dqn_latency_v2[variable].tolist()]
    data_dqn_inequality_v2 = [df_dqn_inequality_v2[variable].tolist()]

    data_ppo_latency_v3 = [df_ppo_latency_v3[variable].tolist()]
    data_ppo_cost_v3 = [df_ppo_cost_v3[variable].tolist()]
    data_ppo_inequality_v3 = [df_ppo_inequality_v3[variable].tolist()]

    data_dqn_cost_v3 = [df_dqn_cost_v3[variable].tolist()]
    data_dqn_latency_v3 = [df_dqn_latency_v3[variable].tolist()]
    data_dqn_inequality_v3 = [df_dqn_inequality_v3[variable].tolist()]

    data_ppo_latency_v4 = [df_ppo_latency_v4[variable].tolist()]
    data_ppo_cost_v4 = [df_ppo_cost_v4[variable].tolist()]
    data_ppo_inequality_v4 = [df_ppo_inequality_v4[variable].tolist()]

    data_dqn_cost_v4 = [df_dqn_cost_v4[variable].tolist()]
    data_dqn_latency_v4 = [df_dqn_latency_v4[variable].tolist()]
    data_dqn_inequality_v4 = [df_dqn_inequality_v4[variable].tolist()]

    a = plt.boxplot(data_ppo_latency_v1, positions=[1], widths=width, flierprops=red_square)
    b = plt.boxplot(data_ppo_cost_v1, positions=[2], widths=width, flierprops=red_square)
    c = plt.boxplot(data_ppo_inequality_v1, positions=[3], widths=width, flierprops=red_square)
    d = plt.boxplot(data_dqn_cost_v1, positions=[4], widths=width, flierprops=red_square)
    e = plt.boxplot(data_dqn_latency_v1, positions=[5], widths=width, flierprops=red_square)
    f = plt.boxplot(data_dqn_inequality_v1, positions=[6], widths=width, flierprops=red_square)

    g = plt.boxplot(data_ppo_latency_v2, positions=[10], widths=width, flierprops=red_square)
    h = plt.boxplot(data_ppo_cost_v2, positions=[11], widths=width, flierprops=red_square)
    i = plt.boxplot(data_ppo_inequality_v2, positions=[12], widths=width, flierprops=red_square)
    j = plt.boxplot(data_dqn_cost_v2, positions=[13], widths=width, flierprops=red_square)
    k = plt.boxplot(data_dqn_latency_v2, positions=[14], widths=width, flierprops=red_square)
    l = plt.boxplot(data_dqn_inequality_v2, positions=[15], widths=width, flierprops=red_square)

    m = plt.boxplot(data_ppo_latency_v3, positions=[19], widths=width, flierprops=red_square)
    n = plt.boxplot(data_ppo_cost_v3, positions=[20], widths=width, flierprops=red_square)
    o = plt.boxplot(data_ppo_inequality_v3, positions=[21], widths=width, flierprops=red_square)
    p = plt.boxplot(data_dqn_cost_v3, positions=[22], widths=width, flierprops=red_square)
    q = plt.boxplot(data_dqn_latency_v3, positions=[23], widths=width, flierprops=red_square)
    r = plt.boxplot(data_dqn_inequality_v3, positions=[24], widths=width, flierprops=red_square)

    s = plt.boxplot(data_ppo_latency_v4, positions=[28], widths=width, flierprops=red_square)
    t = plt.boxplot(data_ppo_cost_v4, positions=[29], widths=width, flierprops=red_square)
    u = plt.boxplot(data_ppo_inequality_v4, positions=[30], widths=width, flierprops=red_square)
    v = plt.boxplot(data_dqn_cost_v4, positions=[31], widths=width, flierprops=red_square)
    w = plt.boxplot(data_dqn_latency_v4, positions=[32], widths=width, flierprops=red_square)
    x = plt.boxplot(data_dqn_inequality_v4, positions=[33], widths=width, flierprops=red_square)


    set_box_color(a, '#77AC30')
    set_box_color(b, '#D95319')
    set_box_color(c, '#3399FF')
    set_box_color(d, '#94E827')
    set_box_color(e, '#F5520C')
    set_box_color(f, '#0481FD')

    set_box_color(g, '#77AC30')
    set_box_color(h, '#D95319')
    set_box_color(i, '#3399FF')
    set_box_color(j, '#94E827')
    set_box_color(k, '#F5520C')
    set_box_color(l, '#0481FD')

    set_box_color(m, '#77AC30')
    set_box_color(n, '#D95319')
    set_box_color(o, '#3399FF')
    set_box_color(p, '#94E827')
    set_box_color(q, '#F5520C')
    set_box_color(r, '#0481FD')

    set_box_color(s, '#77AC30')
    set_box_color(t, '#D95319')
    set_box_color(u, '#3399FF')
    set_box_color(v, '#94E827')
    set_box_color(w, '#F5520C')
    set_box_color(x, '#0481FD')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#77AC30', label='PPO (Latency) - V1')
    plt.plot([], c='#D95319', label='PPO (Cost) - V1')
    plt.plot([], c='#3399FF', label='PPO (Inequality) - V1')
    plt.plot([], c='#94E827', label='DQN (Cost) - V1')
    plt.plot([], c='#F5520C', label='DQN (Latency) - V1')
    plt.plot([], c='#0481FD', label='DQN (Inequality) - V1')

    plt.plot([], c='#77AC30', label='PPO (Latency) - V2')
    plt.plot([], c='#D95319', label='PPO (Cost) - V2')
    plt.plot([], c='#3399FF', label='PPO (Inequality) - V2')
    plt.plot([], c='#94E827', label='DQN (Cost) - V2')
    plt.plot([], c='#F5520C', label='DQN (Latency) - V2')
    plt.plot([], c='#0481FD', label='DQN (Inequality) - V2')

    plt.plot([], c='#77AC30', label='PPO (Latency) - V3')
    plt.plot([], c='#D95319', label='PPO (Cost) - V3')
    plt.plot([], c='#3399FF', label='PPO (Inequality) - V3')
    plt.plot([], c='#94E827', label='DQN (Cost) - V3')
    plt.plot([], c='#F5520C', label='DQN (Latency) - V3')
    plt.plot([], c='#0481FD', label='DQN (Inequality) - V3')

    plt.plot([], c='#77AC30', label='PPO (Latency) - V4')
    plt.plot([], c='#D95319', label='PPO (Cost) - V4')
    plt.plot([], c='#3399FF', label='PPO (Inequality) - V4')
    plt.plot([], c='#94E827', label='DQN (Cost) - V4')
    plt.plot([], c='#F5520C', label='DQN (Latency) - V4')
    plt.plot([], c='#0481FD', label='DQN (Inequality) - V4')

    plt.xticks([3.5, 12.5, 21.5, 30.5], ticks, fontsize=14)
    # plt.xlim(0, 80)
    # plt.ylim(0, ylim)

    plt.xlabel("Evaluated Strategies")
    plt.ylabel(variable)
    plt.legend(fontsize=3)

    plt.savefig(variable + '_box_plot.pdf', dpi=250, bbox_inches='tight')
