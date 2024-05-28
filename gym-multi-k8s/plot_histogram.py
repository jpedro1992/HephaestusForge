import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

if __name__ == "__main__":
    reward = 'multi'  # cost, risk or latency
    # test = '/lat_0.6_cost_0.2_gini_0.2'
    path = "results/karmada/v3/"
    path_model = "_env_karmada_num_clusters_4_reward_"
    testing_path = "testing/"  # "testing/"
    alg_ppo = 'ppo'  # ppo
    alg_dqn = 'dqn'  # dqn

    file_results = "vec_karmada_gym_results_monitor.csv"
    file_results_testing = "0_karmada_gym_results_num_clusters_4.csv"

    window = 200
    max_reward = 100

    # Training
    file_ppo_latency = path + reward + "/latency/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_cost = path + reward + "/cost/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_ppo_inequality = path + reward + "/inequality/" + alg_ppo + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    file_dqn_latency = path + reward + "/latency/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_cost = path + reward + "/cost/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing
    file_dqn_inequality = path + reward + "/inequality/" + alg_dqn + "_deepsets" + path_model + reward + "_totalSteps_200000_run_1/" + testing_path + file_results_testing

    df_ppo_latency = pd.read_csv(file_ppo_latency)
    df_ppo_cost = pd.read_csv(file_ppo_cost)
    df_ppo_inequality = pd.read_csv(file_ppo_inequality)

    df_dqn_cost = pd.read_csv(file_dqn_cost)
    df_dqn_latency = pd.read_csv(file_dqn_latency)
    df_dqn_inequality = pd.read_csv(file_dqn_inequality)

    # Plot Histogram with columns ep_deploy_all, ep_ffd, ep_bfd, ep_bf1b1, ep_nf1b1 during testing
    # Place all dfs in the same figure
    # How many times does each occur during the 2000 episodes?
    # Y: number of times it occurs
    # X: ep_deploy_all, ep_ffd, ep_bfd, ep_bf1b1, ep_nf1b1

    data_ppo_cost = [df_ppo_cost['ep_rejected_requests'].mean(), df_ppo_cost['ep_deploy_all'].mean(), df_ppo_cost['ep_ffd'].mean(),
                    df_ppo_cost['ep_ffi'].mean(), df_ppo_cost['ep_bf1b1'].mean()]

    data_ppo_latency = [df_ppo_latency['ep_rejected_requests'].mean(), df_ppo_latency['ep_deploy_all'].mean(), df_ppo_latency['ep_ffd'].mean(),
                        df_ppo_latency['ep_ffi'].mean(), df_ppo_latency['ep_bf1b1'].mean()] #

    data_ppo_inequality = [df_ppo_inequality['ep_rejected_requests'].mean(), df_ppo_inequality['ep_deploy_all'].mean(), df_ppo_inequality['ep_ffd'].mean(),
                           df_ppo_inequality['ep_ffi'].mean(), df_ppo_inequality['ep_bf1b1'].mean()] # df_ppo_inequality['ep_ffi'].mean(),

    data_dqn_cost = [df_dqn_cost['ep_rejected_requests'].mean(), df_dqn_cost['ep_deploy_all'].mean(), df_dqn_cost['ep_ffd'].mean(),
                     df_dqn_cost['ep_ffi'].mean(), df_dqn_cost['ep_bf1b1'].mean()] # df_dqn_cost['ep_ffi'].mean(),

    data_dqn_latency = [df_dqn_latency['ep_rejected_requests'].mean(), df_dqn_latency['ep_deploy_all'].mean(), df_dqn_latency['ep_ffd'].mean(),
                        df_dqn_latency['ep_ffi'].mean(), df_dqn_latency['ep_bf1b1'].mean()] # df_dqn_latency['ep_ffi'].mean(),

    data_dqn_inequality = [df_dqn_inequality['ep_rejected_requests'].mean(), df_dqn_inequality['ep_deploy_all'].mean(), df_dqn_inequality['ep_ffd'].mean(),
                           df_dqn_inequality['ep_ffi'].mean(), df_dqn_inequality['ep_bf1b1'].mean()] # df_dqn_inequality['ep_ffi'].mean(),

    print(data_ppo_cost)
    print(data_ppo_latency)
    print(data_ppo_inequality)
    print(data_dqn_cost)
    print(data_dqn_latency)
    print(data_dqn_inequality)

    ci_data_ppo_cost = [1.96 * np.std(df_ppo_cost["ep_rejected_requests"]) / np.sqrt(len(df_ppo_cost["ep_rejected_requests"])),
                        1.96 * np.std(df_ppo_cost["ep_deploy_all"]) / np.sqrt(len(df_ppo_cost["ep_deploy_all"])),
                        1.96 * np.std(df_ppo_cost["ep_ffd"]) / np.sqrt(len(df_ppo_cost["ep_ffd"])),
                        1.96 * np.std(df_ppo_cost["ep_ffi"]) / np.sqrt(len(df_ppo_cost["ep_ffi"])),
                        1.96 * np.std(df_ppo_cost["ep_bf1b1"]) / np.sqrt(len(df_ppo_cost["ep_bf1b1"])),
                        # 1.96 * np.std(df_ppo_cost["ep_nf1b1"]) / np.sqrt(len(df_ppo_cost["ep_nf1b1"]))
                        ]

    ci_data_ppo_latency = [1.96 * np.std(df_ppo_cost["ep_rejected_requests"]) / np.sqrt(len(df_ppo_cost["ep_rejected_requests"])),
        100 * 1.96 * np.std(df_ppo_latency["ep_deploy_all"]) / np.sqrt(len(df_ppo_latency["ep_deploy_all"])),
                            1.96 * np.std(df_ppo_latency["ep_ffd"]) / np.sqrt(len(df_ppo_latency["ep_ffd"])),
                            1.96 * np.std(df_ppo_latency["ep_ffi"]) / np.sqrt(len(df_ppo_latency["ep_ffi"])),
                            1.96 * np.std(df_ppo_latency["ep_bf1b1"]) / np.sqrt(len(df_ppo_latency["ep_bf1b1"])),
                            # 1.96 * np.std(df_ppo_latency["ep_nf1b1"]) / np.sqrt(len(df_ppo_latency["ep_nf1b1"]))
                           ]

    ci_data_ppo_inequality = [1.96 * np.std(df_ppo_cost["ep_rejected_requests"]) / np.sqrt(len(df_ppo_cost["ep_rejected_requests"])),
        100 * 1.96 * np.std(df_ppo_inequality["ep_deploy_all"]) / np.sqrt(len(df_ppo_inequality["ep_deploy_all"])),
                                1.96 * np.std(df_ppo_inequality["ep_ffd"]) / np.sqrt(len(df_ppo_inequality["ep_ffd"])),
                                1.96 * np.std(df_ppo_inequality["ep_ffi"]) / np.sqrt(len(df_ppo_inequality["ep_ffi"])),
                                1.96 * np.std(df_ppo_inequality["ep_bf1b1"]) / np.sqrt(len(df_ppo_inequality["ep_bf1b1"])),
                                # 1.96 * np.std(df_ppo_inequality["ep_nf1b1"]) / np.sqrt(len(df_ppo_inequality["ep_nf1b1"]))
                              ]

    ci_data_dqn_cost = [1.96 * np.std(df_ppo_cost["ep_rejected_requests"]) / np.sqrt(len(df_ppo_cost["ep_rejected_requests"])),
        100 * 1.96 * np.std(df_dqn_cost["ep_deploy_all"]) / np.sqrt(len(df_dqn_cost["ep_deploy_all"])),
                        1.96 * np.std(df_dqn_cost["ep_ffd"]) / np.sqrt(len(df_dqn_cost["ep_ffd"])),
                        1.96 * np.std(df_dqn_cost["ep_ffi"]) / np.sqrt(len(df_dqn_cost["ep_ffi"])),
                        1.96 * np.std(df_dqn_cost["ep_bf1b1"]) / np.sqrt(len(df_dqn_cost["ep_bf1b1"])),
                        # 1.96 * np.std(df_dqn_cost["ep_nf1b1"]) / np.sqrt(len(df_dqn_cost["ep_nf1b1"]))
                        ]

    ci_data_dqn_latency = [1.96 * np.std(df_ppo_cost["ep_rejected_requests"]) / np.sqrt(len(df_ppo_cost["ep_rejected_requests"])),
        100 * 1.96 * np.std(df_dqn_latency["ep_deploy_all"]) / np.sqrt(len(df_dqn_latency["ep_deploy_all"])),
                            1.96 * np.std(df_dqn_latency["ep_ffd"]) / np.sqrt(len(df_dqn_latency["ep_ffd"])),
                            1.96 * np.std(df_dqn_latency["ep_ffi"]) / np.sqrt(len(df_dqn_latency["ep_ffi"])),
                            1.96 * np.std(df_dqn_latency["ep_bf1b1"]) / np.sqrt(len(df_dqn_latency["ep_bf1b1"])),
                            # 1.96 * np.std(df_dqn_latency["ep_nf1b1"]) / np.sqrt(len(df_dqn_latency["ep_nf1b1"]))
                           ]

    ci_data_dqn_inequality = [1.96 * np.std(df_ppo_cost["ep_rejected_requests"]) / np.sqrt(len(df_ppo_cost["ep_rejected_requests"])),
        100 * 1.96 * np.std(df_dqn_inequality["ep_deploy_all"]) / np.sqrt(len(df_dqn_inequality["ep_deploy_all"])),
                                1.96 * np.std(df_dqn_inequality["ep_ffd"]) / np.sqrt(len(df_dqn_inequality["ep_ffd"])),
                                1.96 * np.std(df_dqn_inequality["ep_ffi"]) / np.sqrt(len(df_dqn_inequality["ep_ffi"])),
                                1.96 * np.std(df_dqn_inequality["ep_bf1b1"]) / np.sqrt(len(df_dqn_inequality["ep_bf1b1"])),
                                # 1.96 * np.std(df_dqn_inequality["ep_nf1b1"]) / np.sqrt(len(df_dqn_inequality["ep_nf1b1"]))
                              ]

    print(ci_data_ppo_cost)
    # Plotting
    xticks = ["Rejected", "Deploy_all", "FFD", "FFI", "BF1B1"]
    r = np.arange(len(xticks)) * 6
    print(r)
    # [0, 7, 14, 21, 28, 35]
    width = 0.35

    # Plotting PPO bars

    fig = plt.figure()
    plt.bar(r - width - width, data_ppo_cost, width=width, label='DS-PPO (Cost)', capsize=4) # yerr=ci_data_ppo_cost
    plt.bar(r - width, data_ppo_latency, width=width, label='DS-PPO (Latency)', capsize=4) # yerr=ci_data_ppo_latency,
    plt.bar(r, data_ppo_inequality, width=width, label='DS-PPO (Inequality)', capsize=4) # yerr=ci_data_ppo_inequality
    plt.bar(r + width, data_dqn_cost, width=width, label='DS-DQN (Cost)', capsize=4)  # yerr=ci_data_dqn_cost,
    plt.bar(r + width + width, data_dqn_latency, width=width, label='DS-DQN (Latency)', capsize=4)  # yerr=ci_data_dqn_latency,
    plt.bar(r + width + width + width, data_dqn_inequality, width=width, label='DS-DQN (Inequality)',capsize=4)  # yerr=ci_data_dqn_inequality

    plt.xlabel("Potential Actions during an Episode", fontsize=14)
    plt.ylabel("Number of Occurrences", fontsize=14)

    # show and save figure
    plt.legend()
    plt.tight_layout()
    plt.xticks(r, xticks)

    plt.savefig('actions_v3.pdf', dpi=250, bbox_inches='tight')
    plt.close()

    """
    fig = plt.figure()
    
    plt.xlabel("Number of Selected Actions during an Episode", fontsize=14)
    plt.ylabel("Potential Actions", fontsize=14)

    plt.xticks(r, xticks)

    # show and save figure
    plt.legend()
    plt.tight_layout()

    plt.savefig('dqn_actions.pdf', dpi=250, bbox_inches='tight')
    plt.close()
    """