import csv
from dataclasses import dataclass

import gym
import numpy as np
import numpy.typing as npt

# DeploymentRequest Info
from matplotlib import pyplot as plt
from numpy import mean


@dataclass
class DeploymentRequest:
    name: str
    num_replicas: int
    cpu_request: float
    cpu_limit: float  # limits can be left out
    memory_request: float
    memory_limit: float
    arrival_time: float
    latency_threshold: int  # Latency threshold that should be respected
    departure_time: float
    deployed_cluster: int = None  # All replicas deployed in one cluster
    is_deployment_split: bool = False  # has the deployment request been split?
    split_clusters: [] = None  # what is the distribution of the deployment request?
    expected_latency: int = None  # expected latency after deployment
    expected_cost: int = None  # expected cost after deployment


# NOT used by Karmada Scheduling env!
# Request Info for Fog env
@dataclass
class Request:
    cpu: float
    ram: float
    disk: float
    load: float
    arrival_time: float
    departure_time: float
    latency: int
    serving_node: int = None
    service_type: int = None  # currently: 0==SVE 1==SDP 2==APP 3==LAF


# Reverses a dict
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=reverse))


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

def get_c2e_deployment_list():
    deployment_list = [
        # 1 adapter-amqp
        DeploymentRequest(name="adapter-amqp", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=350),
        # 2 adapter-http
        DeploymentRequest(name="adapter-http", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=350),
        # 3 adapter-mqtt
        DeploymentRequest(name="adapter-mqtt", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.3, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=350),
        # 4 artemis
        DeploymentRequest(name="artemis", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.6, memory_limit=0.6,
                          arrival_time=0, departure_time=0,
                          latency_threshold=350),

        # 5 dispatch-router
        DeploymentRequest(name="dispatch-router", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.64, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 6 ditto-connectivity
        DeploymentRequest(name="ditto-connectivity", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.75, memory_limit=1.0,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 7 ditto-gateway
        DeploymentRequest(name="ditto-gateway", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 8 ditto-nginx
        DeploymentRequest(name="ditto-nginx", num_replicas=1,
                          cpu_request=0.05, cpu_limit=0.15,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=450),

        # 9 ditto-policies
        DeploymentRequest(name="ditto-policies", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=400),

        # 10 ditto-swagger-ui
        DeploymentRequest(name="ditto-swagger-ui", num_replicas=1,
                          cpu_request=0.05, cpu_limit=0.1,
                          memory_request=0.016, memory_limit=0.032,
                          arrival_time=0, departure_time=0,
                          latency_threshold=450),

        # 11 ditto-things
        DeploymentRequest(name="ditto-things", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=500),

        # 12 ditto-things-search
        DeploymentRequest(name="ditto-things-search", num_replicas=1,
                          cpu_request=0.2, cpu_limit=2.0,
                          memory_request=0.5, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=500),

        # 13 ditto-mongo-db
        DeploymentRequest(name="ditto-mongo-db", num_replicas=1,
                          cpu_request=0.015, cpu_limit=2.0,
                          memory_request=0.25, memory_limit=0.7,
                          arrival_time=0, departure_time=0,
                          latency_threshold=500),

        # 14 service-auth
        DeploymentRequest(name="service-auth", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.25,
                          arrival_time=0, departure_time=0,
                          latency_threshold=600),

        # 15 service-command-router
        DeploymentRequest(name="service-command-router", num_replicas=1,
                          cpu_request=0.015, cpu_limit=1.0,
                          memory_request=0.25, memory_limit=0.5,
                          arrival_time=0, departure_time=0,
                          latency_threshold=600),

        # 16 service-device-registry
        DeploymentRequest(name="service-device-registry", num_replicas=1,
                          cpu_request=0.2, cpu_limit=1.0,
                          memory_request=0.2, memory_limit=0.4,
                          arrival_time=0, departure_time=0,
                          latency_threshold=600),
    ]
    return deployment_list


def latency_greedy_policy(actions: int, action_mask: npt.NDArray, lat_val: npt.NDArray, lat_threshold: float) -> int:
    """Returns the index of a feasible node with latency < lat val."""
    # Remove Last Five Actions
    cluster_mask = np.logical_and(action_mask[:-actions], lat_val <= lat_threshold)
    feasible_clusters = np.argwhere(cluster_mask == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    if len(feasible_clusters) == 0:
        return len(action_mask) - 1

    return np.random.choice(feasible_clusters)
    # return feasible_clusters[np.argmin(lat_val[feasible_clusters])]


def cost_greedy_policy(actions: int, env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest deployment cost"""
    feasible_clusters = np.argwhere(action_mask[:-actions] == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    mean_cost = []
    for c in feasible_clusters:
        type_id = env.cluster_type[c]
        mean_cost.append(env.default_cluster_types[type_id]['cost'])

    if len(feasible_clusters) == 0:
        return len(action_mask) - 1
    # print("Return: {}".format(feasible_clusters[np.argmin(mean_load)]))
    return feasible_clusters[np.argmin(mean_cost)]


def cpu_greedy_policy(actions: int, env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the lowest CPU usage"""
    feasible_clusters = np.argwhere(action_mask[:-actions] == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    mean_load = []
    # Calculate percentage of allocation for CPU
    for c in feasible_clusters:
        cpu = env.allocated_cpu[c] / env.cpu_capacity[c]
        mean_load.append(cpu)

    if len(feasible_clusters) == 0:
        return len(action_mask) - 1

    # print("Return: {}".format(feasible_clusters[np.argmin(mean_load)]))
    return feasible_clusters[np.argmin(mean_load)]


def binpack_greedy_policy(actions: int, env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the highest CPU usage"""
    feasible_clusters = np.argwhere(action_mask[:-actions] == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    mean_load = []
    # Calculate percentage of allocation for CPU and Memory for feasible clusters
    for c in feasible_clusters:
        cpu = env.allocated_cpu[c] / env.cpu_capacity[c]
        mean_load.append(cpu)

    if len(feasible_clusters) == 0:
        return len(action_mask) - 1

    # print("Return: {}".format(feasible_clusters[np.argmax(mean_load)]))
    return feasible_clusters[np.argmax(mean_load)]


def karmada_greedy_policy(actions: int, env: gym.Env, action_mask: npt.NDArray) -> int:
    """Returns the index of the feasible node with the highest replicas possible based on the Karmada Scheduling algorithm"""
    feasible_clusters = np.argwhere(action_mask[:-actions] == True).flatten()
    # print("Feasible clusters: {}".format(feasible_clusters))

    mean_load = []
    # Calculate percentage of allocation for CPU and Memory for feasible clusters
    for c in feasible_clusters:
        rep_cpu = (env.cpu_capacity[c] - env.allocated_cpu[c]) / env.deployment_request.cpu_request
        rep_mem = (env.memory_capacity[c] - env.allocated_memory[c]) / env.deployment_request.memory_request
        avg = (rep_cpu + rep_mem) / 2
        mean_load.append(avg)

    # print(mean_load)
    if len(feasible_clusters) == 0:
        return len(action_mask) - 1

    # print("Return: {}".format(feasible_clusters[np.argmax(mean_load)]))
    return feasible_clusters[np.argmax(mean_load)]


# TODO: modify function
'''
def save_obs_to_csv(file_name, timestamp, num_pods, desired_replicas, cpu_usage, mem_usage,
                    traffic_in, traffic_out, latency, lstm_1_step, lstm_5_step):
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='') # new
    with file:
        fields = ['date', 'num_pods', 'cpu', 'mem', 'desired_replicas',
                  'traffic_in', 'traffic_out', 'latency', 'lstm_1_step', 'lstm_5_step']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()  # write header
        writer.writerow(
            {'date': timestamp,
             'num_pods': int("{}".format(num_pods)),
             'cpu': int("{}".format(cpu_usage)),
             'mem': int("{}".format(mem_usage)),
             'desired_replicas': int("{}".format(desired_replicas)),
             'traffic_in': int("{}".format(traffic_in)),
             'traffic_out': int("{}".format(traffic_out)),
             'latency': float("{:.3f}".format(latency)),
             'lstm_1_step': int("{}".format(lstm_1_step)),
             'lstm_5_step': int("{}".format(lstm_5_step))}
        )
'''


def save_to_csv(file_name, episode, reward, ep_block_prob, ep_accepted_requests, ep_rejected_requests, ep_deploy_all, ffd, ffi, bf1b1, avg_latency,
                avg_cost, avg_cpu_cluster_selected, gini, execution_time): # bf1b1, nf1b1
    file = open(file_name, 'a+', newline='')  # append
    # file = open(file_name, 'w', newline='')
    with file:
        fields = ['episode', 'reward', 'ep_block_prob', 'ep_accepted_requests', 'ep_rejected_requests', 'ep_deploy_all',
                  'ep_ffd', 'ep_ffi', 'ep_bf1b1', 'avg_latency', 'avg_cost',
                  'avg_cpu_cluster_selected', 'gini', 'execution_time'] # 'ep_bf1b1', 'ep_ffi', 'ep_nf1b1'
        writer = csv.DictWriter(file, fieldnames=fields)
        # writer.writeheader()
        writer.writerow(
            {'episode': episode,
             'reward': float("{:.2f}".format(reward)),
             'ep_block_prob': float("{:.2f}".format(ep_block_prob)),
             'ep_accepted_requests': float("{:.2f}".format(ep_accepted_requests)),
             'ep_rejected_requests': float("{:.2f}".format(ep_rejected_requests)),
             'ep_deploy_all': float("{:.2f}".format(ep_deploy_all)),
             'ep_ffd': float("{:.2f}".format(ffd)),
             'ep_ffi': float("{:.2f}".format(ffi)),
             'ep_bf1b1': float("{:.2f}".format(bf1b1)),
             # 'ep_nf1b1': float("{:.2f}".format(nf1b1)),
             'avg_latency': float("{:.2f}".format(avg_latency)),
             'avg_cost': float("{:.2f}".format(avg_cost)),
             'avg_cpu_cluster_selected': float("{:.2f}".format(avg_cpu_cluster_selected)),
             'gini': float("{:.2f}".format(gini)),
             'execution_time': float("{:.2f}".format(execution_time))}
        )


def normalize(value, min_value, max_value):
    if max_value == min_value:
        return 0.0  # Avoid division by zero if min_value equals max_value
    return (value - min_value) / (max_value - min_value)


# Calculation of Gini Coefficient
# 0 is better - 1 is worse!
def calculate_gini_coefficient(loads):
    n = len(loads)
    total_load = sum(loads)
    mean_load = total_load / n if n != 0 else 0

    if mean_load == 0:
        return 0  # Handle the case where all loads are zero to avoid division by zero

    gini_numerator = sum(abs(loads[i] - loads[j]) for i in range(n) for j in range(n))
    gini_coefficient = gini_numerator / (2 * n ** 2 * mean_load)

    return gini_coefficient
