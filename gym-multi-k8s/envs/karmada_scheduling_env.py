import csv
import math
import operator
from datetime import datetime
import heapq
import time
import random
from statistics import mean

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from envs.utils import DeploymentRequest, get_c2e_deployment_list, save_to_csv, sort_dict_by_value, \
    calculate_gini_coefficient, normalize
import logging

# MAX Number of Replicas per deployment request
MAX_REPLICAS = 8
MIN_REPLICAS = 1

# Actions - for printing purposes
ACTIONS = ["All-", "Divide", "Reject"]

# Reward objectives
# Baseline Strategy: +1 if agent accepts request,
# or -1 if rejects it (if resources were available)
NAIVE = 'naive'

# Risk-Aware: limit the risk of a cluster being full (100% allocated)
# which avoids performance degradation if pods request further resources
RISK_AWARE = 'risk'
RISK_THRESHOLD = 0.75

# BIN_PACK: most allocated clusters preferred
# avoid under utilization, opposing goal to risk aware
BINPACK = "binpack"

# Latency reward function:
LATENCY = 'latency'

# Cost reward function:
COST = 'cost'
MAX_COST = 16  # Defined based on the max cost in DEFAULT_CLUSTER_TYPES
MIN_COST = 1  # Defined based on the min cost in DEFAULT_CLUSTER_TYPES

MULTI = 'multi'

# Defaults for Weights
LATENCY_WEIGHT = 0.6
COST_WEIGHT = 0.2
GINI_WEIGHT = 0.2

# Cluster Types
NUM_CLUSTER_TYPES = 5
DEFAULT_CLUSTER_TYPES = [{"type": "edge_tier_1", "cpu": 2.0, "mem": 2.0, "cost": 1},
                         {"type": "edge_tier_2", "cpu": 2.0, "mem": 4.0, "cost": 2},
                         {"type": "fog_tier_1", "cpu": 2.0, "mem": 8.0, "cost": 4},
                         {"type": "fog_tier_2", "cpu": 4.0, "mem": 16.0, "cost": 8},
                         {"type": "cloud", "cpu": 8.0, "mem": 32.0, "cost": 16}]

# DEFAULTS for Env configuration
DEFAULT_NUM_EPISODE_STEPS = 100
DEFAULT_NUM_CLUSTERS = 4
DEFAULT_ARRIVAL_RATE = 100
DEFAULT_CALL_DURATION = 1
DEFAULT_REWARD_FUNTION = NAIVE
DEFAULT_FILE_NAME_RESULTS = "karmada_gym_results"
NUM_METRICS_CLUSTER = 4
NUM_METRICS_REQUEST = 4

# Variables for spreading strategy
NUM_SPREADING_ACTIONS = 3
FFD = 0
FFI = 1
BF1B1 = 2
# NF1B1 = 3
# BFD = 4

# Defaults for latency
MIN_DELAY = 1  # corresponds to 1ms
MAX_DELAY = 1000  # corresponds to 1000ms

SEED = 42


class KarmadaSchedulingEnv(gym.Env):
    """ Karmada Scheduling env in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, num_clusters=DEFAULT_NUM_CLUSTERS,
                 arrival_rate_r=DEFAULT_ARRIVAL_RATE,
                 call_duration_r=DEFAULT_CALL_DURATION,
                 episode_length=DEFAULT_NUM_EPISODE_STEPS,
                 reward_function=DEFAULT_REWARD_FUNTION,
                 latency_weight=LATENCY_WEIGHT,
                 cost_weight=COST_WEIGHT,
                 gini_weight=GINI_WEIGHT,
                 min_replicas=MIN_REPLICAS,
                 max_replicas=MAX_REPLICAS,
                 seed=SEED,
                 file_results_name=DEFAULT_FILE_NAME_RESULTS):

        # Define action and observation space
        super(KarmadaSchedulingEnv, self).__init__()
        self.name = "karmada_gym"
        self.__version__ = "0.0.1"
        self.reward_function = reward_function

        self.num_clusters = num_clusters
        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.episode_length = episode_length
        self.running_requests: list[DeploymentRequest] = []

        # For Request generation
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        # For Latency purposes
        self.latency_matrix = np.zeros((num_clusters, num_clusters))
        self.num_metrics_latency = 0
        self.latency = np.zeros(num_clusters)

        self.seed = seed
        self.np_random, seed = seeding.np_random(self.seed)

        logging.info(
            "[Init] Env: {} | Version {} | Num_Clusters: {} |".format(self.name, self.__version__, num_clusters))

        # Defined as a matrix having as rows the nodes and columns their associated metrics
        self.observation_space = spaces.Box(low=0.0,
                                            high=1000.0,
                                            shape=(num_clusters + NUM_SPREADING_ACTIONS + 1,
                                                   NUM_METRICS_CLUSTER + NUM_METRICS_REQUEST + 2),
                                            dtype=np.float32)

        # Default latency matrix
        for n1 in range(num_clusters):
            for n2 in range(num_clusters):
                if n1 == n2:  # for the same node assume 0
                    self.latency_matrix[n1][n2] = 0
                else:
                    self.latency_matrix[n1][n2] = self.np_random.integers(low=MIN_DELAY, high=MAX_DELAY)

            self.latency[n1] = mean(self.latency_matrix[n1])

        # logging.info("[Init] Latency Matrix: {}".format(self.latency_matrix))
        # logging.info("[Init] Latency: {}".format(self.latency))

        # Action Space
        # deploy the service on cluster 1,2,..., n + spreading actions + reject it
        self.num_actions = num_clusters + NUM_SPREADING_ACTIONS + 1

        # Discrete action space
        self.action_space = spaces.Discrete(self.num_actions)

        # Action and Observation Space
        logging.info("[Init] Action Space: {}".format(self.action_space))
        logging.info("[Init] Observation Space: {}".format(self.observation_space))
        # logging.info("[Init] Observation Space Shape: {}".format(self.observation_space.shape))

        # Setting the experiment based on Cloud2Edge (C2E) deployments
        self.deploymentList = get_c2e_deployment_list()
        self.deployment_request = None

        # New: Resource capacities based on cluster type
        self.cpu_capacity = np.zeros(num_clusters)
        self.memory_capacity = np.zeros(num_clusters)
        self.default_cluster_types = DEFAULT_CLUSTER_TYPES
        self.cluster_type = [0] * num_clusters  # np.zeros(num_clusters)

        logging.info("[Init] Resource Capacity calculation... ")
        for c in range(num_clusters):
            type = int(self.np_random.integers(low=0, high=NUM_CLUSTER_TYPES))
            self.cluster_type[c] = type
            self.cpu_capacity[c] = DEFAULT_CLUSTER_TYPES[type]['cpu']
            self.memory_capacity[c] = DEFAULT_CLUSTER_TYPES[type]['mem']
            logging.info("[Init] Cluster id: {} | Type: {} | cpu: {} | mem: {}".format(c + 1,
                                                                                       DEFAULT_CLUSTER_TYPES[type][
                                                                                           'type'],
                                                                                       self.cpu_capacity[c],
                                                                                       self.memory_capacity[c]))

        # Keeps track of allocated resources
        self.allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=num_clusters)
        self.allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=num_clusters)

        # Keeps track of Free resources for deployment requests
        self.free_cpu = np.zeros(num_clusters)
        self.free_memory = np.zeros(num_clusters)

        for n in range(num_clusters):
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

        # Variables for divide strategy
        self.split_number_replicas = np.zeros(num_clusters)
        self.calculated_split_number_replicas = np.zeros(num_clusters)

        '''
        logging.info("[Init] Resources:")
        logging.info("[Init] CPU Capacity: {}".format(self.cpu_capacity))
        logging.info("[Init] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Init] CPU free: {}".format(self.free_cpu))
        logging.info("[Init] MEM Capacity: {}".format(self.memory_capacity))
        logging.info("[Init] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Init] MEM free: {}".format(self.free_memory))
        logging.info("[Init] Cluster Types: {}".format(self.cluster_type))
        '''
        # Variables for rewards
        self.latency_weight = latency_weight
        self.cost_weight = cost_weight
        self.gini_weight = gini_weight

        # Variables for logging
        self.current_step = 0
        self.current_time = 0
        self.penalty = False
        self.accepted_requests = 0
        self.offered_requests = 0
        self.ep_accepted_requests = 0
        self.next_request()

        # Info & episode over
        self.total_reward = 0
        self.episode_over = False
        self.info = {}
        self.block_prob = 0
        self.ep_block_prob = 0
        self.avg_latency = []
        self.avg_cost = []
        self.avg_cpu_usage_percentage_cluster_selected = []
        self.avg_load_served = np.zeros(num_clusters)

        # Keep track of spreading actions
        self.deploy_all = 0
        self.deploy_ffd = 0
        self.deploy_ffi = 0
        self.deploy_bf1b1 = 0
        # self.deploy_nf1b1 = 0

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = file_results_name + ".csv"
        self.obs_csv = self.name + "_obs.csv"

    # Reset Function
    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.ep_accepted_requests = 0
        self.penalty = False

        self.block_prob = 0
        self.ep_block_prob = 0

        self.avg_latency = []
        self.avg_cost = []
        self.avg_load_served = np.zeros(self.num_clusters)
        self.avg_cpu_usage_percentage_cluster_selected = []

        # Reset Deployment Data
        self.deploymentList = get_c2e_deployment_list()

        for n1 in range(self.num_clusters):
            for n2 in range(self.num_clusters):
                if n1 == n2:  # for the same node assume 0
                    self.latency_matrix[n1][n2] = 0
                else:
                    self.latency_matrix[n1][n2] = self.np_random.integers(low=MIN_DELAY, high=MAX_DELAY)

            self.latency[n1] = mean(self.latency_matrix[n1])

        logging.info("[Reset] Resource Capacity calculation... ")
        self.cluster_type = [0] * self.num_clusters  # np.zeros(num_clusters)

        for c in range(self.num_clusters):
            type = int(self.np_random.integers(low=0, high=NUM_CLUSTER_TYPES))
            self.cluster_type[c] = type
            self.cpu_capacity[c] = DEFAULT_CLUSTER_TYPES[type]['cpu']
            self.memory_capacity[c] = DEFAULT_CLUSTER_TYPES[type]['mem']
            logging.info("[Init] Cluster id: {} | Type: {} | cpu: {} | mem: {}".format(c + 1,
                                                                                       DEFAULT_CLUSTER_TYPES[type][
                                                                                           'type'],
                                                                                       self.cpu_capacity[c],
                                                                                       self.memory_capacity[c]))

        # Keeps track of allocated resources
        self.allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=self.num_clusters)
        self.allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=self.num_clusters)

        self.free_cpu = np.zeros(self.num_clusters)
        self.free_memory = np.zeros(self.num_clusters)
        for n in range(self.num_clusters):
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

        '''
        logging.info("[Reset] Resources:")
        logging.info("[Reset] CPU Capacity: {}".format(self.cpu_capacity))
        logging.info("[Reset] CPU allocated: {}".format(self.allocated_cpu))
        logging.info("[Reset] CPU free: {}".format(self.free_cpu))
        logging.info("[Reset] MEM Capacity: {}".format(self.memory_capacity))
        logging.info("[Reset] MEM allocated: {}".format(self.allocated_memory))
        logging.info("[Reset] MEM free: {}".format(self.free_memory))
        '''

        # Keep track of spreading actions
        self.deploy_all = 0
        self.deploy_ffd = 0
        self.deploy_ffi = 0
        self.deploy_bf1b1 = 0
        # self.deploy_nf1b1 = 0

        # Variables for divide strategy
        self.split_number_replicas = np.zeros(self.num_clusters)
        self.calculated_split_number_replicas = np.zeros(self.num_clusters)

        # return obs
        return np.array(self.get_state())

    # Step function
    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        # Execute one time step within the environment
        self.offered_requests += 1
        self.take_action(action)

        # Calculate Reward
        reward = self.get_reward()
        self.total_reward += reward

        # Find correct action move for logging purposes
        move = ""
        if action < self.num_clusters:
            move = ACTIONS[0] + "cluster-" + str(action + 1)
        elif self.num_clusters <= action < self.num_clusters + NUM_SPREADING_ACTIONS:
            move = ACTIONS[1]
        else:
            move = ACTIONS[2]

        # Logging Step and Total Reward
        logging.info('[Step {}] | Action: {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, move, reward, self.total_reward))

        # Get next request
        self.next_request()

        # Update observation
        ob = self.get_state()

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.save_obs_to_csv(self.obs_csv, np.array(ob), date)

        # episode results to save
        self.block_prob = 1 - (self.accepted_requests / self.offered_requests)
        self.ep_block_prob = 1 - (self.ep_accepted_requests / self.current_step)

        if len(self.avg_latency) == 0 and len(self.avg_cost) == 0 and len(
                self.avg_cpu_usage_percentage_cluster_selected) == 0:
            avg_c = 1
            avg_l = 1
            avg_cpu = 1
        else:
            avg_c = mean(self.avg_cost)
            avg_l = mean(self.avg_latency)
            avg_cpu = mean(self.avg_cpu_usage_percentage_cluster_selected)

        self.info = {
            "reward_step": float("{:.2f}".format(reward)),
            "action": float("{:.2f}".format(action)),
            # "block_prob": float("{:.2f}".format(self.block_prob)),
            "reward": float("{:.2f}".format(self.total_reward)),
            "ep_block_prob": float("{:.2f}".format(self.ep_block_prob)),
            "ep_accepted_requests": float("{:.2f}".format(self.ep_accepted_requests)),
            "ep_rejected_requests": float("{:.2f}".format(self.episode_length - self.ep_accepted_requests)),
            "ep_deploy_all": float("{:.2f}".format(self.deploy_all)),
            "ep_ffd": float("{:.2f}".format(self.deploy_ffd)),
            "ep_ffi": float("{:.2f}".format(self.deploy_ffi)),
            "ep_bf1b1": float("{:.2f}".format(self.deploy_bf1b1)),
            # "ep_nf1b1": float("{:.2f}".format(self.deploy_nf1b1)),
            'avg_latency': float("{:.2f}".format(avg_l)),
            'avg_cost': float("{:.2f}".format(avg_c)),
            'avg_cpu_cluster_selected': float("{:.2f}".format(avg_cpu)),
            'gini': float("{:.2f}".format(calculate_gini_coefficient(self.avg_load_served))),
            'executionTime': float("{:.2f}".format(self.execution_time))
        }

        if self.current_step == self.episode_length:
            self.episode_count += 1
            self.episode_over = True
            self.execution_time = time.time() - self.time_start

            gini = calculate_gini_coefficient(self.avg_load_served)

            logging.info("[Step] Episode finished, saving results to csv...")
            save_to_csv(self.file_results, self.episode_count,
                        self.total_reward, self.ep_block_prob,
                        self.ep_accepted_requests,
                        self.episode_length - self.ep_accepted_requests,
                        self.deploy_all,
                        self.deploy_ffd,
                        self.deploy_ffi,
                        self.deploy_bf1b1,
                        # self.deploy_nf1b1,
                        mean(self.avg_latency),
                        mean(self.avg_cost),
                        mean(self.avg_cpu_usage_percentage_cluster_selected),
                        gini,
                        self.execution_time)

        # return ob, reward, self.episode_over, self.info
        return np.array(ob), reward, self.episode_over, self.info

    # TODO: Future work: design reward function based on Multi-Objective Function
    # Reward Function
    def get_reward(self):
        """ Calculate Rewards """
        if self.reward_function == NAIVE:
            if self.penalty:
                if not self.check_if_cluster_is_really_full():
                    logging.info("[NAIVE] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[NAIVE] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:
                return 1

        # Multi-Objective Reward Function
        elif self.reward_function == MULTI:
            if self.penalty:
                if not self.check_if_cluster_is_really_full():
                    logging.info("[MULTI] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[MULTI] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:  # If deployment is not split
                lat = 0
                cost = 0
                if not self.deployment_request.is_deployment_split:
                    logging.info('[MULTI] Deployment not split...')

                    # Cost
                    type_id = self.cluster_type[self.deployment_request.deployed_cluster]
                    cost = DEFAULT_CLUSTER_TYPES[type_id]['cost']

                    # Latency
                    lat = self.latency[self.deployment_request.deployed_cluster]

                else:  # If deployment is split
                    logging.info('[MULTI] Deployment split...')
                    # Cost
                    cost = self.deployment_request.expected_cost

                    # Latency
                    lat = self.deployment_request.expected_latency

                gini = calculate_gini_coefficient(self.avg_load_served)
                logging.info('[Multi Reward] Latency: {} | Cost: {} | Gini: {} |'.format(lat, cost, gini))

                lat = normalize(lat, MIN_DELAY, MAX_DELAY)
                cost = normalize(cost, MIN_COST, MAX_COST)

                reward = self.latency_weight * (1 - lat) + self.cost_weight * (1 - cost) + self.gini_weight * (1 - gini)

                logging.info(
                    '[Multi Reward] latency norm: {} | cost norm: {} | gini: {} | reward: {}'.format(lat, cost, gini,
                                                                                                     reward))
                logging.info('[Multi Reward] latency part: {} | cost part: {} | gini part: {}'.format(
                    self.latency_weight * (1 - lat), self.cost_weight * (1 - cost), self.gini_weight * (1 - gini)))
                return reward

        # Latency Reward Function
        elif self.reward_function == LATENCY:
            logging.info('[Get Reward] Latency Reward Funtion Selected...')
            if self.penalty:
                if not self.check_if_cluster_is_really_full():
                    logging.info("[Get Reward] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[Get Reward] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return 1
            else:  # If deployment is not split
                t = self.deployment_request.latency_threshold
                lat = 0
                if not self.deployment_request.is_deployment_split:
                    lat = self.latency[self.deployment_request.deployed_cluster]
                    logging.info('[Get Reward] Latency Reward All - Threshold: {} | latency: {}'.format(t, lat))

                else:  # If deployment is split
                    lat = self.deployment_request.expected_latency
                    logging.info('[Get Reward] Latency Reward Divide - Threshold: {} | latency: {}'.format(t, lat))

                if t > lat:
                    return 1
                else:
                    return -1

        # Cost-aware reward function
        elif self.reward_function == COST:
            logging.info('[Get Reward] Cost Reward Funtion Selected...')
            if self.penalty:
                if not self.check_if_cluster_is_really_full():
                    logging.info("[Get Reward] Penalty = True, and resources "
                                 "were available, penalize the agent...")
                    return -1
                else:  # agent should not be penalized
                    logging.info("[Get Reward] Penalty = True, but resources "
                                 "were not available, do not penalize the agent...")
                    return MAX_COST - MIN_COST
            else:  # If deployment is not split
                if not self.deployment_request.is_deployment_split:
                    c = self.deployment_request.deployed_cluster
                    type_id = self.cluster_type[c]
                    cost = DEFAULT_CLUSTER_TYPES[type_id]['cost']
                    logging.info('[Get Reward] Cost Reward All - type_id {} - cost: {}'.format(type_id, cost))
                else:  # If deployment is split
                    cost = self.deployment_request.expected_cost
                    logging.info('[Get Reward] Cost Reward Divide - cost: {}'.format(cost))

                return float("{:.2f}".format(MAX_COST - cost))
        else:
            logging.info('[Get Reward] Unrecognized reward: {}'.format(self.reward_function))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    # Apply the action selected by the RL agent
    def take_action(self, action):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == self.episode_length:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # Possible Actions: Place all replicas together or split them.
        # Known as NP-hard problem (Bin pack with fragmentation)
        # Any ideas for heuristic? We can later compare with an ILP/MILP model...
        # Check first if "Place all" Action can be performed
        if action < self.num_clusters:
            if self.check_if_cluster_is_full_after_full_deployment(action):
                self.penalty = True
                logging.info('[Take Action] Block the selected action since cluster will be full!')
                # Do not raise error since algorithm might not support action mask
                # raise ValueError("Action mask is not working properly. Full nodes should be always masked.")
            else:
                # accept request
                self.accepted_requests += 1
                self.ep_accepted_requests += 1
                self.deploy_all += 1
                self.deployment_request.deployed_cluster = action
                self.penalty = False
                # Update allocated amounts
                self.allocated_cpu[action] += self.deployment_request.cpu_request * self.deployment_request.num_replicas
                self.allocated_memory[
                    action] += self.deployment_request.memory_request * self.deployment_request.num_replicas

                self.avg_cpu_usage_percentage_cluster_selected.append(
                    100 * (self.allocated_cpu[action] / self.cpu_capacity[action]))
                self.avg_load_served[action] += self.deployment_request.num_replicas

                # Update free resources
                self.free_cpu[action] = self.cpu_capacity[action] - self.allocated_cpu[action]
                self.free_memory[action] = self.memory_capacity[action] - self.allocated_memory[action]
                self.enqueue_request(self.deployment_request)

                # Latency and Cost updates
                self.increase_latency(action, 1.15)  # 15% increase max
                self.avg_latency.append(self.latency[action])
                type_id = self.cluster_type[action]
                self.avg_cost.append(DEFAULT_CLUSTER_TYPES[type_id]['cost'])

                # Save expected latency and cost in deployment request
                self.deployment_request.expected_latency = self.latency[action]
                self.deployment_request.expected_cost = DEFAULT_CLUSTER_TYPES[type_id]['cost']

        # FFD increasing Strategy
        elif action == self.num_clusters + FFD:
            if self.deployment_request.num_replicas == 1:
                logging.info('[Take Action] Block FFD strategy since only one replica... ')
                self.penalty = True
            else:
                logging.info('[Take Action] FFD strategy chosen... ')
                div = self.first_fit_decreasing_heuristic(self.deployment_request.num_replicas,
                                                          self.deployment_request.cpu_request,
                                                          self.deployment_request.memory_request, self.num_clusters,
                                                          self.free_cpu, self.free_memory)

                if self.check_if_clusters_are_full_after_split_deployment(div):
                    self.penalty = True
                    logging.info('[Take Action] Block the FFD strategy since cluster will be full!')
                else:
                    # accept request
                    self.penalty = False
                    self.accepted_requests += 1
                    self.ep_accepted_requests += 1
                    self.deploy_ffd += 1
                    self.deployment_request.split_clusters = div
                    self.deployment_request.is_deployment_split = True

                    # logging.info("[Divide] Before")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))
                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))

                    avg_l = 0
                    avg_c = 0
                    avg_cpu = 0
                    clusters = 0
                    for d in range(len(div)):
                        # Update allocated amounts
                        self.allocated_cpu[d] += self.deployment_request.cpu_request * div[d]
                        self.allocated_memory[d] += self.deployment_request.memory_request * div[d]
                        avg_cpu += 100 * (self.allocated_cpu[d] / self.cpu_capacity[d])
                        # Update free resources
                        self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                        self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                        # Latency updates
                        avg_l += self.latency[d] * div[d]
                        self.increase_latency(d, 1.05)  # 5% increase max for split

                        # Cost Updates
                        type_id = int(self.cluster_type[d])
                        avg_c += DEFAULT_CLUSTER_TYPES[type_id]['cost'] * div[d]

                        # Load updates
                        self.avg_load_served[d] += div[d]

                    avg_l = avg_l / self.deployment_request.num_replicas
                    avg_c = avg_c / self.deployment_request.num_replicas
                    avg_cpu = avg_cpu / self.deployment_request.num_replicas

                    self.avg_latency.append(avg_l)
                    self.avg_cost.append(avg_c)
                    self.avg_cpu_usage_percentage_cluster_selected.append(avg_cpu)
                    '''
                    logging.info("[FFD] Average Latency: {}".format(avg_l))
                    logging.info("[FFD] Average Cost: {}".format(avg_c))
                    logging.info("[FFD] Average CPU: {}".format(avg_cpu))
                    logging.info("[FFD] Average Load: {}".format(self.avg_load_served))                 
                    '''

                    # Save expected latency and cost in deployment request
                    self.deployment_request.expected_latency = avg_l
                    self.deployment_request.expected_cost = avg_c

                    # logging.info("[Divide] After")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))

                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))
                    self.enqueue_request(self.deployment_request)

        # FFI decreasing strategy
        elif action == self.num_clusters + FFI:
            if self.deployment_request.num_replicas == 1:
                logging.info('[Take Action] Block FFI strategy since only one replica... ')
                self.penalty = True
            else:
                logging.info('[Take Action] Divide FFI chosen... ')
                div = self.first_fit_decreasing_heuristic(self.deployment_request.num_replicas,
                                                          self.deployment_request.cpu_request,
                                                          self.deployment_request.memory_request,
                                                          self.num_clusters,
                                                          self.free_cpu, self.free_memory)

                if self.check_if_clusters_are_full_after_split_deployment(div):
                    self.penalty = True
                    logging.info('[Take Action] Block the FFI strategy since cluster will be full!')
                else:
                    # accept request
                    self.penalty = False
                    self.accepted_requests += 1
                    self.ep_accepted_requests += 1
                    self.deploy_ffi += 1
                    self.deployment_request.split_clusters = div
                    self.deployment_request.is_deployment_split = True

                    avg_l = 0
                    avg_c = 0
                    avg_cpu = 0
                    clusters = 0
                    for d in range(len(div)):
                        # Update allocated amounts
                        self.allocated_cpu[d] += self.deployment_request.cpu_request * div[d]
                        self.allocated_memory[d] += self.deployment_request.memory_request * div[d]
                        avg_cpu += 100 * (self.allocated_cpu[d] / self.cpu_capacity[d])
                        # Update free resources
                        self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                        self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                        # Latency updates
                        avg_l += self.latency[d] * div[d]
                        self.increase_latency(d, 1.05)  # 5% increase max for split

                        # Cost Updates
                        type_id = int(self.cluster_type[d])
                        avg_c += DEFAULT_CLUSTER_TYPES[type_id]['cost'] * div[d]

                        # Load updates
                        self.avg_load_served[d] += div[d]

                    avg_l = avg_l / self.deployment_request.num_replicas
                    avg_c = avg_c / self.deployment_request.num_replicas
                    avg_cpu = avg_cpu / self.deployment_request.num_replicas

                    self.avg_latency.append(avg_l)
                    self.avg_cost.append(avg_c)
                    self.avg_cpu_usage_percentage_cluster_selected.append(avg_cpu)

                    # Save expected latency and cost in deployment request
                    self.deployment_request.expected_latency = avg_l
                    self.deployment_request.expected_cost = avg_c

                    # logging.info("[Divide] After")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))

                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))
                    self.enqueue_request(self.deployment_request)
        # BF1B1 spreading strategy
        elif action == self.num_clusters + BF1B1:
            if self.deployment_request.num_replicas == 1:
                logging.info('[Take Action] Block BF1B1 strategy since only one replica... ')
                self.penalty = True
            else:
                logging.info('[Take Action] BF1B1 chosen... ')
                div = self.best_fit_heuristic_one_by_one(self.deployment_request.num_replicas,
                                                         self.deployment_request.cpu_request,
                                                         self.deployment_request.memory_request,
                                                         self.num_clusters,
                                                         self.free_cpu, self.free_memory)

                if self.check_if_clusters_are_full_after_split_deployment(div):
                    self.penalty = True
                    logging.info('[Take Action] Block the BF1B1 strategy since cluster will be full!')
                else:
                    # accept request
                    self.penalty = False
                    self.accepted_requests += 1
                    self.ep_accepted_requests += 1
                    self.deploy_bf1b1 += 1
                    self.deployment_request.split_clusters = div
                    self.deployment_request.is_deployment_split = True

                    # logging.info("[Divide] Before")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))
                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))

                    avg_l = 0
                    avg_c = 0
                    avg_cpu = 0
                    clusters = 0
                    for d in range(len(div)):
                        # Update allocated amounts
                        self.allocated_cpu[d] += self.deployment_request.cpu_request * div[d]
                        self.allocated_memory[d] += self.deployment_request.memory_request * div[d]
                        avg_cpu += 100 * (self.allocated_cpu[d] / self.cpu_capacity[d])
                        # Update free resources
                        self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                        self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                        # Latency updates
                        avg_l += self.latency[d] * div[d]
                        self.increase_latency(d, 1.05)  # 5% increase max for split

                        # Cost Updates
                        type_id = int(self.cluster_type[d])
                        avg_c += DEFAULT_CLUSTER_TYPES[type_id]['cost'] * div[d]

                        # Load updates
                        self.avg_load_served[d] += div[d]

                    avg_l = avg_l / self.deployment_request.num_replicas
                    avg_c = avg_c / self.deployment_request.num_replicas
                    avg_cpu = avg_cpu / self.deployment_request.num_replicas

                    self.avg_latency.append(avg_l)
                    self.avg_cost.append(avg_c)
                    self.avg_cpu_usage_percentage_cluster_selected.append(avg_cpu)

                    # logging.info("[BF1B1] Average Latency: {}".format(avg_l))
                    # logging.info("[BF1B1] Average Cost: {}".format(avg_c))
                    # logging.info("[BF1B1] Average CPU: {}".format(avg_cpu))
                    # logging.info("[BF1B1] Average Load: {}".format(self.avg_load_served))

                    # Save expected latency and cost in deployment request
                    self.deployment_request.expected_latency = avg_l
                    self.deployment_request.expected_cost = avg_c
                    self.enqueue_request(self.deployment_request)

        # Reject the request: give the agent a penalty, especially if the request could have been accepted
        elif action == self.num_clusters + NUM_SPREADING_ACTIONS:
            self.penalty = True
        else:
            logging.info('[Take Action] Unrecognized Action: {}'.format(action))

        '''
        # BF1B1 spreading strategy
        elif action == self.num_clusters + BF1B1:
            if self.deployment_request.num_replicas == 1:
                logging.info('[Take Action] Block BF1B1 strategy since only one replica... ')
                self.penalty = True
            else:
                logging.info('[Take Action] BF1B1 chosen... ')
                div = self.best_fit_heuristic_one_by_one(self.deployment_request.num_replicas,
                                                                 self.deployment_request.cpu_request,
                                                                 self.deployment_request.memory_request,
                                                                 self.num_clusters,
                                                                 self.free_cpu, self.free_memory)

                if self.check_if_clusters_are_full_after_split_deployment(div):
                        self.penalty = True
                        logging.info('[Take Action] Block the BF1B1 strategy since cluster will be full!')
                else:
                    # accept request
                    self.penalty = False
                    self.accepted_requests += 1
                    self.ep_accepted_requests += 1
                    self.deploy_bf1b1 += 1
                    self.deployment_request.split_clusters = div
                    self.deployment_request.is_deployment_split = True

                    # logging.info("[Divide] Before")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))
                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))

                    avg_l = 0
                    avg_c = 0
                    avg_cpu = 0
                    clusters = 0
                    for d in range(len(div)):
                        # Update allocated amounts
                        self.allocated_cpu[d] += self.deployment_request.cpu_request * div[d]
                        self.allocated_memory[d] += self.deployment_request.memory_request * div[d]
                        avg_cpu += 100 * (self.allocated_cpu[d] / self.cpu_capacity[d])
                        # Update free resources
                        self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                        self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                        # Latency updates
                        avg_l += self.latency[d] * div[d]
                        self.increase_latency(d, 1.05)  # 5% increase max for split

                        # Cost Updates
                        type_id = int(self.cluster_type[d])
                        avg_c += DEFAULT_CLUSTER_TYPES[type_id]['cost'] * div[d]

                        # Load updates
                        self.avg_load_served[d] += div[d]

                    avg_l = avg_l / self.deployment_request.num_replicas
                    avg_c = avg_c / self.deployment_request.num_replicas
                    avg_cpu = avg_cpu / self.deployment_request.num_replicas

                    self.avg_latency.append(avg_l)
                    self.avg_cost.append(avg_c)
                    self.avg_cpu_usage_percentage_cluster_selected.append(avg_cpu)

                    logging.info("[BF1B1] Average Latency: {}".format(avg_l))
                    logging.info("[BF1B1] Average Cost: {}".format(avg_c))
                    logging.info("[BF1B1] Average CPU: {}".format(avg_cpu))
                    logging.info("[BF1B1] Average Load: {}".format(self.avg_load_served))

                    # Save expected latency and cost in deployment request
                    self.deployment_request.expected_latency = avg_l
                    self.deployment_request.expected_cost = avg_c

                    # logging.info("[Divide] After")
                    # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                    # logging.info("[Divide] CPU free: {}".format(self.free_cpu))

                    # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                    # logging.info("[Divide] MEM free: {}".format(self.free_memory))
                    self.enqueue_request(self.deployment_request)

                # NF1B1 spreading strategy
                elif action == self.num_clusters + NF1B1:
                    if self.deployment_request.num_replicas == 1:
                        logging.info('[Take Action] Block NF1B1 strategy since only one replica... ')
                        self.penalty = True
                    else:
                        logging.info('[Take Action] NF1B1 chosen... ')
                        div = self.best_fit_decreasing_heuristic(self.deployment_request.num_replicas,
                                                                 self.deployment_request.cpu_request,
                                                                 self.deployment_request.memory_request, self.num_clusters,
                                                                 self.free_cpu, self.free_memory)

                        if self.check_if_clusters_are_full_after_split_deployment(div):
                            self.penalty = True
                            logging.info('[Take Action] Block the NF1B1 strategy since cluster will be full!')
                        else:
                            # accept request
                            self.penalty = False
                            self.accepted_requests += 1
                            self.ep_accepted_requests += 1
                            self.deploy_nf1b1 += 1
                            self.deployment_request.split_clusters = div
                            self.deployment_request.is_deployment_split = True

                            # logging.info("[Divide] Before")
                            # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                            # logging.info("[Divide] CPU free: {}".format(self.free_cpu))
                            # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                            # logging.info("[Divide] MEM free: {}".format(self.free_memory))

                            avg_l = 0
                            avg_c = 0
                            avg_cpu = 0
                            clusters = 0
                            for d in range(len(div)):
                                # Update allocated amounts
                                self.allocated_cpu[d] += self.deployment_request.cpu_request * div[d]
                                self.allocated_memory[d] += self.deployment_request.memory_request * div[d]
                                avg_cpu += 100 * (self.allocated_cpu[d] / self.cpu_capacity[d])
                                # Update free resources
                                self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                                self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                                # Latency updates
                                avg_l += self.latency[d] * div[d]
                                self.increase_latency(d, 1.05)  # 5% increase max for split

                                # Cost Updates
                                type_id = int(self.cluster_type[d])
                                avg_c += DEFAULT_CLUSTER_TYPES[type_id]['cost'] * div[d]

                                # Load updates
                                self.avg_load_served[d] += div[d]

                            avg_l = avg_l / self.deployment_request.num_replicas
                            avg_c = avg_c / self.deployment_request.num_replicas
                            avg_cpu = avg_cpu / self.deployment_request.num_replicas

                            self.avg_latency.append(avg_l)
                            self.avg_cost.append(avg_c)
                            self.avg_cpu_usage_percentage_cluster_selected.append(avg_cpu)

                            logging.info("[NF1B1] Average Latency: {}".format(avg_l))
                            logging.info("[NF1B1] Average Cost: {}".format(avg_c))
                            logging.info("[NF1B1] Average CPU: {}".format(avg_cpu))
                            logging.info("[NF1B1] Average Load: {}".format(self.avg_load_served))

                            # Save expected latency and cost in deployment request
                            self.deployment_request.expected_latency = avg_l
                            self.deployment_request.expected_cost = avg_c

                            # logging.info("[Divide] After")
                            # logging.info("[Divide] CPU allocated: {}".format(self.allocated_cpu))
                            # logging.info("[Divide] CPU free: {}".format(self.free_cpu))

                            # logging.info("[Divide] MEM allocated: {}".format(self.allocated_memory))
                            # logging.info("[Divide] MEM free: {}".format(self.free_memory))
                            self.enqueue_request(self.deployment_request)
                '''

    # Current Strategy: First Fit Decreasing (FFD)
    def first_fit_decreasing_heuristic(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        logging.info('[first_fit_decreasing_heuristic] Num. replicas to distribute: {}'.format(num_replicas))
        distribution = [0] * num_clusters

        # min and max replicas
        min_replicas = 1
        max_replicas = num_replicas

        # Distribute the replicas across clusters
        for n in range(num_clusters):
            self.split_number_replicas[n] = min(free_cpu[n] / cpu_req,
                                                free_mem[n] / mem_req)

        # logging.info('[Take Action] Split factors: {}'.format(self.split_number_replicas))
        min_factor = int(math.ceil(min(self.split_number_replicas)))
        if min_factor >= max_replicas:
            min_factor = max_replicas - 1  # To really distribute at the end

        # logging.info('[Take Action] Min factor: {}'.format(min_factor))
        # Sort the clusters by their remaining capacity (CPU) in decreasing order
        sorted_clusters_cpu = {}
        for n in range(num_clusters):
            sorted_clusters_cpu[str(n)] = free_cpu[n]

        sorted_clusters_cpu = sort_dict_by_value(sorted_clusters_cpu, reverse=True)

        for key, value in sorted_clusters_cpu.items():
            n = int(key)
            if num_replicas == 0:
                break
            if num_replicas > 0 and min_factor < num_replicas and (
                    (cpu_req * min_factor < value) and (mem_req * min_factor < free_mem[n])):
                distribution[n] += min_factor
                num_replicas -= min_factor
            elif num_replicas > 0 and ((cpu_req < value) and (mem_req < free_mem[n])):
                distribution[n] += min_replicas
                num_replicas -= min_replicas

        # Still distribute remaining replicas if needed
        if num_replicas == 0:
            logging.info('[first_fit_decreasing_heuristic] Replicas division: {}'.format(distribution))
            return distribution
        else:
            logging.info('[first_fit_decreasing_heuristic] Replicas still to distribute...')
            for n in range(num_clusters):
                if num_replicas == 0:
                    break

                if (cpu_req < free_cpu[n]) and (mem_req < free_mem[n]):
                    distribution[n] += min_replicas
                    num_replicas -= min_replicas

            logging.info('[first_fit_decreasing_heuristic] Replicas division: {}'.format(distribution))
            return distribution

    # Current Strategy: First Fit Increasing (FFI)
    def first_fit_increasing_heuristic(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        logging.info('[Divide] Num. replicas to distribute: {}'.format(num_replicas))
        distribution = [0] * num_clusters

        # Calculate split factors
        split_factors = [min(free_cpu[n] / cpu_req, free_mem[n] / mem_req) for n in range(num_clusters)]

        # Calculate minimum factor
        min_factor = int(math.ceil(min(split_factors)))
        if min_factor >= num_replicas:
            min_factor = num_replicas - 1  # To really distribute at the end

        # Sort the clusters by their remaining capacity (CPU) in increasing order
        sorted_clusters_cpu = sorted(range(num_clusters), key=lambda x: free_cpu[x])

        for n in sorted_clusters_cpu:
            if num_replicas == 0:
                break

            if num_replicas > 0 and min_factor < num_replicas and (cpu_req < free_cpu[n]) and (mem_req < free_mem[n]):
                distribution[n] += min_factor
                num_replicas -= min_factor

        # Still distribute remaining replicas if needed
        if num_replicas > 0:
            logging.info('[Divide] Replicas still to distribute...')
            for n in range(num_clusters):
                if num_replicas == 0:
                    break

                if (cpu_req < free_cpu[n]) and (mem_req < free_mem[n]):
                    distribution[n] += 1
                    num_replicas -= 1

        logging.info('[Divide] Replicas division: {}'.format(distribution))
        return distribution

    def best_fit_heuristic_one_by_one(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        logging.info('[best_fit_heuristic_one_by_one] Num. replicas to distribute: {}'.format(num_replicas))
        distribution = [0] * num_clusters

        # Distribute the replicas across clusters
        for _ in range(num_replicas):
            # Sort the clusters by their remaining capacity (CPU) in increasing order
            sorted_clusters_cpu = sorted(range(num_clusters), key=lambda x: free_cpu[x])

            best_fit_bin = None
            best_fit_space = float('inf')

            for cluster_idx in sorted_clusters_cpu:
                if free_cpu[cluster_idx] >= cpu_req and free_mem[cluster_idx] >= mem_req:
                    space = free_cpu[cluster_idx] - cpu_req + free_mem[cluster_idx] - mem_req
                    if space < best_fit_space:
                        best_fit_bin = cluster_idx
                        best_fit_space = space

            if best_fit_bin is not None:
                distribution[best_fit_bin] += 1
                free_cpu[best_fit_bin] -= cpu_req
                free_mem[best_fit_bin] -= mem_req

        logging.info('[best_fit_heuristic_one_by_one] Replicas division: {}'.format(distribution))
        return distribution

    '''
    def best_fit_decreasing_heuristic(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        logging.info('[best_fit_heuristic] Num. replicas to distribute: {}'.format(num_replicas))
        distribution = [0] * num_clusters

        # min and max replicas
        min_replicas = 1
        max_replicas = num_replicas

        # Distribute the replicas across clusters
        for n in range(num_clusters):
            self.split_number_replicas[n] = min(free_cpu[n] / cpu_req,
                                                free_mem[n] / mem_req)

        # logging.info('[Take Action] Split factors: {}'.format(self.split_number_replicas))
        min_factor = int(math.ceil(min(self.split_number_replicas)))
        if min_factor >= max_replicas:
            min_factor = max_replicas - 1  # To really distribute at the end

        # logging.info('[Take Action] Min factor: {}'.format(min_factor))
        # Sort the clusters by their remaining capacity (CPU) in decreasing order
        sorted_clusters_cpu = {}
        for n in range(num_clusters):
            sorted_clusters_cpu[str(n)] = free_cpu[n]

        sorted_clusters_cpu = sort_dict_by_value(sorted_clusters_cpu, reverse=True)

        for key, value in sorted_clusters_cpu.items():
            n = int(key)
            if num_replicas == 0:
                break
            if num_replicas > 0 and min_factor < num_replicas and (
                    (cpu_req * min_factor < value) and (mem_req * min_factor < free_mem[n])):
                distribution[n] += min_factor
                num_replicas -= min_factor
            elif num_replicas > 0 and ((cpu_req < value) and (mem_req < free_mem[n])):
                best_fit_bin = None
                best_fit_space = float('inf')
                for cluster_idx in range(num_clusters):
                    if free_cpu[cluster_idx] >= cpu_req and free_mem[cluster_idx] >= mem_req:
                        space = free_cpu[cluster_idx] - cpu_req + free_mem[cluster_idx] - mem_req
                        if space < best_fit_space:
                            best_fit_bin = cluster_idx
                            best_fit_space = space
                if best_fit_bin is not None:
                    distribution[best_fit_bin] += min_replicas
                    num_replicas -= min_replicas

        # Still distribute remaining replicas if needed
        if num_replicas == 0:
            logging.info('[best_fit_heuristic] Replicas division: {}'.format(distribution))
            return distribution
        else:
            logging.info('[best_fit_heuristic] Replicas still to distribute...')
            for n in range(num_clusters):
                if num_replicas == 0:
                    break

                if (cpu_req < free_cpu[n]) and (mem_req < free_mem[n]):
                    distribution[n] += min_replicas
                    num_replicas -= min_replicas

            logging.info('[best_fit_heuristic] Replicas division: {}'.format(distribution))
            return distribution
    

    def next_fit_heuristic_one_by_one(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        logging.info('[next_fit_heuristic_one_by_one] Num. replicas to distribute: {}'.format(num_replicas))
        distribution = [0] * num_clusters

        # Index to keep track of the last used bin
        last_bin_idx = 0

        # Distribute the replicas across clusters
        for _ in range(num_replicas):
            # Find the next bin to use starting from the last used bin
            for _ in range(num_clusters):
                cluster_idx = (last_bin_idx + 1) % num_clusters
                if free_cpu[cluster_idx] >= cpu_req and free_mem[cluster_idx] >= mem_req:
                    distribution[cluster_idx] += 1
                    free_cpu[cluster_idx] -= cpu_req
                    free_mem[cluster_idx] -= mem_req
                    last_bin_idx = cluster_idx
                    break

        logging.info('[next_fit_heuristic_one_by_one] Replicas division: {}'.format(distribution))
        return distribution
    '''

    def get_state(self):
        # Get Observation state
        cluster = np.full(shape=(NUM_SPREADING_ACTIONS + 1, NUM_METRICS_CLUSTER + 1), fill_value=-1)

        observation = np.stack([self.allocated_cpu,
                                self.cpu_capacity,
                                self.allocated_memory,
                                self.memory_capacity,
                                self.latency],
                               axis=1)

        # Condition the elements in the set with the current node request
        request_demands = np.tile(
            np.array(
                [self.deployment_request.num_replicas,
                 self.deployment_request.cpu_request,
                 self.deployment_request.memory_request,
                 self.deployment_request.latency_threshold,
                 self.dt]
            ),
            (self.num_clusters + NUM_SPREADING_ACTIONS + 1, 1),
        )
        '''
        logging.info('[Get State]: cluster: {}'.format(cluster))
        logging.info('[Get State]: cluster shape: {}'.format(cluster.shape))
        logging.info('[Get State]: observation: {}'.format(observation))
        logging.info('[Get State]: observation shape: {}'.format(observation.shape))
        logging.info('[Get State]: request demands: {}'.format(request_demands))
        logging.info('[Get State]: request demands shape: {}'.format(request_demands.shape))
        '''
        observation = np.concatenate([observation, cluster], axis=0)
        # logging.info('[Get State]: concatenation: {}'.format(observation))
        # logging.info('[Get State]: concatenation shape: {}'.format(observation.shape))

        observation = np.concatenate([observation, request_demands], axis=1)
        # logging.info('[Get State]: concatenation: {}'.format(observation))
        # logging.info('[Get State]: concatenation shape: {}'.format(observation.shape))

        return observation

    # Save observation to csv file
    def save_obs_to_csv(self, obs_file, obs, date):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        cluster_obs = {}
        with file:
            fields.append('date')
            for n in range(self.num_clusters):
                fields.append("cluster_" + str(n + 1) + '_allocated_cpu')
                fields.append("cluster_" + str(n + 1) + '_cpu_capacity')
                fields.append("cluster_" + str(n + 1) + '_allocated_memory')
                fields.append("cluster_" + str(n + 1) + '_memory_capacity')
                fields.append("cluster_" + str(n + 1) + '_num_replicas')
                fields.append("cluster_" + str(n + 1) + '_cpu_request')
                fields.append("cluster_" + str(n + 1) + '_memory_request')
                fields.append("cluster_" + str(n + 1) + '_dt')

            # logging.info("[Save Obs] fields: {}".format(fields))

            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            cluster_obs = {}
            cluster_obs.update({fields[0]: date})

            for n in range(self.num_clusters):
                i = self.get_iteration_number(n)
                cluster_obs.update({fields[i + 1]: obs[n][0]})
                cluster_obs.update({fields[i + 2]: obs[n][1]})
                cluster_obs.update({fields[i + 3]: obs[n][2]})
                cluster_obs.update({fields[i + 4]: obs[n][3]})
                cluster_obs.update({fields[i + 5]: obs[n][4]})
                cluster_obs.update({fields[i + 6]: obs[n][5]})
                cluster_obs.update({fields[i + 7]: obs[n][6]})
                cluster_obs.update({fields[i + 8]: obs[n][7]})
            writer.writerow(cluster_obs)
        return

    def get_iteration_number(self, n):
        num_fields_per_cluster = 8
        return num_fields_per_cluster * n

    def enqueue_request(self, request: DeploymentRequest) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    # Action masks
    def action_masks(self):
        valid_actions = np.ones(self.num_clusters + NUM_SPREADING_ACTIONS + 1, dtype=bool)
        logging.info('[Action Mask]: (Before) Valid actions {} |'.format(valid_actions))

        for i in range(self.num_clusters):
            if self.check_if_cluster_is_full_after_full_deployment(i):
                valid_actions[i] = False
            else:
                valid_actions[i] = True

        # 4 additional actions: 3 strategies + Reject
        valid_actions[self.num_clusters + FFD] = True
        valid_actions[self.num_clusters + FFI] = True
        valid_actions[self.num_clusters + BF1B1] = True
        # valid_actions[self.num_clusters + NF1B1] = True
        valid_actions[self.num_clusters + NUM_SPREADING_ACTIONS] = True
        logging.info('[Action Mask]: Valid actions {} |'.format(valid_actions))
        return valid_actions

    # Double-check if the selected cluster is full
    def check_if_cluster_is_full_after_full_deployment(self, action):
        total_cpu = self.deployment_request.num_replicas * self.deployment_request.cpu_request
        total_memory = self.deployment_request.num_replicas * self.deployment_request.memory_request

        if (self.allocated_cpu[action] + total_cpu > 0.95 * self.cpu_capacity[action]
                or self.allocated_memory[action] + total_memory > 0.95 * self.memory_capacity[action]):
            logging.info('[Check]: Cluster {} is full...'.format(action + 1))
            return True

        return False

    # Double-check if the selected clusters are full (spread strategy)
    def check_if_clusters_are_full_after_split_deployment(self, div):
        for d in range(len(div)):
            total_cpu = self.deployment_request.cpu_request * div[d]
            total_memory = self.deployment_request.memory_request * div[d]

            if (self.allocated_cpu[d] + total_cpu > 0.95 * self.cpu_capacity[d]
                    or self.allocated_memory[d] + total_memory > 0.95 * self.memory_capacity[d]):
                logging.info('[Check]: Cluster {} is full...'.format(d))
                return True

        return False

    # Increase latency in the episode
    def increase_latency(self, n, factor):
        avg_value = self.latency[n]
        for n2 in range(self.num_clusters):
            if n == n2:  # for the same node assume 0
                self.latency_matrix[n][n2] = 0
            else:
                prev = self.latency_matrix[n][n2]
                new_latency = max(min(prev * factor, MAX_DELAY), MIN_DELAY)

                self.latency_matrix[n][n2] = new_latency

                if self.latency_matrix[n][n2] == 0:
                    self.latency_matrix[n][n2] = 1.0

                # logging.info("[Increase Latency] previous latency: {} "
                #             "| updated Latency: {}".format(prev, self.latency_matrix[n][n2]))

                self.latency_matrix[n2][n] = self.latency_matrix[n][n2]

        # Update Latency of all nodes
        for c in range(self.num_clusters):
            self.latency[c] = mean(self.latency_matrix[c])

        logging.info("[Increase Latency] cluster: {} | previous latency: {} "
                     "| updated Latency: {}".format(n + 1, avg_value, self.latency[n]))

    # Decrease Latency in the episode
    def decrease_latency(self, n, factor):
        avg_value = self.latency[n]
        for n2 in range(self.num_clusters):
            if n == n2:  # for the same node assume 0
                self.latency_matrix[n][n2] = 0
            else:
                prev = self.latency_matrix[n][n2]
                new_latency = max(min(prev / factor, MAX_DELAY), MIN_DELAY)

                self.latency_matrix[n][n2] = new_latency

                if self.latency_matrix[n][n2] == 0:
                    self.latency_matrix[n][n2] = 1.0

                self.latency_matrix[n2][n] = self.latency_matrix[n][n2]

                # logging.info("[Decrease Latency] previous latency: {} "
                #             "| updated Latency: {}".format(prev, self.latency_matrix[n][n2]))

        # Update Latency of all nodes
        for c in range(self.num_clusters):
            self.latency[c] = mean(self.latency_matrix[c])

        logging.info("[Decrease Latency] cluster: {} | previous latency: {} "
                     "| updated Latency: {}".format(n + 1, avg_value, self.latency[n]))

    # Remove deployment request
    def dequeue_request(self):
        _, deployment_request = heapq.heappop(self.running_requests)
        # logging.info("[Dequeue] Request {}...".format(deployment_request))
        logging.info("[Dequeue] Request will be terminated...")
        # logging.info("[Dequeue] Before: ")
        # logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        # logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        # logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        # logging.info("[Dequeue] MEM free: {}".format(self.free_memory))

        if deployment_request.is_deployment_split:
            # logging.info("[Dequeue] Deployment is split...")
            for d in range(self.num_clusters):
                total_cpu = self.deployment_request.cpu_request * self.deployment_request.split_clusters[d]
                total_memory = self.deployment_request.memory_request * self.deployment_request.split_clusters[d]

                # Update allocate amounts
                self.allocated_cpu[d] -= total_cpu
                self.allocated_memory[d] -= total_memory

                # Update free resources
                self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                # Decrease Latency if replicas were there
                if total_cpu != 0:
                    self.decrease_latency(d, 1.10)  # only 10% if split
        else:
            # logging.info("[Dequeue] Deployment is not split...")
            n = deployment_request.deployed_cluster
            total_cpu = self.deployment_request.num_replicas * self.deployment_request.cpu_request
            total_memory = self.deployment_request.num_replicas * self.deployment_request.memory_request

            # Update allocate amounts
            self.allocated_cpu[n] -= total_cpu
            self.allocated_memory[n] -= total_memory

            # Update free resources
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

            # Decrease Latency
            self.decrease_latency(n, 1.15)  # 15% max reduction

        # logging.info("[Dequeue] After: ")
        # logging.info("[Dequeue] CPU allocated: {}".format(self.allocated_cpu))
        # logging.info("[Dequeue] CPU free: {}".format(self.free_cpu))
        # logging.info("[Dequeue] MEM allocated: {}".format(self.allocated_memory))
        # logging.info("[Dequeue] MEM free: {}".format(self.free_memory))

    # Check if all clusters are full
    def check_if_cluster_is_really_full(self) -> bool:
        is_full = [self.check_if_cluster_is_full_after_full_deployment(i) for i in range(self.num_clusters)]
        return np.all(is_full)

    # Create a deployment request
    def deployment_generator(self):
        deployment_list = get_c2e_deployment_list()
        n = self.np_random.integers(low=0, high=len(deployment_list))
        d = deployment_list[n - 1]

        if self.min_replicas == self.max_replicas:
            d.num_replicas = self.min_replicas
        else:
            d.num_replicas = self.np_random.integers(low=self.min_replicas, high=self.max_replicas)
        return d

    # Select (random) the next deployment request
    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        self.deployment_request = self.deployment_generator()
        logging.info('[Next Request]: Name: {} | Replicas: {}'.format(self.deployment_request.name,
                                                                      self.deployment_request.num_replicas))
