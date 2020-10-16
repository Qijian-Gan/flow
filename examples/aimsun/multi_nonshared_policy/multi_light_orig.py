import numpy as np
from gym.spaces import Box, Tuple, Discrete
import math

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.envs import Env
from flow.networks import Network
from aimsun_props import Aimsun_Params

ADDITIONAL_ENV_PARAMS = {'target_nodes': [3329, 3344, 3370, 3341, 3369],
                         # 'observed_nodes': [3386, 3371, 3362, 3373],
                         'num_incoming_edges_per_node': 4,
                         'num_detector_types': 4,
                         'num_measures': 2,
                         'detection_interval': (0, 15, 0),
                         'statistical_interval': (0, 15, 0),
                         'replication_list': ['Replication 8050297',  # 5-11
                                              'Replication 8050315',  # 10-14
                                              'Replication 8050322'
                                              ]}  # 14-21
# the replication list should be copied in load.py

RLLIB_N_ROLLOUTS = 3  # copy from train_rllib.py

np.random.seed(1234567890)

# read csv of Node Parameters
ap = Aimsun_Params('/home/cjrsantos/flow/flow/utils/aimsun/aimsun_props.csv')

def rescale_bar(array, NewMin, NewMax):
    rescaled_action = []
    OldMin = 0
    OldMax = 70
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)
    for OldValue in array:
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        rescaled_action.append(NewValue)
    return rescaled_action

def rescale_act(actions_array, target_value, current_value):
    rescaled_actions = []
    target_value = round(target_value)
    for duration in actions_array:
        if current_value == 0:
            new_action = 0
        else:
            new_action = math.ceil(target_value*duration/current_value)
        rescaled_actions.append(int(new_action))
    if sum(rescaled_actions) > target_value:
        x = sum(rescaled_actions) - target_value
        rescaled_actions[-1] = int(rescaled_actions[-1] - x)
    return rescaled_actions


class MultiLightEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='aimsun'):
        for param in ADDITIONAL_ENV_PARAMS:
            if param not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(param))

        super().__init__(env_params, sim_params, network, simulator)
        self.additional_params = env_params.additional_params

        self.episode_counter = 0
        self.detection_interval = self.additional_params['detection_interval'][1]*60  # assuming minutes for now
        self.k.simulation.set_detection_interval(*self.additional_params['detection_interval'])
        self.k.simulation.set_statistical_interval(*self.additional_params['statistical_interval'])
        self.k.traffic_light.set_replication_seed(np.random.randint(2e9))

        # target intersections
        self.target_nodes = env_params.additional_params["target_nodes"]

        # reset_offset_durations
        for node_id in self.target_nodes:
            default_offset = self.k.traffic_light.get_intersection_offset(node_id)
            self.k.traffic_light.change_intersection_offset(node_id, -default_offset)

        self.edge_detector_dict = {}
        self.edges_with_detectors = {}
        self.past_cumul_queue = {}
        self.current_phase_timings = np.zeros(int(len(self.target_nodes)))
        #ap_keys = dict.fromkeys(['control_id', 'num_rings', 'green_phases', 'cc_dict', 'sum_interphase', 'max_dict', 'max_p'])
        #self.aimsun_props = {dict.fromkeys(self.target_nodes, ap_keys)}
        self.aimsun_props = {}
        # change to {node_id: {ring:, gp: , cc_dict:, sum_int:, max_dict: , maxp:}}

        # get node values
        for node_id in self.target_nodes:
            self.aimsun_props[node_id] = {}
            self.past_cumul_queue[node_id] = {}
            self.edge_detector_dict[node_id] = {}
            self.rep_name, _ = self.k.traffic_light.get_replication_name(node_id)
            incoming_edges = self.k.traffic_light.get_incoming_edges(node_id)

            # get initial detector values
            for edge_id in incoming_edges:
                detector_dict = self.k.traffic_light.get_detectors_on_edge(edge_id)
                through = detector_dict['through']
                right = detector_dict['right']
                left = detector_dict['left']
                advanced = detector_dict['advanced']
                type_map = {"through": through, "right": right, "left": left, "advanced": advanced}

                self.edge_detector_dict[node_id][edge_id] = type_map
                self.past_cumul_queue[node_id][edge_id] = 0
            # get control plan params

            control_id, num_rings = self.k.traffic_light.get_control_ids(
                node_id)  # self.control_id = list, num_rings = list
            max_dict, max_p = ap.get_max_dict(node_id, self.rep_name)

            self.aimsun_props[node_id]['green_phases'] = ap.get_green_phases(node_id)
            self.aimsun_props[node_id]['cc_dict'] = ap.get_cp_cycle_dict(node_id, self.rep_name)
            self.aimsun_props[node_id]['sum_interphase'] = ap.get_sum_interphase_per_ring(node_id)[0]
            self.aimsun_props[node_id]['control_id'] = control_id
            self.aimsun_props[node_id]['num_rings'] = num_rings
            self.aimsun_props[node_id]['max_dict'] = max_dict
            self.aimsun_props[node_id]['max_p'] = max_p

        self.ignore_policy = False

    @property
    def action_space(self): #defined for a single agent
        """See class definition."""
        return Tuple(9 * (Discrete(0,),))  # 8+1 (probabilities)

    @property
    def observation_space(self): #defined for a single agent
        """See class definition."""
        ap = self.additional_params
        shape = (ap['num_incoming_edges_per_node']
                 * (ap['num_detector_types'])*ap['num_measures'])
        return Box(low=0, high=30, shape=(shape, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):

        if self.ignore_policy:
            #print('self.ignore_policy is True')
            return
        cycle = self.cycle - self.sum_interphase
        def_actions = np.array(rl_actions).flatten()
        actions = rescale_bar(def_actions,10,90)
        barrier = actions[-1]/100
        sum_barrier = [round(cycle*barrier), cycle - round(cycle*barrier)]
        action_rings = [[actions[0:2],actions[2:4]],[actions[4:6],actions[6:8]]]

        for i in range(len(action_rings)):
            ring = action_rings[i]
            for j in range(len(ring)):
                phase_pair = ring[j]
                if sum(phase_pair) != sum_barrier[j]:
                    action_rings[i][j] = rescale_act(phase_pair, sum_barrier[j], sum(phase_pair))
        #print(action_rings)

        rescaled_actions = np.array(action_rings).flatten()
        phase_list = self.observed_phases
        for phase, action, maxd in zip(phase_list, rescaled_actions, self.maxd_list):
            if action:
                if action > maxd:
                    maxout = action
                else:
                    maxout = maxd
                self.k.traffic_light.change_phase_duration(self.node_id, phase, action, maxd)
                #phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(self.node_id, phase)
                #print(phase, action, phase_duration)

        self.current_phase_timings = rescaled_actions
        self.sum_barrier = [sum(rescaled_actions[0:2]), sum(rescaled_actions[2:4])]
        # Get control_id & replication name every step

        ###########################################################################################

        if self.ignore_policy:
            #print('self.ignore_policy is True')
            return
        
        # names of RL agents in the network
        agent_ids = [node_id for node_id in self.target_nodes]

        #define different actions for different multiagents
        n1_action = rl_actions['n1']
        n2_action = rl_actions['n2']
        #n3_action = rl_actions['n3']
        #n4_action = rl_actions['n4']
        #n5_action = rl_actions['n5']

        # rl_action = n1_action + n2_action + n3_action + n4_action + n5_action

        #use base env method to convert actions to phase timing for tl agents
        

        delta = 113
        self.rep_name, _ = self.k.traffic_light.get_replication_name(3344)
        default_actions = np.array(rl_actions).flatten()
        def_actions = np.round(rescale_phase_pair(default_actions, 14, 100))
        all_actions = []

        for node_id in self.target_nodes:
            control_id, num_rings = self.k.traffic_light.get_control_ids(
                node_id)  # self.control_id = list, num_rings = list
            max_dict, max_p = ap.get_max_dict(node_id, self.rep_name)

            self.aimsun_props[node_id]['cc_dict'] = ap.get_cp_cycle_dict(node_id, self.rep_name)
            self.aimsun_props[node_id]['control_id'] = control_id
            self.aimsun_props[node_id]['max_dict'] = max_dict
            self.aimsun_props[node_id]['max_p'] = max_p

            cycle = self.aimsun_props[node_id]['cc_dict'][control_id]
            cycle = cycle - self.aimsun_props[node_id]['sum_interphase']
            phase_list = self.aimsun_props[node_id]['green_phases']
            cur_maxdl = max_p[control_id]
            maxd_list = max_dict[cur_maxdl]

            if node_id == 3329:
                actions = np.array([x/delta for x in def_actions[:4]])
                prob_barrier = [actions[-1], (1 - actions[-1])]
                sum_barrier = [round(prob_barrier[0]*cycle), round(prob_barrier[1]*cycle)]
                prob_phase = np.array([[actions[0], actions[1]], [actions[2]]])
                actionf = []

                # probability
                for i in range(len(prob_phase)):  # prob phase = [[0,1],[2]] = 2
                    ring = prob_phase[i]  # i=0, ring = [0,1], i=1, ring = [2]
                    for j in range(len(ring)):  # 0 and 1
                        if i == 1:
                            new_phase1, new_phase2 = round(
                                ring[j]*sum_barrier[1]), sum_barrier[1] - round(ring[j]*sum_barrier[1])
                        else:
                            new_phase1, new_phase2 = round(
                                ring[j]*sum_barrier[j]), sum_barrier[j] - round(ring[j]*sum_barrier[j])
                        # Setting minimum green to 5
                        if new_phase1 < 5:
                            new_phase1 = 5
                            new_phase2 = sum_barrier[j] - new_phase1
                        elif new_phase2 < 5:
                            new_phase2 = 5
                            new_phase1 = sum_barrier[j] - new_phase2
                        phase_pair = [new_phase1, new_phase2]
                        actionf.append(phase_pair)

                cur_actions = np.array(actionf).flatten()
                new_actions = np.insert(cur_actions, 4, sum_barrier[0]+4)
                all_actions.append(list(new_actions))
                for phase, action, maxd in zip(phase_list, new_actions, maxd_list):
                    if action:
                        if action > maxd:
                            maxout = action
                        else:
                            maxout = maxd
                        self.k.traffic_light.change_phase_duration(node_id, phase, action, maxout)
                        #phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(node_id, phase)

            elif node_id == 3344:
                actions = np.array([x/delta for x in def_actions[4:9]])
                prob_barrier = [actions[-1], 1 - actions[-1]]
                sum_barrier = [round(prob_barrier[0]*cycle), round(prob_barrier[1]*cycle)]
                prob_phase = np.array([[actions[0], actions[1]], [actions[2], actions[3]]])
                # gives the length of barriers
                actionf = []

                # probability
                for i in range(len(prob_phase)):  # [[0,1],[2,3]]
                    ring = prob_phase[i]  # [0,1]
                    for j in range(len(ring)):
                        new_phase1, new_phase2 = round(
                            ring[j]*sum_barrier[j]), sum_barrier[j] - round(ring[j]*sum_barrier[j])
                        # Setting minimum green to 5
                        if new_phase1 < 5:
                            new_phase1 = 5
                            new_phase2 = sum_barrier[j] - new_phase1
                        elif new_phase2 < 5:
                            new_phase2 = 5
                            new_phase1 = sum_barrier[j] - new_phase2
                        phase_pair = [new_phase1, new_phase2]
                        actionf.append(phase_pair)

                new_actions = np.array(actionf).flatten()
                all_actions.append(list(new_actions))
                for phase, action, maxd in zip(phase_list, new_actions, maxd_list):
                    if action:
                        if action > maxd:
                            maxout = action
                        else:
                            maxout = maxd
                        self.k.traffic_light.change_phase_duration(node_id, phase, action, maxout)
                        #phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(node_id, phase)

            elif node_id == 3370:
                actions = np.array([x/delta for x in def_actions[9:12]])
                prob_barrier = [actions[-1], 1 - (actions[-1])]
                sum_barrier = [round(prob_barrier[0]*cycle), round(prob_barrier[1]*cycle)]
                prob_phase = np.array([[actions[0], actions[1]]])
                # gives the length of barriers
                actionf = []

                # probability
                for i in range(len(prob_phase)):  # [[0,1],[2,3]]
                    ring = prob_phase[i]  # [0,1]
                    for j in range(len(ring)):
                        new_phase1, new_phase2 = round(
                            ring[j]*sum_barrier[0]), sum_barrier[0] - round(ring[j]*sum_barrier[0])
                        # Setting minimum green to 5
                        if new_phase1 < 5:
                            new_phase1 = 5
                            new_phase2 = sum_barrier[0] - new_phase1
                        elif new_phase2 < 5:
                            new_phase2 = 5
                            new_phase1 = sum_barrier[0] - new_phase2
                        phase_pair = [new_phase1, new_phase2]
                        actionf.append(phase_pair)

                cur_actions = np.array(actionf).flatten()
                new_actions = np.insert(cur_actions, 2, sum_barrier[1])
                new_actions = np.append(new_actions, sum_barrier[1])
                all_actions.append(list(new_actions))

                for phase, action, maxd in zip(phase_list, new_actions, maxd_list):
                    if action:
                        if action > maxd:
                            maxout = action
                        else:
                            maxout = maxd
                        self.k.traffic_light.change_phase_duration(node_id, phase, action, maxout)
                        #phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(node_id, phase)

            elif node_id == 3341:
                actions = np.array([x/delta for x in def_actions[12:15]])
                prob_barrier = [actions[-1], 1 - (actions[-1])]
                sum_barrier = [round(prob_barrier[0]*cycle), round(prob_barrier[1]*cycle)]
                prob_phase = np.array([[actions[0], actions[1]]])
                # gives the length of barriers
                actionf = []

                # probability
                for i in range(len(prob_phase)):  # [[0,1],[2,3]]
                    ring = prob_phase[i]  # [0,1]
                    for j in range(len(ring)):
                        new_phase1, new_phase2 = round(
                            ring[j]*sum_barrier[0]), sum_barrier[0] - round(ring[j]*sum_barrier[0])
                        # Setting minimum green to 5
                        if new_phase1 < 5:
                            new_phase1 = 5
                            new_phase2 = sum_barrier[0] - new_phase1
                        elif new_phase2 < 5:
                            new_phase2 = 5
                            new_phase1 = sum_barrier[0] - new_phase2
                        phase_pair = [new_phase1, new_phase2]
                        actionf.append(phase_pair)

                cur_actions = np.array(actionf).flatten()
                new_actions = np.insert(cur_actions, 2, sum_barrier[1])
                new_actions = np.append(new_actions, sum_barrier[1] + 5)
                all_actions.append(list(new_actions))

                for phase, action, maxd in zip(phase_list, new_actions, maxd_list):
                    if action:
                        if action > maxd:
                            maxout = action
                        else:
                            maxout = maxd
                        self.k.traffic_light.change_phase_duration(node_id, phase, action, maxout)
                        #phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(node_id, phase)

            elif node_id == 3369:
                actions = def_actions[15] / delta
                prob_barrier = [actions, (1 - actions)]
                cur_actions = [round(prob_barrier[0]*cycle), round(prob_barrier[1]*cycle),
                               round(prob_barrier[0]*cycle)]
                cur_actions = np.array(cur_actions).flatten()
                new_actions = np.append(cur_actions, round(prob_barrier[1]*cycle) + 5)
                all_actions.append(list(new_actions))
                # gives the length of barriers

                for phase, action, maxd in zip(phase_list, new_actions, maxd_list):
                    if action:
                        if action > maxd:
                            maxout = action
                        else:
                            maxout = maxd
                        self.k.traffic_light.change_phase_duration(node_id, phase, action, maxout)
                        #phase_duration, maxd, _ = self.k.traffic_light.get_duration_phase(node_id, phase)

        self.current_phase_timings = all_actions

    def get_state(self, rl_id=None, **kwargs):
        """See class definition."""

        ap = self.additional_params

        num_nodes = len(self.target_nodes)
        num_edges = ap['num_incoming_edges_per_node']
        num_detectors_types = (ap['num_detector_types'])
        num_measures = (ap['num_measures'])
        normal = 2000

        ma_state = {key: 0, for key in self.target_nodes}
        shape = (num_edges, num_detectors_types, num_measures)
        for i, (node, edge) in enumerate(self.edge_detector_dict.items()):
            det_state = np.zeros(shape)
            for j, (edge_id, detector_info) in enumerate(edge.items()):
                for k, (detector_type, detector_ids) in enumerate(detector_info.items()):
                    if detector_type == 'through':
                        index = (i, j, 0)
                        # flow, occup = 0, []
                        # for detector in detector_ids:
                        #     count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                        #     flow += (count/self.detection_interval)/(normal/3600)
                        #     occup.append(occupancy)
                        # det_state[(*index, 0)] = flow
                        # try:
                        #     det_state[(*index, 1)] = sum(occup)/len(occup)  # mean
                        # except ZeroDivisionError:
                        #     det_state[(*index, 1)] = 0
                    elif detector_type == 'right':
                        index = (i, j, 1)
                        # flow, occup = 0, []
                        # for detector in detector_ids:
                        #     count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                        #     flow += (count/self.detection_interval)/(normal/3600)
                        #     occup.append(occupancy)
                        # det_state[(*index, 0)] = flow
                        # try:
                        #     det_state[(*index, 1)] = sum(occup)/len(occup)  # mean
                        # except ZeroDivisionError:
                        #     det_state[(*index, 1)] = 0
                    elif detector_type == 'left':
                        index = (i, j, 2)
                        # flow, occup = 0, []
                        # for detector in detector_ids:
                        #     count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                        #     flow += (count/self.detection_interval)/(normal/3600)
                        #     occup.append(occupancy)
                        # det_state[(*index, 0)] = flow
                        # try:
                        #     det_state[(*index, 1)] = sum(occup)/len(occup)
                        # except ZeroDivisionError:
                        #     det_state[(*index, 1)] = 0
                    elif detector_type == 'advanced':
                        index = (i, j, 3)

                    flow, occup = 0, []
                    for detector in detector_ids:
                        count, occupancy = self.k.traffic_light.get_detector_count_and_occupancy(int(detector))
                        flow += (count/self.detection_interval)/(normal/3600)
                        occup.append(occupancy)
                    det_state[(*index, 0)] = flow
                    try:
                        det_state[(*index, 1)] = sum(occup)/len(occup)
                    except ZeroDivisionError:
                        det_state[(*index, 1)] = 0

            state = det_state.flatten()
            ma_state[node] = state

        return ma_state

    def compute_reward(self, rl_actions, **kwargs):
        """Computes the sum of queue lengths at all intersections in the network."""
        # change to queue + util per node
        reward = 0
        node_gUtil = self.k.traffic_light.get_green_util(3322)
        ma_reward = {key: 0, for key in self.target_nodes}
        for i, node_id in enumerate(self.target_nodes):
            r_queue = 0
            gUtil = node_gUtil[i]
            a1 = 1
            a0 = 0.2

            for section_id in self.past_cumul_queue[node_id]:  # self.past_cumul_queue[node_id]
                current_cumul_queue = self.k.traffic_light.get_cumulative_queue_length(section_id)
                queue = current_cumul_queue - self.past_cumul_queue[node_id][section_id]
                self.past_cumul_queue[node_id][section_id] = current_cumul_queue

                r_queue += queue
            new_reward = ((a0*r_queue) + (a1*gUtil))

            reward -= (new_reward ** 2) * 100
            ma_reward[node_id] = reward

        print(f'{self.k.simulation.time:.0f}', '\t', f'{ma_reward}', '\t',
              self.current_phase_timings[0], '\t', self.current_phase_timings[1], '\t',
              self.current_phase_timings[2], '\t', self.current_phase_timings[3], '\t',
              self.current_phase_timings[4])

        return ma_reward

    def step(self, rl_actions):
        """See parent class."""

        self.time_counter += self.env_params.sims_per_step
        self.step_counter += self.env_params.sims_per_step

        self.apply_rl_actions(rl_actions)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        for _ in range(self.env_params.sims_per_step):
            self.k.simulation.update(reset=False)

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.warmup_steps +
                self.env_params.horizon)  # or crash

        # compute the info for each agent
        infos = {}

        # compute the reward
        reward = self.compute_reward(rl_actions)

        return next_observation, reward, done, infos

    def reset(self):
        """See parent class.

        The AIMSUN simulation is reset along with other variables.
        """
        # reset the step counter
        self.step_counter = 0

        if self.episode_counter:
            self.k.simulation.reset_simulation()

            episode = self.episode_counter % RLLIB_N_ROLLOUTS

            print('-----------------------')
            print(f'Episode {RLLIB_N_ROLLOUTS if not episode else episode} of {RLLIB_N_ROLLOUTS} complete')
            print('Resetting simulation')
            print('-----------------------')

        # increment episode count
        self.episode_counter += 1

        rep_name, rep_seed = self.k.traffic_light.get_replication_name(3344)
        print('-----------------------')
        print(f'Replication Name: {rep_name} Seed: {rep_seed}')
        print('-----------------------')

        # reset variables
        for node_id in self.target_nodes:
            for section_id in self.past_cumul_queue[node_id]:
                self.past_cumul_queue[node_id][section_id] = 0

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation

## MOVE THESE TO TRAIN_MA_RLLIB

test_env = create_env()
obs_space = test_env.observation_space

def gen_policy(act_space):
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}

N_phases = 8
def base_action(n_phases):
    return n_phases*[Discrete(100,)] + (N_phases-n_phases)*[Discrete(0)]  

def n1_action():
    space = base_action(7) 
    return Tuple(*space)  # 5 (probabilities)

def n2_action():
    space = base_action(8) 
    return Tuple(*space)  # 5 (probabilities)

def n3_action():
    space = base_action(6) # ??? no signal group: check if 5 or 6
    return Tuple(*space)  # 5 (probabilities)

def n4_action():
    space = base_action(6) 
    return Tuple(*space)  # 5 (probabilities)

def n5_action():
    space = base_action(4) 
    return Tuple(*space)  # 5 (probabilities)

POLICY_GRAPHS = {'n1': gen_policy(n1_action),
                 'n2': gen_policy(n2_action),
                 'n3': gen_policy(n3_action),
                 'n4': gen_policy(n4_action),
                 'n5': gen_policy(n5_action)}

def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    return agent_id