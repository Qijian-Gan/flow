import os
import json

import ray
import numpy as np

from flow.utils.rllib import FlowParamsEncoder
from flow.utils.registry import make_create_env
from flow.core.params import AimsunParams, NetParams, VehicleParams, EnvParams, InitialConfig

from multi_light import CoordinatedNetwork, MultiLightEnv, ADDITIONAL_ENV_PARAMS

try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class

from gym.spaces import Box, Tuple, Discrete
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.registry import register_env
from ray import tune

SIM_STEP = 1  # copy to run.py #sync time

# hardcoded to AIMSUN's statistics update interval (5 minutes)
DETECTOR_STEP = 900  # copy to run.py #Cj: every 15 minutes

TIME_HORIZON = 3600*4 - DETECTOR_STEP  # 14400
HORIZON = int(TIME_HORIZON//SIM_STEP)  # 18000

RLLIB_N_CPUS = 3
RLLIB_HORIZON = int(TIME_HORIZON//DETECTOR_STEP)  # 16

RLLIB_N_ROLLOUTS = 3  # copy to coordinated_lights.py
RLLIB_TRAINING_ITERATIONS = 1000000

net_params = NetParams(template=os.path.abspath("scenario_one_hour.ang"))
initial_config = InitialConfig()
vehicles = VehicleParams()
env_params = EnvParams(horizon=HORIZON,
                       warmup_steps= int(np.ceil(120/DETECTOR_STEP)), # 1
                       sims_per_step=int(DETECTOR_STEP/SIM_STEP),  # 900
                       additional_params=ADDITIONAL_ENV_PARAMS)
sim_params = AimsunParams(sim_step=SIM_STEP,
                          render=False,
                          restart_instance=False,
                          #   replication_name="Replication (one hour)",
                          replication_name=ADDITIONAL_ENV_PARAMS['replication_list'][0],
                          centroid_config_name="Centroid Configuration 8040652")


flow_params = dict(
    exp_tag="multi_nonshared_trial1",
    env_name=MultiLightEnv,
    network=CoordinatedNetwork,
    simulator='aimsun',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
)

create_env, gym_name = make_create_env(params=flow_params)
# Register as rllib env
register_env(gym_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space

def gen_policy(act_space):
    """Generate a policy in RLlib."""
    return PPOTFPolicy, obs_space, act_space, {}

N_phases = 9
def base_action(n_phases):
    return n_phases*[Discrete(100,)] + (N_phases-n_phases)*[Discrete(0)]  

def n1_action():
    space = base_action(7) 
    return Tuple(*space)  # 5 (probabilities)

def n2_action():
    space = base_action(9) 
    return Tuple(*space)  # 5 (probabilities)
#
#def n3_action():
#    space = base_action(6) # ??? no signal group: check if 5 or 6
#    return Tuple(*space)  # 5 (probabilities)
#
#def n4_action():
#    space = base_action(6) 
#    return Tuple(*space)  # 5 (probabilities)
#
#def n5_action():
#    space = base_action(4) 
#    return Tuple(*space)  # 5 (probabilities)

POLICY_GRAPHS = {'3329': gen_policy(Tuple(7*[Discrete(100,)])),
                 '3344': gen_policy(Tuple(9*[Discrete(100,)]))}
                 #'n3': gen_policy(n3_action),
                 #'n4': gen_policy(n4_action),
                 #'n5': gen_policy(n5_action)}

POLICIES_TO_TRAIN = ['3329','3344']

def policy_mapping_fn(agent_id):
    """Map a policy in RLlib."""
    return agent_id


def setup_exps(version=0):
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = RLLIB_N_CPUS
    config["sgd_minibatch_size"] = 32 # 16
    config["train_batch_size"] = RLLIB_HORIZON * RLLIB_N_ROLLOUTS  # 16*3
    config["sample_batch_size"] = RLLIB_HORIZON * RLLIB_N_ROLLOUTS
    config["model"].update({"fcnet_hiddens": [64, 64, 64]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config['clip_actions'] = False  # (ev) temporary ray bug
    config["horizon"] = RLLIB_HORIZON  # not same as env horizon.
    config["vf_loss_coeff"] = 1
    config["gamma"] = 0.999
    # config["lr"] = 5e-4 #vary, lr
    config["lr_schedule"] = [[0, 5e-3], [120000, 5e-4]]

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    print("policy_graphs", POLICY_GRAPHS)
    config['multiagent'].update({'policies': POLICY_GRAPHS})
    config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    config['multiagent'].update({'policies_to_train': POLICIES_TO_TRAIN})

    #create_env, gym_name = make_create_env(params=flow_params)

    return alg_run, gym_name, config


if __name__ == "__main__":
    ray.init(num_cpus=RLLIB_N_CPUS + 1, object_store_memory=int(1e8))

    alg_run, gym_name, config = setup_exps()
    trials = ray.tune.run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 1,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": RLLIB_TRAINING_ITERATIONS,
            },
            #"restore": '/home/damian/ray_results/multi_light_trial1/PPO_MultiLightEnv-v0_aa8aa4f8_2020-09-30_06-10-40u92vq2gv/checkpoint_1728/checkpoint-1728',
            # "local_dir": os.path.abspath("./ray_results"),
            "keep_checkpoints_num": 7,
            "checkpoint_score_attr": "episode_reward_mean"
        }
    },
        resume=False)
