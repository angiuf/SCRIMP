import os

import numpy as np
import torch
import wandb

from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import WarehouseEnv
from model import Model
from util import reset_env, make_gif, set_global_seeds
import csv

NUM_TIMES = 200
# CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3], [16, 20, 0.0], [16, 20, 0.15], [16, 20, 0.3], [32, 30, 0.0],
#         [32, 30, 0.15], [32, 30, 0.3], [64, 40, 0.0], [64, 40, 0.15], [64, 40, 0.3], [128, 40, 0.0],
#         [128, 40, 0.15], [128, 40, 0.3]]
# CASE  = [[4, 50], [8, 50], [16, 50], [32, 50], [64, 50], [128, 50], [256, 50]] # [num_agents, env_size]
CASE  = [[128, 50], [256, 50]] # [num_agents, env_size]
set_global_seeds(SetupParameters.SEED)


def one_step(env0, actions, model0, pre_value, input_state, ps, one_episode_perf, message, episodic_buffer0):
    obs, vector, reward, done, _, on_goal, _, _, _, _, _, max_on_goal, num_collide, _, modify_actions = env0.joint_step(
        actions, one_episode_perf['episode_len'], model0, pre_value, input_state, ps, no_reward=False, message=message,
        episodic_buffer=episodic_buffer0)

    one_episode_perf['collide'] += num_collide
    vector[:, :, -1] = modify_actions
    one_episode_perf['episode_len'] += 1
    return reward, obs, vector, done, one_episode_perf, max_on_goal, on_goal


def evaluate(eval_env, model0, device, episodic_buffer0, num_agent, case_id, save_gif0):
    """Evaluate Model."""
    one_episode_perf = {'episode_len': 0, 'max_goals': 0, 'collide': 0, 'success_rate': 0, 'total_steps': 0, 'avg_steps': 0, 'max_steps': 0}
    episode_frames = []

    done, _, obs, vector, _ = reset_env(eval_env, num_agent, case_id)

    episodic_buffer0.reset(2e6, num_agent)
    new_xy = eval_env.get_positions()
    episodic_buffer0.batch_add(new_xy)

    message = torch.zeros((1, num_agent, NetParameters.NET_SIZE)).to(torch.device('cuda:0'))
    hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device),
                    torch.zeros((num_agent, NetParameters.NET_SIZE // 2)).to(device))

    if save_gif0:
        episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

    step_count = 0
    episode_length = 0
    solution = [[eval_env.get_positions()[i] + (0,)] for i in range(num_agent)]
    agents_steps = np.zeros(num_agent)
    while not done:
        actions, hidden_state, v_all, ps, message = model0.final_evaluate(obs, vector, hidden_state, message, num_agent,
                                                                          greedy=False)
        

        rewards, obs, vector, done, one_episode_perf, max_on_goals, on_goal = one_step(eval_env, actions, model0, v_all,
                                                                                       hidden_state, ps,
                                                                                       one_episode_perf, message,
                                                                                       episodic_buffer0)
        # Update the solution
        episode_length += 1
        for i in range(num_agent):
            solution[i].append(eval_env.get_positions()[i] + (episode_length,))

        # Count non zero actions
        steps = np.count_nonzero(actions)
        step_count += steps

        # Update the agents steps
        for i in range(num_agent):
            if actions[i] != 0:
                agents_steps[i] += 1

        new_xy = eval_env.get_positions()
        processed_rewards, _, intrinsic_reward, min_dist = episodic_buffer0.if_reward(new_xy, rewards, done, on_goal)

        vector[:, :, 3] = rewards
        vector[:, :, 4] = intrinsic_reward
        vector[:, :, 5] = min_dist

        if save_gif0:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        if done:
            if one_episode_perf['episode_len'] < EnvParameters.EPISODE_LEN - 1:
                one_episode_perf['success_rate'] = 1
            one_episode_perf['max_goals'] = max_on_goals
            one_episode_perf['collide'] = one_episode_perf['collide'] / (
                    (one_episode_perf['episode_len'] + 1) * num_agent) * 100
            one_episode_perf['total_steps'] = step_count
            one_episode_perf['avg_steps'] = step_count / num_agent
            one_episode_perf['max_steps'] = np.max(agents_steps)
            if save_gif0:
                if not os.path.exists(RecordingParameters.GIFS_PATH):
                    os.makedirs(RecordingParameters.GIFS_PATH)
                images = np.array(episode_frames)
                make_gif(images, '{}/evaluation.gif'.format(
                    RecordingParameters.GIFS_PATH))

    return one_episode_perf, solution

def get_csv_logger(model_dir, default_model_name):
    csv_path = os.path.join(model_dir, "log-"+default_model_name+".csv")
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

if __name__ == "__main__":
    # download trained model0
    dataset_path = '/home/andrea/Thesis/baselines/Dataset/'
    map_name = '50_55_simple_warehouse'
    model_path = './final'
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    model = Model(0, torch.device('cuda:0'))
    model.network.load_state_dict(torch.load(path_checkpoint)['model'])
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    model_name = 'evaluation_custom_warehouse_SCRIMP_' + date
    model_output_name = "SCRIMP"
    csv_file, csv_logger = get_csv_logger(model_path, model_name)

    output_dir = dataset_path + map_name + "/output/" + model_output_name + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # recording
    wandb_id = wandb.util.generate_id()
    wandb.init(project='SCRIMP_evaluation',
               name='evaluation_global_SCRIMP',
               entity=RecordingParameters.ENTITY,
               notes=RecordingParameters.EXPERIMENT_NOTE,
               config=all_args,
               id=wandb_id,
               resume='allow')
    print('id is:{}'.format(wandb_id))
    print('Launching wandb...\n')
    save_gif = True

    # start evaluation
    for k in CASE:
        # remember to modify the corresponding code (size,prob) in the 'mapf_gym.py'
        episodic_buffer = EpisodicBuffer(2e6, k[0])
        env = WarehouseEnv(dataset_path, map_name, k[0], k[1])

        all_perf_dict = {'episode_len': [], 'max_goals': [], 'collide': [], 'success_rate': [], 'total_steps': [], 'avg_steps': [], 'max_steps': []}
        all_perf_dict_std = {'episode_len': [], 'max_goals': [], 'collide': [], 'total_steps': [], 'avg_steps': [], 'max_steps': []}
        print('agent: {}, world: {}'.format(k[0], k[1]))

        # Create output directory if it doesn't exist
        output_agent_dir = output_dir + str(k[0]) + "_agents" + "/"
        if not os.path.exists(output_agent_dir):
            os.makedirs(output_agent_dir)

        for j in range(NUM_TIMES):
            eval_performance_dict, solution = evaluate(env, model, torch.device('cuda:0'), episodic_buffer, k[0], j, save_gif)
            save_gif = False  # here we only record gif once
            if j % 20 == 0:
                print(j)

            for i in eval_performance_dict.keys():  # for one episode
                if i == 'episode_len':
                    if eval_performance_dict['success_rate'] == 1:
                        all_perf_dict[i].append(eval_performance_dict[i])  # only record success episode
                    else:
                        continue
                else:
                    all_perf_dict[i].append(eval_performance_dict[i])

            # Save the solution
            out = dict()
            out["finished"] = True if eval_performance_dict['success_rate'] == 1 else False
            if out["finished"]:
                out["total_step"] = eval_performance_dict['total_steps']
                out["avg_step"] = eval_performance_dict['avg_steps']
                out["max_step"] = eval_performance_dict['max_steps']
                out["episode_length"] = eval_performance_dict['episode_len']
            out["collision_rate"] = eval_performance_dict['collide']

            save_dict = {"metrics": out, "solution": solution}
            filepath = output_agent_dir + "solution_" + model_output_name + "_" + map_name + "_" + str(k[0]) + "_agents_ID_" + str(j).zfill(5) + ".npy"
            np.save(filepath, solution)

        for i in all_perf_dict.keys():  # for all episodes
            if i != 'success_rate':
                all_perf_dict_std[i] = np.std(all_perf_dict[i])
            all_perf_dict[i] = np.nanmean(all_perf_dict[i])

        print('EL: {}, MR: {}, CO: {}, SR: {}, total_steps: {}, avg_steps: {}, max_step: {}'.format(round(all_perf_dict['episode_len'], 2),
                                                    round(all_perf_dict['max_goals'], 2),
                                                    round(all_perf_dict['collide'] * 100, 2),
                                                    all_perf_dict['success_rate'] * 100,
                                                    round(all_perf_dict['total_steps'], 2),
                                                    round(all_perf_dict['avg_steps'], 2),
                                                    round(all_perf_dict['max_steps'], 2)))
        print('EL_STD: {}, MR_STD: {}, CO_STD: {}, total_step_STD: {}, avg_step_STD: {}'.format(round(all_perf_dict_std['episode_len'], 2),
                                                            round(all_perf_dict_std['max_goals'], 2),
                                                            round(all_perf_dict_std['collide'] * 100, 2),
                                                            round(all_perf_dict_std['total_steps'], 2),
                                                            round(all_perf_dict_std['avg_steps'], 2),
                                                            round(all_perf_dict_std['max_steps'], 2)))
        
        header = ["n_agents", "success_rate", "collision_rate", "total_step", "avg_step", "max_step", "episode_length", "max_goals", "collision_rate_std", "total_step_std", "avg_step_std", "max_step_std", "episode_length_std", "max_goals_std", "collision_rate_min", "total_step_min", "avg_step_min", "max_step_min", "episode_length_min", "max_goals_min", "collision_rate_max", "total_step_max", "avg_step_max", "max_step_max", "episode_length_max", "max_goals_max"]
        data = [k[0], all_perf_dict['success_rate'], all_perf_dict['collide'], all_perf_dict['total_steps'], all_perf_dict['avg_steps'], all_perf_dict['max_steps'], all_perf_dict['episode_len'], all_perf_dict['max_goals'], all_perf_dict_std['collide'], all_perf_dict_std['total_steps'], all_perf_dict_std['avg_steps'], all_perf_dict_std['max_steps'], all_perf_dict_std['episode_len'], all_perf_dict_std['max_goals'], np.min(all_perf_dict['collide']), np.min(all_perf_dict['total_steps']), np.min(all_perf_dict['avg_steps']), np.min(all_perf_dict['max_steps']), np.min(all_perf_dict['episode_len']), np.min(all_perf_dict['max_goals']), np.max(all_perf_dict['collide']), np.max(all_perf_dict['total_steps']), np.max(all_perf_dict['avg_steps']), np.max(all_perf_dict['max_steps']), np.max(all_perf_dict['episode_len']), np.max(all_perf_dict['max_goals'])]
        if k[0] == 4:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        print('-----------------------------------------------------------------------------------------------')

    print('finished')
    wandb.finish()
