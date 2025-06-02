from util import reset_env, make_gif, set_global_seeds
import datetime
import csv
import json
import numpy as np
import torch
import wandb
import os
from alg_parameters import *
from episodic_buffer import EpisodicBuffer
from mapf_gym import WarehouseEnv
from model import Model
from util import reset_env, make_gif, set_global_seeds
import csv
from pathlib import Path

NUM_TIMES = 200
# CASE = [[8, 10, 0], [8, 10, 0.15], [8, 10, 0.3], [16, 20, 0.0], [16, 20, 0.15], [16, 20, 0.3], [32, 30, 0.0],
#         [32, 30, 0.15], [32, 30, 0.3], [64, 40, 0.0], [64, 40, 0.15], [64, 40, 0.3], [128, 40, 0.0],
#         [128, 40, 0.15], [128, 40, 0.3]]
# CASE  = [[4, 50], [8, 50], [16, 50], [32, 50], [64, 50], [128, 50], [256, 50]] # [num_agents, env_size]
CASE  = [[128, 50], [256, 50]] # [num_agents, env_size]
set_global_seeds(SetupParameters.SEED)

def count_collisions(solution, obstacle_map):
    """Count agent-agent and obstacle collisions in the solution."""
    agent_agent_collisions = 0
    obstacle_collisions = 0
    num_agents = 0
    
    # Convert solution format to timestep-based format
    timestep_based_solution = []
    if len(solution) > 0:
        # Find max timestep
        max_timestep = 0
        for agent_path in solution:
            for pos in agent_path:
                max_timestep = max(max_timestep, pos[2])
                
        num_agents = len(solution)
        # Initialize timestep-based solution
        timestep_based_solution = [[] for _ in range(max_timestep + 1)]
        
        # Fill in the positions for each timestep
        for agent_idx, agent_path in enumerate(solution):
            positions = {}  # Dictionary to store position at each timestep
            for pos in agent_path:
                positions[pos[2]] = (pos[0], pos[1])
            
            # Ensure every timestep has a position
            for t in range(max_timestep + 1):
                if t in positions:
                    timestep_based_solution[t].append(positions[t])
                elif t > 0 and t-1 in positions:
                    # If missing, use previous position
                    timestep_based_solution[t].append(positions[t-1])
                else:
                    # Should not happen in proper solutions
                    timestep_based_solution[t].append((-1, -1))
    
    # Now count collisions
    for timestep in range(len(timestep_based_solution)):
        positions_at_timestep = timestep_based_solution[timestep]
        current_agent_positions = []
        
        for agent_idx in range(len(positions_at_timestep)):
            agent_pos = positions_at_timestep[agent_idx]
            
            # Check for obstacle collisions
            if agent_pos[0] >= 0 and agent_pos[1] >= 0:  # Valid position
                if agent_pos[0] < obstacle_map.shape[0] and agent_pos[1] < obstacle_map.shape[1]:
                    if obstacle_map[agent_pos[0], agent_pos[1]] == -1:
                        obstacle_collisions += 1
            
            # Prepare for agent-agent collision check
            current_agent_positions.append(agent_pos)
        
        # Agent-agent collision check
        for i in range(len(current_agent_positions)):
            for j in range(i+1, len(current_agent_positions)):
                if current_agent_positions[i] == current_agent_positions[j]:
                    # Count collision for both agents involved
                    agent_agent_collisions += 2
    
    return agent_agent_collisions, obstacle_collisions

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
    import time
    
    one_episode_perf = {'episode_len': 0, 'max_goals': 0, 'collide': 0, 'success_rate': 0, 
                       'total_steps': 0, 'avg_steps': 0, 'max_steps': 0, 'min_steps': 0,
                       'total_costs': 0, 'avg_costs': 0, 'max_costs': 0, 'min_costs': 0,
                       'time': 0, 'crashed': False, 'agent_coll_rate': 0, 'obstacle_coll_rate': 0, 'total_coll_rate': 0}
    episode_frames = []

    start_time = time.time()
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
    agents_costs = np.zeros(num_agent)
    
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

        # Update the agents steps and costs
        for i in range(num_agent):
            if actions[i] != 0:
                agents_steps[i] += 1
            # Each agent pays cost until reaching goal
            if not on_goal[i]:
                agents_costs[i] = episode_length + 1

        new_xy = eval_env.get_positions()
        processed_rewards, _, intrinsic_reward, min_dist = episodic_buffer0.if_reward(new_xy, rewards, done, on_goal)

        vector[:, :, 3] = rewards
        vector[:, :, 4] = intrinsic_reward
        vector[:, :, 5] = min_dist

        if save_gif0:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        if done:
            one_episode_perf['time'] = time.time() - start_time
            if one_episode_perf['episode_len'] < EnvParameters.EPISODE_LEN - 1:
                one_episode_perf['success_rate'] = 1
            else:
                one_episode_perf['success_rate'] = 0
                
            one_episode_perf['max_goals'] = max_on_goals
            one_episode_perf['total_steps'] = step_count
            one_episode_perf['avg_steps'] = step_count / num_agent
            one_episode_perf['max_steps'] = np.max(agents_steps)
            one_episode_perf['min_steps'] = np.min(agents_steps)
            one_episode_perf['total_costs'] = np.sum(agents_costs)
            one_episode_perf['avg_costs'] = np.mean(agents_costs)
            one_episode_perf['max_costs'] = np.max(agents_costs)
            one_episode_perf['min_costs'] = np.min(agents_costs)
            
            # Calculate collision rates
            if episode_length > 0 and num_agent > 0:
                agent_coll, obs_coll = count_collisions(solution, eval_env.get_obstacle_map())
                one_episode_perf['crashed'] = (agent_coll + obs_coll) > 0
                one_episode_perf['agent_coll_rate'] = agent_coll / (episode_length * num_agent)
                one_episode_perf['obstacle_coll_rate'] = obs_coll / (episode_length * num_agent)
                one_episode_perf['total_coll_rate'] = (agent_coll + obs_coll) / (episode_length * num_agent)
                one_episode_perf['collide'] = one_episode_perf['total_coll_rate']
            else:
                one_episode_perf['crashed'] = False
                one_episode_perf['agent_coll_rate'] = 0
                one_episode_perf['obstacle_coll_rate'] = 0
                one_episode_perf['total_coll_rate'] = 0
                one_episode_perf['collide'] = 0
                
            if save_gif0:
                make_gif(episode_frames, str(eval_env.num_agents) + 'agents_' + str(case_id) + '.gif')

    return one_episode_perf, solution

def get_csv_logger(model_dir, default_model_name):
    csv_path = Path(model_dir) / f"log-{default_model_name}.csv"
    create_folders_if_necessary(csv_path)
    csv_file = csv_path.open("a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path):
    dirname = path.parent
    if not dirname.is_dir():
        dirname.mkdir(parents=True)

if __name__ == "__main__":
    print("Starting evaluation script...")
    print("Python path check passed")
    
    # download trained model0
    print("Setting up paths...")
    base_path = Path(__file__).parent
    dataset_path = base_path.parent / 'Dataset'
    model_path = base_path / 'model'
    path_checkpoint = model_path / "net_checkpoint.pkl"
    
    # Check if files exist before proceeding
    print(f"Checking if checkpoint exists: {path_checkpoint}")
    if not path_checkpoint.exists():
        print(f"ERROR: Checkpoint file not found at {path_checkpoint}")
        exit(1)
    print("Checkpoint file exists!")
    
    print(f"Checking if dataset path exists: {dataset_path}")
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found at {dataset_path}")
        exit(1)
    print("Dataset path exists!")
    
    print("Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    print(f"Loading model from: {path_checkpoint}")
    print("Creating model...")
    
    try:
        # Try with CUDA first, fallback to CPU if needed
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        model = Model(0, device)
        print("Model created successfully!")
    except Exception as e:
        print(f"Error creating model: {e}")
        print("Trying with CPU...")
        try:
            device = torch.device('cpu')
            model = Model(0, device)
            print("Model created successfully with CPU!")
        except Exception as e2:
            print(f"Error creating model with CPU: {e2}")
            exit(1)
    
    print("Loading model state dict...")
    try:
        checkpoint = torch.load(path_checkpoint, map_location=device)
        print("Checkpoint loaded successfully!")
        model.network.load_state_dict(checkpoint['model'])
        print("Model state dict loaded successfully!")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        exit(1)
    
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    model_name = 'evaluation_custom_warehouse_SCRIMP_' + date
    model_output_name = "SCRIMP"
    print(f"Model name: {model_name}")

    # Map configurations for testing
    map_configurations = [
        # {
        #     "map_name": "15_15_simple_warehouse",
        #     "size": 15,
        #     "n_tests": 200,  # Reduced for testing
        #     "list_num_agents": [4, 8, 12, 16, 20, 22]  # Reduced for testing
        # },
        # {
        #     "map_name": "50_55_simple_warehouse", 
        #     "size": 50,
        #     "n_tests": 200,
        #     "list_num_agents": [4, 8, 16, 32, 64, 128, 256]
        # },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128, 256]
        },
    ]

    header = ["n_agents", 
              "success_rate", "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_step", "total_step_std", "total_step_min", "total_step_max",
              "avg_step", "avg_step_std", "avg_step_min", "avg_step_max",
              "max_step", "max_step_std", "max_step_min", "max_step_max",
              "min_step", "min_step_std", "min_step_min", "min_step_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]

    # Skip wandb for now to isolate the issue
    print("Skipping wandb initialization for debugging...")
    # try:
    #     wandb_id = wandb.util.generate_id()
    #     print(f"Generated wandb ID: {wandb_id}")
    #     
    #     wandb.init(project='SCRIMP_evaluation',
    #                name='evaluation_global_SCRIMP',
    #                entity=RecordingParameters.ENTITY,
    #                notes=RecordingParameters.EXPERIMENT_NOTE,
    #                config=all_args,
    #                id=wandb_id,
    #                resume='allow')
    #     print('id is:{}'.format(wandb_id))
    #     print('Launching wandb...\n')
    # except Exception as e:
    #     print(f"Error initializing wandb: {e}")
    #     print("Continuing without wandb...")

    print("Starting main evaluation loop...")
    
    # Process each map configuration
    for config_idx, config in enumerate(map_configurations):
        print(f"Processing configuration {config_idx + 1}/{len(map_configurations)}")
        
        map_name = config["map_name"]
        size = config["size"]
        n_tests = config["n_tests"]
        list_num_agents = config["list_num_agents"]

        print(f"\nProcessing map: {map_name}")
        
        # Check if map exists
        map_path = dataset_path / map_name
        print(f"Checking map path: {map_path}")
        if not map_path.exists():
            print(f"WARNING: Map path does not exist: {map_path}")
            print(f"Available maps in {dataset_path}:")
            try:
                available_maps = list(dataset_path.iterdir())
                for available_map in available_maps:
                    print(f"  - {available_map.name}")
            except Exception as e:
                print(f"Error listing maps: {e}")
            continue
        print("Map path exists!")
        
        # Create output directory for results
        output_dir = map_path / "output" / model_output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup CSV logger
        results_path = base_path / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
        csv_filename_base = f'{model_output_name}_{sanitized_map_name}_{date}'
        csv_file, csv_logger = get_csv_logger(results_path, csv_filename_base)

        csv_logger.writerow(header)
        csv_file.flush()

        # start evaluation
        for num_agents in list_num_agents:
            print(f"Starting tests for {num_agents} agents on map {map_name}")
            
            # remember to modify the corresponding code (size,prob) in the 'mapf_gym.py'
            episodic_buffer = EpisodicBuffer(2e6, num_agents)
            env = WarehouseEnv(dataset_path, map_name, num_agents, size)

            # Create output directory if it doesn't exist
            output_agent_dir = output_dir / f"{num_agents}_agents"
            output_agent_dir.mkdir(parents=True, exist_ok=True)

            # Initialize result storage
            results = {
                'finished': [], 'time': [], 'episode_length': [],
                'total_steps': [], 'avg_steps': [], 'max_steps': [], 'min_steps': [],
                'total_costs': [], 'avg_costs': [], 'max_costs': [], 'min_costs': [],
                'crashed': [], 'agent_coll_rate': [], 'obstacle_coll_rate': [], 'total_coll_rate': []
            }

            save_gif = False
            for j in range(n_tests):
                eval_performance_dict, solution = evaluate(env, model, torch.device('cuda:0'), episodic_buffer, num_agents, j, save_gif)
                save_gif = False  # here we only record gif once
                if j % 20 == 0:
                    print(j)

                # Save results
                results['finished'].append(eval_performance_dict['success_rate'])
                if eval_performance_dict['success_rate']:
                    results['time'].append(eval_performance_dict['time'])
                    results['episode_length'].append(eval_performance_dict['episode_len'])
                    results['total_steps'].append(eval_performance_dict['total_steps'])
                    results['avg_steps'].append(eval_performance_dict['avg_steps'])
                    results['max_steps'].append(eval_performance_dict['max_steps'])
                    results['min_steps'].append(eval_performance_dict['min_steps'])
                    results['total_costs'].append(eval_performance_dict['total_costs'])
                    results['avg_costs'].append(eval_performance_dict['avg_costs'])
                    results['max_costs'].append(eval_performance_dict['max_costs'])
                    results['min_costs'].append(eval_performance_dict['min_costs'])
                    results['agent_coll_rate'].append(eval_performance_dict['agent_coll_rate'])
                    results['obstacle_coll_rate'].append(eval_performance_dict['obstacle_coll_rate'])
                    results['total_coll_rate'].append(eval_performance_dict['total_coll_rate'])
                    results['crashed'].append(eval_performance_dict['crashed'])

                # Save solution to file
                out = dict()
                out["finished"] = eval_performance_dict['success_rate'] == 1
                if out["finished"]:
                    out["time"] = eval_performance_dict['time']
                    out["episode_length"] = eval_performance_dict['episode_len']
                    out["total_step"] = eval_performance_dict['total_steps']
                    out["avg_step"] = eval_performance_dict['avg_steps']
                    out["max_step"] = eval_performance_dict['max_steps']
                    out["min_step"] = eval_performance_dict['min_steps']
                    out["total_costs"] = eval_performance_dict['total_costs']
                    out["avg_costs"] = eval_performance_dict['avg_costs']
                    out["max_costs"] = eval_performance_dict['max_costs']
                    out["min_costs"] = eval_performance_dict['min_costs']
                    out["agent_coll_rate"] = eval_performance_dict['agent_coll_rate']
                    out["obstacle_coll_rate"] = eval_performance_dict['obstacle_coll_rate']
                    out["total_coll_rate"] = eval_performance_dict['total_coll_rate']
                    out["crashed"] = eval_performance_dict['crashed']
                out["collision_rate"] = eval_performance_dict['collide']

                solution_filepath = output_agent_dir / f"solution_{model_output_name}_{map_name}_{num_agents}_agents_ID_{str(j).zfill(3)}.txt"
                with open(solution_filepath, 'w') as f:
                    f.write("Metrics:\n")
                    json.dump(out, f, indent=4)
                    f.write("\n\nSolution:\n")
                    if solution:
                        for agent_path in solution:
                            f.write(f"{agent_path}\n")
                    else:
                        f.write("No solution found.\n")

            # Calculate aggregated metrics
            final_results = {}
            final_results['finished'] = np.sum(results['finished']) / len(results['finished']) if len(results['finished']) > 0 else 0

            print('SR: {:.2f}%, EL: {}, TO: {}, AV: {}, MA: {}, CO: {:.2f}%'.format(
                final_results['finished'] * 100,
                np.mean(results['episode_length']) if results['episode_length'] else 0,
                np.mean(results['total_steps']) if results['total_steps'] else 0,
                np.mean(results['avg_steps']) if results['avg_steps'] else 0,
                np.mean(results['max_steps']) if results['max_steps'] else 0,
                np.mean(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0))

            # Write results to CSV
            data = [num_agents,
                    final_results['finished'] * 100,  # convert to percentage
                    np.mean(results['time']) if results['time'] else 0,
                    np.std(results['time']) if results['time'] else 0,
                    np.min(results['time']) if results['time'] else 0,
                    np.max(results['time']) if results['time'] else 0,
                    np.mean(results['episode_length']) if results['episode_length'] else 0,
                    np.std(results['episode_length']) if results['episode_length'] else 0,
                    np.min(results['episode_length']) if results['episode_length'] else 0,
                    np.max(results['episode_length']) if results['episode_length'] else 0,
                    np.mean(results['total_steps']) if results['total_steps'] else 0,
                    np.std(results['total_steps']) if results['total_steps'] else 0,
                    np.min(results['total_steps']) if results['total_steps'] else 0,
                    np.max(results['total_steps']) if results['total_steps'] else 0,
                    np.mean(results['avg_steps']) if results['avg_steps'] else 0,
                    np.std(results['avg_steps']) if results['avg_steps'] else 0,
                    np.min(results['avg_steps']) if results['avg_steps'] else 0,
                    np.max(results['avg_steps']) if results['avg_steps'] else 0,
                    np.mean(results['max_steps']) if results['max_steps'] else 0,
                    np.std(results['max_steps']) if results['max_steps'] else 0,
                    np.min(results['max_steps']) if results['max_steps'] else 0,
                    np.max(results['max_steps']) if results['max_steps'] else 0,
                    np.mean(results['min_steps']) if results['min_steps'] else 0,
                    np.std(results['min_steps']) if results['min_steps'] else 0,
                    np.min(results['min_steps']) if results['min_steps'] else 0,
                    np.max(results['min_steps']) if results['min_steps'] else 0,
                    np.mean(results['total_costs']) if results['total_costs'] else 0,
                    np.std(results['total_costs']) if results['total_costs'] else 0,
                    np.min(results['total_costs']) if results['total_costs'] else 0,
                    np.max(results['total_costs']) if results['total_costs'] else 0,
                    np.mean(results['avg_costs']) if results['avg_costs'] else 0,
                    np.std(results['avg_costs']) if results['avg_costs'] else 0,
                    np.min(results['avg_costs']) if results['avg_costs'] else 0,
                    np.max(results['avg_costs']) if results['avg_costs'] else 0,
                    np.mean(results['max_costs']) if results['max_costs'] else 0,
                    np.std(results['max_costs']) if results['max_costs'] else 0,
                    np.min(results['max_costs']) if results['max_costs'] else 0,
                    np.max(results['max_costs']) if results['max_costs'] else 0,
                    np.mean(results['min_costs']) if results['min_costs'] else 0,
                    np.std(results['min_costs']) if results['min_costs'] else 0,
                    np.min(results['min_costs']) if results['min_costs'] else 0,
                    np.max(results['min_costs']) if results['min_costs'] else 0,
                    np.mean(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,  # convert to percentage
                    np.std(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.min(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.max(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.mean(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,  # convert to percentage
                    np.std(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.min(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.max(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.mean(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,  # convert to percentage
                    np.std(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.min(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.max(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0
                   ]
            csv_logger.writerow(data)
            csv_file.flush()

            print('-----------------------------------------------------------------------------------------------')

        csv_file.close()

    print('finished')
    # wandb.finish()
