# %%
import pandas as pd
from tqdm import tqdm
import os
import yaml

import sys
sys.path.append('..')
from mlrose_hiive import QueensGenerator
from mlrose_hiive import SARunner
import mlrose_hiive

# %%
SEED = 0
TEMPERATURE_LIST = [1e0, 1e1, 1e2, 1e3, 1e4]
DECAY_LIST = {
    'GeomDecay': mlrose_hiive.GeomDecay, 
    # 'ExpDecay': mlrose_hiive.ExpDecay, 
    # 'ArithDecay': mlrose_hiive.ArithDecay,
}

PROBLEM_SIZE_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['problem_size']]
ITERATIONS_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['iterations'] * 10]
MAX_ATTEMPTS_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['attempts']]
NUM_RUNS = yaml.load(open('parameters.yml'), yaml.Loader)['num_runs']

assert(len(PROBLEM_SIZE_LIST) == 1)
EXPERIMENT_NAME = f'Queens_{PROBLEM_SIZE_LIST[0]}_SA'

output_dir = f'metrics/{EXPERIMENT_NAME}'
os.makedirs(output_dir, exist_ok=True)

# %%
all_df = pd.DataFrame()
group_i = 0
run_i = 0
for problem_size in PROBLEM_SIZE_LIST:
    print(f'Problem Size: {problem_size}')
    for max_iterations in ITERATIONS_LIST:
        print(f'Max Iterations: {max_iterations}')
        for max_attempts in MAX_ATTEMPTS_LIST:
            print(f'Max Attempts: {max_attempts}')
            for start_temperature in TEMPERATURE_LIST:
                print(f"Start Temperature: {start_temperature}")
                for decay_str, decay_cls in DECAY_LIST.items():
                    print(f"Decay type: {decay_str}")
                    for i in tqdm(range(NUM_RUNS)):
                        problem = QueensGenerator().generate(seed=SEED+i, size=problem_size)
                        sa = SARunner(
                            problem=problem,
                            experiment_name='dontcare',
                            output_directory='./',
                            seed=SEED+i,
                            iteration_list=[max_iterations],
                            max_attempts=max_attempts,
                            temperature_list=[start_temperature],
                            decay_list=[decay_cls],
                        )
                        _, df_run_curves = sa.run()
                        df_run_curves['group_number'] = group_i
                        df_run_curves['run_number'] = run_i
                        df_run_curves['problem_size'] = problem_size
                        df_run_curves['max_iterations'] = max_iterations
                        df_run_curves['max_attempts'] = max_attempts
                        df_run_curves['start_temperature'] = start_temperature
                        df_run_curves['decay_type'] = decay_str

                        print(f"Max Fitness: {df_run_curves['Fitness'].max()}")
                        print(f"Max Iteration: {df_run_curves['Iteration'].max()}")

                        all_df = pd.concat([all_df, df_run_curves], axis=0)
                        run_i += 1
                    group_i += 1
all_df.reset_index(inplace=True, drop=True)

# %%
print(f"Max: {all_df['Fitness'].max()}")
print(f"Min: {all_df['Fitness'].min()}")

# %%
all_df.to_csv(os.path.join(output_dir, 'learning_curve.csv'), index=False)


