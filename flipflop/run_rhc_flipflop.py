# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import yaml

import sys
sys.path.append('..')
from mlrose_hiive import FlipFlopGenerator
from mlrose_hiive import RHCRunner

# %%
GENERATOR = FlipFlopGenerator
SEED = 0
PROBLEM_SIZE_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['problem_size']]
ITERATIONS_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['iterations']]
MAX_ATTEMPTS_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['attempts']]
NUM_RUNS = yaml.load(open('parameters.yml'), yaml.Loader)['num_runs']
POPULATION_SIZES_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['population_size']]
RESTART_LIST = [0, 25, 50, 75, 100]

assert(len(PROBLEM_SIZE_LIST) == 1)
EXPERIMENT_NAME = f'FlipFlop_{PROBLEM_SIZE_LIST[0]}_iters={ITERATIONS_LIST[0]}_pop={POPULATION_SIZES_LIST[0]}_att={MAX_ATTEMPTS_LIST[0]}_RHC'

# %%
output_dir = f'metrics/{EXPERIMENT_NAME}'
os.makedirs(output_dir, exist_ok=True)

# %%
all_df = pd.DataFrame()
group_i = 0
run_i = 0
for problem_size in PROBLEM_SIZE_LIST:
    for iterations in ITERATIONS_LIST:
        for max_attempts in MAX_ATTEMPTS_LIST:
            for restarts in RESTART_LIST:
                for i in tqdm(range(NUM_RUNS)):
                    problem = FlipFlopGenerator().generate(seed=SEED+i, size=problem_size)
                    sa = RHCRunner(
                        problem=problem,
                        experiment_name='dontcare',
                        output_directory='./',
                        seed=SEED,
                        max_attempts=max_attempts,
                        iteration_list=[iterations],
                        restart_list=[restarts],
                    )
                    _, df_run_curves = sa.run()
                    df_run_curves['group_number'] = group_i
                    df_run_curves['run_number'] = run_i
                    df_run_curves['problem_size'] = problem_size
                    df_run_curves['max_iterations'] = iterations
                    df_run_curves['max_attempts'] = max_attempts
                    df_run_curves['max_restarts'] = restarts

                    print(f"Max Fitness: {df_run_curves['Fitness'].max()}")

                    all_df = pd.concat([all_df, df_run_curves], axis=0)
                    run_i += 1
                group_i += 1
all_df.reset_index(inplace=True, drop=True)

# %%
all_df['Fitness'].min(), all_df['Fitness'].max()

# %%
all_df.to_csv(os.path.join(output_dir, 'learning_curve.csv'), index=False)


