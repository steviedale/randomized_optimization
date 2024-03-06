# %%
import pandas as pd
from tqdm import tqdm
import os
import yaml

import sys
sys.path.append('..')
from mlrose_hiive import TSPGenerator
from mlrose_hiive import GARunner

# %%
SEED = 0
MUTATION_RATE_LIST = [0.25, 0.5, 0.75, 0.9]

PROBLEM_SIZE_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['problem_size']]
ITERATIONS_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['iterations']]
MAX_ATTEMPTS_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['attempts']]
NUM_RUNS = yaml.load(open('parameters.yml'), yaml.Loader)['num_runs']
POPULATION_SIZES_LIST = [yaml.load(open('parameters.yml'), yaml.Loader)['population_size']]

assert(len(PROBLEM_SIZE_LIST) == 1)
EXPERIMENT_NAME = f'TSP_GA'

# %%
output_dir = f'metrics_size={PROBLEM_SIZE_LIST[0]}_iters={ITERATIONS_LIST[0]}_pop={POPULATION_SIZES_LIST[0]}_attempts={MAX_ATTEMPTS_LIST[0]}/{EXPERIMENT_NAME}'
os.makedirs(output_dir, exist_ok=True)

# %%
all_df = pd.DataFrame()
group_i = 0
run_i = 0
for problem_size in PROBLEM_SIZE_LIST:
    print(f'Problem Size: {problem_size}')
    for iterations in ITERATIONS_LIST:
        print(f'Iterations: {iterations}')
        for max_attempts in MAX_ATTEMPTS_LIST:
            print(f'Max Attempts: {max_attempts}')
            for population_size in POPULATION_SIZES_LIST:
                print(f'Population Size: {population_size}')
                for mutation_rate in MUTATION_RATE_LIST:
                    print(f"Mutation Rate: {mutation_rate}")
                    for i in tqdm(range(NUM_RUNS)):
                        problem = TSPGenerator().generate(seed=SEED+i, number_of_cities=problem_size)
                        sa = GARunner(
                            problem=problem,
                            experiment_name='dontcare',
                            output_directory='./',
                            seed=SEED+i,
                            max_attempts=max_attempts,
                            iteration_list=[iterations],
                            population_sizes=[population_size],
                            mutation_rates=[mutation_rate]
                        )
                        _, df_run_curves = sa.run()
                        df_run_curves['group_number'] = group_i
                        df_run_curves['run_number'] = run_i
                        df_run_curves['problem_size'] = problem_size
                        df_run_curves['max_iterations'] = iterations
                        df_run_curves['max_attempts'] = max_attempts
                        df_run_curves['population_size'] = population_size
                        df_run_curves['mutation_rate'] = mutation_rate

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

# %%



