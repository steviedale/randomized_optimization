

def get_run_group_dfs_rhc(all_df):
    for run_i in sorted(all_df['run_number'].unique()):
        mask = all_df['run_number'] == run_i
        temp_df = all_df[mask]
        all_df.loc[mask, 'total_iterations'] = temp_df['Iteration'].max()
        all_df.loc[mask, 'total_time'] = temp_df['Time'].max()
        all_df.loc[mask, 'best_fitness'] = temp_df['Fitness'].max()
        all_df.loc[mask, 'total_fevals'] = temp_df['FEvals'].max()
        all_df.loc[mask, 'total_restarts'] = temp_df['current_restart'].max()
    run_df = all_df[[
        'run_number', 'group_number', 'problem_size', 'max_iterations', 'max_attempts', 'max_restarts',
        'total_iterations', 'total_time', 'best_fitness', 'total_fevals', 'total_restarts']]
    run_df.drop_duplicates(inplace=True)

    for group_i in sorted(run_df['group_number'].unique()):
        mask = run_df['group_number'] == group_i
        temp_df = run_df[mask]
        for key in 'total_iterations', 'total_time', 'best_fitness', 'total_fevals', 'total_restarts':
            run_df.loc[mask, f"{key}_mean"] = temp_df[key].mean()
            run_df.loc[mask, f"{key}_std"] = temp_df[key].std()
    group_df = run_df[[
        'group_number', 'problem_size', 'max_iterations', 'max_attempts', 'max_restarts',
        'total_iterations_mean', 'total_iterations_std', 'total_time_mean', 'total_time_std', 'best_fitness_mean',
        'best_fitness_std', 'total_fevals_mean', 'total_fevals_std', 'total_restarts_mean', 'total_restarts_std']]
    group_df.drop_duplicates(inplace=True)

    return run_df, group_df


def get_run_group_dfs_sa(all_df):
    for run_i in sorted(all_df['run_number'].unique()):
        mask = all_df['run_number'] == run_i
        temp_df = all_df[mask]
        all_df.loc[mask, 'total_iterations'] = temp_df['Iteration'].max()
        all_df.loc[mask, 'total_time'] = temp_df['Time'].max()
        all_df.loc[mask, 'best_fitness'] = temp_df['Fitness'].max()
        all_df.loc[mask, 'total_fevals'] = temp_df['FEvals'].max()
    run_df = all_df[[
        'run_number', 'group_number', 'problem_size', 'max_iterations', 'max_attempts', 'start_temperature', 'decay_type',
        'total_iterations', 'total_time', 'best_fitness', 'total_fevals']]
    run_df.drop_duplicates(inplace=True)
    len(run_df)

    for group_i in sorted(run_df['group_number'].unique()):
        mask = run_df['group_number'] == group_i
        temp_df = run_df[mask]
        for key in 'total_iterations', 'total_time', 'best_fitness', 'total_fevals':
            run_df.loc[mask, f"{key}_mean"] = temp_df[key].mean()
            run_df.loc[mask, f"{key}_std"] = temp_df[key].std()
    group_df = run_df[[
        'group_number', 'problem_size', 'max_iterations', 'max_attempts', 'start_temperature', 'decay_type',
        'total_iterations_mean', 'total_iterations_std', 'total_time_mean', 'total_time_std', 'best_fitness_mean',
        'best_fitness_std', 'total_fevals_mean', 'total_fevals_std']]
    group_df.drop_duplicates(inplace=True)
    len(group_df)

    return run_df, group_df


def get_run_group_dfs_mimic(all_df):
    for run_i in sorted(all_df['run_number'].unique()):
        mask = all_df['run_number'] == run_i
        temp_df = all_df[mask]
        all_df.loc[mask, 'total_iterations'] = temp_df['Iteration'].max()
        all_df.loc[mask, 'total_time'] = temp_df['Time'].max()
        all_df.loc[mask, 'best_fitness'] = temp_df['Fitness'].max()
        all_df.loc[mask, 'total_fevals'] = temp_df['FEvals'].max()
    run_df = all_df[[
        'run_number', 'group_number', 'problem_size', 'max_iterations', 'max_attempts', 'population_size', 'keep_percent',
        'total_iterations', 'total_time', 'best_fitness', 'total_fevals']]
    run_df.drop_duplicates(inplace=True)
    len(run_df)

    for group_i in sorted(run_df['group_number'].unique()):
        mask = run_df['group_number'] == group_i
        temp_df = run_df[mask]
        for key in 'total_iterations', 'total_time', 'best_fitness', 'total_fevals':
            run_df.loc[mask, f"{key}_mean"] = temp_df[key].mean()
            run_df.loc[mask, f"{key}_std"] = temp_df[key].std()
    group_df = run_df[[
        'group_number', 'problem_size', 'max_iterations', 'max_attempts', 'population_size', 'keep_percent',
        'total_iterations_mean', 'total_iterations_std', 'total_time_mean', 'total_time_std', 'best_fitness_mean',
        'best_fitness_std', 'total_fevals_mean', 'total_fevals_std']]
    group_df.drop_duplicates(inplace=True)
    len(group_df)

    return run_df, group_df