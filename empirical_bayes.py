import numpy as np
import pandas as pd
import seaborn as sns


def multi_sample_size_js_estimator(df, group_id_col, data_col, pooled):
    """
    This is a plug-and-play function that duplicates logic in simulations.ipynb 

    :param df: A DataFrame with a group ID column and a data column. Each row
        represents an observation.
    :param group_id_col: The name of the group ID column
    :param data_col: The name of the data column.
    :param pooled: A boolean flag indicating whether you want to pool your estimate of
        group variances or estimate them separately.
    :returns: a data frame containing some statistics for each group. The `mean`
        column represents the MLE estimate of the group mean. The `theta_hat_js`
        column represents the MSS James-Stein estimate. The `theta_hat_jsp`
        column represents the MSS James-Stein pooled estimate.
    """

    df = df.assign(
        mean = df.groupby(group_id_col)[data_col].transform(np.mean),
        n = df.groupby(group_id_col)[data_col].transform(len))

    stats = df.groupby(group_id_col)[data_col].agg({
        'mean': np.mean,
        'n': len,
        'dof': lambda x: max(len(x) - 1, 1),
        'std2': np.var})
    
    # n=1 edge case.
    default_std2 = (df.query('n > 1')[data_col] - df.query('n > 1')['mean']).var()
    stats.loc[stats['n'] < 2, 'std2'] = default_std2
    stats.loc[stats['n'] < 2, 'dof'] = 1
    
    stats = stats.assign(
        btw_group_std2 = df.groupby(group_id_col)[data_col].mean().var(),
        global_mean = stats['mean'].mean(),
        pooled_std2 = (stats['std2'] * stats['dof']).sum() / (stats['dof'].sum()))
        
    stats = stats.assign(
        sem2 = stats['std2'] / stats['n'],
        pooled_sem2 = stats['pooled_std2'] / stats['n'])

    if pooled:
        stats = stats.assign(
            B_hat_jsp = (stats['pooled_sem2'] / stats['btw_group_std2']).clip(0,1))
        stats = stats.assign(
            theta_hat_jsp = stats['B_hat_jsp'] * stats['global_mean'] + (1 - stats['B_hat_jsp']) * stats['mean'])

    else:    
        stats = stats.assign(
            B_hat_js = (stats['sem2'] / stats['btw_group_std2']).clip(0,1))
        stats = stats.assign(
            theta_hat_js = stats['B_hat_js'] * stats['global_mean'] + (1 - stats['B_hat_js']) * stats['mean'])

    return stats


# Create some data with unequal group sizes.
iris = sns.load_dataset("iris")
df = pd.concat([
    iris.query('species == "setosa"').head(5),
    iris.query('species == "versicolor"').head(7),
    iris.query('species == "virginica"').head(4)], axis=0)

# Multi Sample Size James-Stein Estimator
stats_mss_js = multi_sample_size_js_estimator(df, group_id_col='species', data_col='sepal_width', pooled=False)

# Multi Sample Size Pooled James-Stein Estimator
stats_mss_js_pooled = multi_sample_size_js_estimator(df, group_id_col='species', data_col='sepal_width', pooled=True)

