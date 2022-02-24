from pathlib import Path
import argparse
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

# alt.renderers.enable('mimetype')

def get_cwd() -> Path:
    return Path.cwd()


def get_exp_name(path: Path) -> str:
    return path.parent.name


def ensure_path_exists(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_and_merge_dfs(files: List[Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(x) for x in files]
    return pd.concat(dfs, ignore_index=True)

def order_client_names(names: List[str]) -> List[str]:
    return sorted(names, key=lambda x: float(re.findall(r'\d+', x)[0]))

def plot_client_duration(df: pd.DataFrame):
    small_df = df[['round_id', 'train_duration', 'test_duration', 'round_duration', 'node_name']].melt(id_vars=['round_id', 'node_name'], var_name='type')
    ordered_clients = order_client_names(small_df['node_name'].unique())
    plt.figure()
    g = sns.FacetGrid(small_df, col="type", sharey=False)
    g.map(sns.boxplot, "node_name", "value", order=ordered_clients)
    for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()

    plt.figure()
    g = sns.FacetGrid(small_df, col="type", sharey=False, hue='node_name', hue_order=ordered_clients)
    g.map(sns.lineplot, "round_id", "value")
    for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.show()


def analyse(path: Path):
    cwd = get_cwd()
    output_path = cwd / get_exp_name(path)
    ensure_path_exists(output_path)
    all_files = [x for x in path.iterdir() if x.is_file()]
    federator_files = [x for x in all_files if 'federator' in x.name]
    client_files = [x for x in all_files if x.name.startswith('client')]

    federator_data = load_and_merge_dfs(federator_files)
    client_data = load_and_merge_dfs(client_files)

    # print(len(client_data), len(federator_data))
    plot_client_duration(client_data)
    # What do we want to plot in terms of data?


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic experiment analysis')
    parser.add_argument('path', type=str, help='Path pointing to experiment results files')
    args = parser.parse_args()
    analyse(Path(args.path))
