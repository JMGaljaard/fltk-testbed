from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_data(df: pd.DataFrame, name: str, meta_data: dict):
    # type = 'feature'
    first_cls_layer = int(meta_data[name])
    df['type'] = 'feature'
    mask = df['layer_id'] >= first_cls_layer
    df.loc[mask, 'type'] = 'classifier'
    mask = df['layer_id'] < first_cls_layer
    df.loc[mask, 'type'] = 'feature'

    tmp = df.groupby(['execution_id', 'event', 'id_type_combined', 'layer_id', 'type']).time.mean().reset_index()
    sorted = tmp.sort_values(['event', 'execution_id'], ascending=[False, True])

    grouped_df = tmp.groupby(['event', 'type']).sum().reset_index()[['event', 'type', 'time']]
    grouped_df['model'] = name
    # for idx, row in df.iterrows():
    #     print(idx, row)
    return grouped_df


def parse_stability_data(data: List[pd.DataFrame], save_to_file: bool = False, filename: str = 'stability_data.csv'):
    df_list = []
    for idx, df in enumerate(data):
        df['idx'] = idx
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    if save_to_file:
        combined_df.to_csv(filename)
    return combined_df


def stability_plot(df: pd.DataFrame):
    # for idx, df in enumerate(data):
    #     print(idx)
    pass

def calc_metric(df, start_cls_layer):
    df['type'] = 'feature'
    mask = df['layer_id'] >= start_cls_layer
    df.loc[mask, 'type'] = 'classifier'
    mask = df['layer_id'] < start_cls_layer
    df.loc[mask, 'type'] = 'feature'
    combined = df.groupby(['event', 'type', 'idx']).sum().reset_index()

    features_f: pd.DataFrame = combined[(combined['type'] == 'feature') & (combined['event'] == 'forward')][['time', 'idx']]
    classifier_f = combined[(combined['type'] == 'classifier') & (combined['event'] == 'forward')][['time', 'idx']]
    features_b = combined[(combined['type'] == 'feature') & (combined['event'] == 'backward')][['time', 'idx']]
    classifier_b = combined[(combined['type'] == 'classifier') & (combined['event'] == 'backward')][['time', 'idx']]

    features_f2: pd.DataFrame = combined[(combined['type'] == 'feature') & (combined['event'] == 'forward')]
    classifier_f2 = combined[(combined['type'] == 'classifier') & (combined['event'] == 'forward')]
    features_b2 = combined[(combined['type'] == 'feature') & (combined['event'] == 'backward')]
    classifier_b2 = combined[(combined['type'] == 'classifier') & (combined['event'] == 'backward')]

    plt.figure()
    # sns.lineplot(data=pd.concat([features_b2, features_f2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time', hue='type')
    sns.lineplot(data=pd.concat([features_f2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time', hue='event')
    plt.title('Weak offloaded Client')
    plt.show()

    plt.figure()
    # sns.lineplot(data=pd.concat([features_b2, features_f2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time', hue='type')
    sns.lineplot(data=pd.concat([features_f2, features_b2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time',
                 hue='event')
    plt.title('Original Weak client')
    plt.show()
    plt.figure()
    sns.lineplot(data=pd.concat([features_f2, features_b2], ignore_index=True), x='idx', y='time', hue='event')
    plt.title('Offload')
    plt.show()

    plt.figure()
    # sns.lineplot(data=pd.concat([features_b2, features_f2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time', hue='type')
    sns.lineplot(data=pd.concat([features_f2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time')
    plt.title('Weak offloaded Client #2')
    plt.show()

    plt.figure()
    # sns.lineplot(data=pd.concat([features_b2, features_f2, classifier_b2, classifier_f2], ignore_index=True), x='idx', y='time', hue='type')
    sns.lineplot(data=pd.concat([features_f2, features_b2, classifier_b2, classifier_f2], ignore_index=True), x='idx',
                 y='time')
    plt.title('Original Weak client #2')
    plt.show()
    plt.figure()
    sns.lineplot(data=pd.concat([features_f2, features_b2], ignore_index=True), x='idx', y='time')
    plt.title('Offload #2')
    plt.show()


    features_f.rename(columns={'time': 'time_f_f'}, inplace=True)
    classifier_f.rename(columns={'time': 'time_c_f'}, inplace=True)
    features_b.rename(columns={'time': 'time_f_b'}, inplace=True)
    classifier_b.rename(columns={'time': 'time_c_b'}, inplace=True)

    combined_df = features_f.copy(deep=True)
    combined_df = combined_df.merge(classifier_f, on='idx')
    combined_df = combined_df.merge(features_b, on='idx')
    combined_df = combined_df.merge(classifier_b, on='idx')

    combined_df['offload_time'] = combined_df['time_f_f'] + combined_df['time_f_b']
    combined_df['gained_time'] = combined_df['time_c_f'] + combined_df['time_f_f'] + combined_df['time_f_b']

    data_list = []
    for _, row in combined_df.iterrows():
        data_list.append([row['offload_time'], 'offload', row['idx']])
        data_list.append([row['gained_time'], 'gained', row['idx']])
    # offload = features_f.copy(deep=True)
    # frozen = features_f.copy(deep=True)
    #
    # offload['time'] += features_b['time']
    # frozen['time'] = classifier_f['time'].values + classifier_b['time'].values
    # Compute time of part that is offloaded to strong node

    return pd.DataFrame(data_list, columns=['time', 'type', 'idx'])

if __name__ == '__main__':
    print('Hello world')


    df = pd.read_csv('stability_data.csv')
    calc = calc_metric(df, 15)

    plt.figure()
    sns.lineplot(data=calc, x='idx', y='time', hue='type')
    plt.show()
    # first = df.head(10)
    # groups = df.groupby(['idx', 'layer_id', 'event'])
    # # df['layer_id'] = pd.to_
    # df['layer_id'] = df['layer_id'].astype(str)
    # plt.figure()
    # # sns.lineplot(data=df, x='idx', y='time', hue='layer_id')
    # g = sns.FacetGrid(df, col="event", hue='layer_id')
    # g.map(sns.lineplot, "idx", "time")
    # plt.show()

    # for i in groups.groups:
    #     print(groups.groups[i])
#
# clean_df = parse_data(df, model_name, meta_data)
# meta_data = {
#         'lenet-5': 6,
#         'alexnet': 13,
#         'vgg16': 13,
#         'cifar_10_cnn': 15
#     }