import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




if __name__ == '__main__':
    exp_name = 'output/exp_p3_w4_s4_deadline'

    general_file = f'{exp_name}-general_data.csv'
    print(f'Loading data file: {general_file}')
    df = pd.read_csv(general_file)
    print(df)


    plt.figure()
    sns.pointplot(data=df, x='epoch', y='accuracy')
    plt.title('Accuracy per epoch')
    plt.show()

    plt.figure()
    # sns.pointplot(data=df[df['epoch'] > 1], x='epoch', y='duration')
    sns.pointplot(data=df, x='epoch', y='duration')
    plt.title('Train time per epoch')
    plt.show()

    dfs = []
    for file in [f'{exp_name}_client1_epochs.csv', f'{exp_name}_client2_epochs.csv', f'{exp_name}_client3_epochs.csv', f'{exp_name}_client4_epochs.csv']:
        dfs.append(pd.read_csv(file))
    client_df = pd.concat(dfs, ignore_index=True)

    print('Loading client data')
    plt.figure()
    # sns.pointplot(data=client_df[client_df['epoch_id'] > 1], x='epoch_id', y='duration_train', hue='client_id')
    sns.pointplot(data=client_df, x='epoch_id', y='duration_train', hue='client_id')
    plt.title('Train time per epoch clients')
    plt.show()

    plt.figure()
    sns.pointplot(data=client_df, x='epoch_id', y='accuracy', hue='client_id')
    plt.title('Accuracy per epoch clients')
    plt.show()

