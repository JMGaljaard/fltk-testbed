import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




if __name__ == '__main__':
    df = pd.read_csv('output/general_data.csv')
    print(df)


    plt.figure()
    sns.pointplot(data=df, x='epoch', y='accuracy')
    plt.show()