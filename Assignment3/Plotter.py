import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    # created a function for plotting scatter diagram which uses seaborn.
    def plot_scatter_diagram(df, dependent_column_name):
        fig, axes = plt.subplots(4, int((len(df.columns)/4) + 1),figsize=(15,15))
        for i, col_name in enumerate(df.columns):
            if col_name != dependent_column_name:
                sns.scatterplot(x=col_name, y=dependent_column_name, data=df, ax=axes[i%4, i//4])
        plt.tight_layout()
        plt.show()

    def plot_histogram(df):
        fig,axes = plt.subplots(4, int(len(df.columns)//4)+1, figsize=(15,15))
        for i, col_name in enumerate(df.columns):
            axes[i%4, i//4].hist(df[col_name], bins=100)
            axes[i%4, i//4].set_xlabel(col_name)
            axes[i%4, i//4].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()