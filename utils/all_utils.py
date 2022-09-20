from matplotlib import  pyplot as plt
import os
from matplotlib.colors import ListedColormap
import numpy as np


def preparedata(df):
    x=df.drop(['Y'],axis=1)
    y=df['Y']
    return x,y

#Plots

def save_plot(df, model, filename='plot.png', plot_dir='plots'):
    def _create_plot(df):
        df.plot(kind='scatter', x='x1', y='x2', c='Y', s=100, cmap='summer_r')
        plt.axhline(y=0, color='Black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='Black', linestyle='--', linewidth=1)
        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _Decision_plot(X, y, classifier, resolution=0.02):
        colors = ('cyan', 'lightgreen')
        cmap = ListedColormap(colors)
        X = X.values
        x1 = X[:, 0]
        x2 = X[:, 1]
        x1_min, x1_max = x1.min() - 1, x1.max() + 1
        x2_min, x2_max = x2.min() - 1, x2.max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution)
                               )
        y_hat = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        y_hat = y_hat.reshape(xx1.shape)
        plt.contourf(xx1, xx2, y_hat, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.plot()

    x, y = preparedata(df)

    _create_plot(df)
    _Decision_plot(x, y, model)

    os.makedirs(plot_dir, exist_ok=True)  # create Folder
    plot_path = os.path.join(plot_dir, filename)  # join file name with folder name to create path
    plt.savefig(plot_path)

