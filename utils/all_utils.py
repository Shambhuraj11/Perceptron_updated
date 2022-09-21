from matplotlib import  pyplot as plt
import os
from matplotlib.colors import ListedColormap
import numpy as np
import logging


def preparedata(df):
    """
    It returns label and Independent Features

    :param df (pd.DataFrame): This takes a dataframe

    :return: THis return same dataframe but without label column separated

    """

    logging.info("Start Preparing Data for Model")
    x=df.drop(['Y'],axis=1)
    y=df['Y']
    return x,y

#Plots

def save_plot(df, model, filename='plot.png', plot_dir='plots'):
    """
    This Function takes arguments to create plot and decision plots

    :param df (pd.DataFrame): This takes our data frame
    :param model: This is model that we created after training
    :param filename: This image Name of graph
    :param plot_dir: It is folder in which image will be stored

    """
    def _create_plot(df):
        logging.info('Creating Plot')
        df.plot(kind='scatter', x='x1', y='x2', c='Y', s=100, cmap='summer_r')
        plt.axhline(y=0, color='Black', linestyle='--', linewidth=1)
        plt.axvline(x=0, color='Black', linestyle='--', linewidth=1)
        figure = plt.gcf()
        figure.set_size_inches(10, 8)

    def _Decision_plot(X, y, classifier, resolution=0.02):
        colors = ('cyan', 'lightgreen')
        logging.info('Creating Decision Regions')
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
    logging.info(f'Saving plot at {plot_path}')

