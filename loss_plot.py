import os
import sys
import time
import math
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

def numpy_ewma_vectorized(data, window):
    """Exponentially Weighted Moving Average"""
    if type(data) != np.ndarray:
        data = np.array(data)
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    # vectors
    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    # calc
    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

class LossPlot:

    def __init__(self,
                epochs,
                data_len,
                batch_size,
                plot_interval=10,
                dir=None,
                window=21,
                type='exma'):
        self.epochs = epochs
        self.data_len = data_len
        self.batch_size = batch_size
        self.interval = plot_interval
        self.iters_per_epoch = math.ceil(data_len/batch_size)
        self.total_iters = self.iters_per_epoch * epochs
        window = int(window)
        self.window = window if window % 2 == 0 else window + 1
        self.dir = dir

        # create plot
        sns.set_style("dark")
        plt.show()
        self.ax = plt.gca()
        self.x_axis = np.linspace(0, epochs-1, self.total_iters)
        self.train_loss = []
        self.test_loss = []
        self.line,        = self.ax.plot([], [], color='b', alpha=0.3)
        self.line_smooth, = self.ax.plot([], [], color='b', label='train loss')
        self.line_test,   = self.ax.plot([], [], color='r', alpha=0.8, label='test loss')

        # annotation
        self.ax.set_title(f'Loss over epochs')
        self.ax.set_xlabel('epochs')
        self.ax.set_ylabel('loss')
        self.ax.legend()
        self.ax.grid()
        plt.xticks(range(epochs))

    def add_item(self, loss):
        # print('update')
        self.train_loss.append(loss)
        xdata = self.x_axis[:len(self.train_loss)]
        idx = len(xdata)
        if idx % self.interval != 0 and idx != self.total_iters - 1:
            return

        # new data
        self.line.set_ydata(self.train_loss)
        self.line.set_xdata(xdata)
        if type == 'savgol':
            if len(self.train_loss) > self.window:
                train_smooth = savgol_filter(self.train_loss, self.window, 3)
                self.line_smooth.set_ydata(train_smooth)
                self.line_smooth.set_xdata(xdata)
        else:
            train_smooth = numpy_ewma_vectorized(self.train_loss, self.window)
            self.line_smooth.set_ydata(train_smooth)
            self.line_smooth.set_xdata(xdata)

        self.ax.set_ylim(0, max(self.train_loss))
        self.ax.set_xlim(0, self.epochs)
        # update
        plt.draw()
        plt.pause(1e-17)

        # last iteration, keep the plot
        # print(self.total_iters, len(self.train_loss))
        # if self.total_iters == len(self.train_loss):
        #     plt.show()
        if self.dir:
            plt.savefig(f'{self.dir}/loss_plot.png')

    def add_test_item(self, loss):
        self.test_loss.append(loss)
        xdata = np.arange(1, len(self.test_loss)+1)

        self.line_test.set_ydata(self.test_loss)
        self.line_test.set_xdata(xdata)
