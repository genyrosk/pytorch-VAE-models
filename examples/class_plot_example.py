import os
import sys
import time
import math
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from loss_plot import LossPlot

data_len = 1000
batch_size = 64
epochs = 10
iters_per_epoch = math.ceil(data_len/batch_size)
total_iters = iters_per_epoch*epochs
# data 
x_axis = np.linspace(0, 9, total_iters)
train_losses = 700 - 2 * np.linspace(0,total_iters, total_iters) + np.random.normal(0, 100, total_iters)
test_losses = 600 - 0.4 * np.linspace(0,total_iters,epochs) + np.random.normal(0, 100, epochs)

loss_plot = LossPlot(epochs=epochs,
                     data_len=data_len,
                     batch_size=batch_size,
                     plot_interval=10,
                     window=21)

for idx, loss in enumerate(train_losses):
    loss_plot.add_item(loss)
    time.sleep(0.05)

    if (idx + 1) % iters_per_epoch == 0:
        epoch = (idx + 1) // iters_per_epoch - 1
        loss_plot.add_test_item(test_losses[epoch])
