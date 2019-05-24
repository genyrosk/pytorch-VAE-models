import os
import sys
import time
import numpy as np
from tensorboardX import SummaryWriter

x1 = 700 - 0.4 * np.linspace(0,1000,1000) + np.random.normal(0, 100, 1000)
x2 = 600 - 0.4 * np.linspace(0,1000,1000) + np.random.normal(0, 100, 1000)

writer = SummaryWriter()

for i, (a, b) in enumerate(zip(x1, x2)):
    time.sleep(0.01)
    writer.add_scalars('data/scalar_group', {'train': a,
                                             'test':b}, i)
