import numpy as np
import matplotlib.pyplot as plt
class Gomi():
    def __init__(self, path):
        self.path = path
        self.g    = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        self.h    = [0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    def save(self):
        plt.plot(self.g, label='training loss')
        plt.plot(self.h, label='validation loss')
        plt.legend()
        plt.savefig(f'train/gomi.png', format='png', dpi=300)