import collections
import xgboost as xgb
import tempfile
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse
Any = object()

_Model = Any  # real type is Union[Booster, CVPack]; need more work

class Plotting(xgb.callback.TrainingCallback):
    '''Plot evaluation result during training.  Only for demonstration purpose as it's quite
    slow to draw.

    '''
    def __init__(self, rounds):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.rounds = rounds
        self.lines = {}
        self.fig.show()
        self.x = np.linspace(0, self.rounds, self.rounds)
        plt.ion()

    def _get_key(self, data, metric):
        return f'{data}-{metric}'

    def after_iteration(self, model, epoch, evals_log):
        '''Update the plot.'''
        if not self.lines:
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    expanded = log + [0] * (self.rounds - len(log))
                    self.lines[key],  = self.ax.plot(self.x, expanded, label=key)
                    self.ax.legend()
        else:
            # https://pythonspot.com/matplotlib-update-plot/
            for data, metric in evals_log.items():
                for metric_name, log in metric.items():
                    key = self._get_key(data, metric_name)
                    expanded = log + [0] * (self.rounds - len(log))
                    self.lines[key].set_ydata(expanded)
            self.fig.canvas.draw()
        # False to indicate training should not stop.
        return False

class CustomLearningRate(xgb.callback.LearningRateScheduler):
    def __init__(self, learning_rates=0.01) -> None:
        self.lr = 0
        def custom_learning_rate(boosting_round):           
            self.lr = 1.0 / (boosting_round + 1)*0.1
            return 0.1 / (boosting_round + 1) 
        learning_rates_func = custom_learning_rate
        assert callable(learning_rates_func) or isinstance(
            learning_rates_func, collections.abc.Sequence
        )
        if callable(learning_rates_func):
            self.learning_rates = learning_rates_func
        else:
            # self.learning_rates = lambda epoch: cast(Sequence, learning_rates)[epoch]
            pass

        super().__init__(learning_rates = learning_rates_func)
        self.pre_metric=0.

    def after_iteration(
        # self, model: _Model, epoch: int, evals_log: xgb.TrainingCallback.EvalsLog
        self, model: _Model, epoch: int, evals_log
        ) -> bool:
        for data, metric in evals_log.items():
            for metric_name, log in reversed(metric.items()):
                if self.pre_metric>=log[-1]:
                    model.set_param("learning_rate", self.learning_rates(epoch))
                    # print('down', self.pre_metric)
                else:
                    print('saved', log[-1])
                    model.save_model('best_model{}.pt'.format(log[-1]))

                    pass
                self.pre_metric=log[-1]
                
                break
            break
        # False to indicate training should not stop.
        return False