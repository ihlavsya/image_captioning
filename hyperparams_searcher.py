"""module for finding optimal hyperparams"""
import sys

import numpy as np

from storage import Storage
from model_trainer import ModelTrainer


class HyperparamsSearcher():
    """instance that helps find best hyperprams with fixed
    dataset, pretrained_weights"""

    def __init__(self, dataloaders, wv_wrapper, path):
        """init instance for running ModelTrainer"""
        self.dataloaders = dataloaders
        self.wv_wrapper = wv_wrapper
        self.path = path

    def train_small_test_version(self, hyperparams_dict):
        """perform training on small test data"""
        trainer = ModelTrainer(self.dataloaders, hyperparams_dict,
                               self.wv_wrapper, self.path)
        model, losses, accuracies = trainer.train(epochs=3)
        return model, losses, accuracies

    def find_hyperparams(self, rate_r_range, gamma_r_range, max_count=10):
        """find best hyperparams within input ranges"""
        best_val = - float(sys.maxsize)
        best_model = None
        best_stats = None
        results = {}
        for i in range(max_count):
            print("iteration {0}".format(i))
            rate_r = np.random.uniform(rate_r_range)[0]
            gamma_r = np.random.uniform(gamma_r_range)[0]
            lr = (10**rate_r)
            gamma = (10**gamma_r)
            hyperparams_dict = self.get_hyperparams_dict(lr, gamma)
            print("rate_r: {0}, gamma_r: {1}".format(rate_r, gamma_r))
            try:
                model, losses, accuracies = self.train_small_test_version(
                    hyperparams_dict)
            except RuntimeError as ex:
                print(ex)
                continue
            # we take here last accuracy for simplicity
            val_accuracy = accuracies["val"][-1]
            train_loss = losses["train"][-1]
            print("validation accuracy:{},train_loss:{},rate_r:{},gamma_r:{}"
                  .format(val_accuracy, train_loss, rate_r, gamma_r))

            stats = {"losses": losses, "accuracies": accuracies}
            results[(rate_r, gamma_r)] = (
                val_accuracy, train_loss, lr, gamma, stats)
            if best_val < val_accuracy:
                best_val = val_accuracy
                best_model = model
                best_stats = stats

        return best_val, best_model, best_stats, results

    def get_hyperparams_dict(self, lr, gamma):
        """create and return filled dict"""
        hyperparams_dict = Storage.BASE_HYPERPARAMS_DICT
        hyperparams_dict["learning_rate"] = lr
        hyperparams_dict["gamma"] = gamma
        return hyperparams_dict
