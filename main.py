"""module for running complete process of loading data and training models"""
import pickle

from hyperparams_searcher import HyperparamsSearcher
from storage import Storage
from helper import plot_results, get_dataloaders


def main():
    """function for experimenting and running complete pipeline"""
    with open(Storage.TEST_WV_WRAPPER_FILENAME, "rb") as file:
        wv_wrapper = pickle.load(file)

    wv = wv_wrapper["wv"]
    data_dir = "dataset1"
    data_types = {"train": "train2017", "val": "val2017"}
    path = "test_models_weights"
    dataloaders = get_dataloaders(data_dir, data_types, wv)
    a = next(iter(dataloaders["val"]))
    
    # hyperparams_searcher = HyperparamsSearcher(dataloaders, wv_wrapper, path)
    # rate_r_range = (-6, 1)
    # gamma_r_range = (-6, 2)
    # result = hyperparams_searcher.find_hyperparams(rate_r_range, gamma_r_range)
    # best_val, best_model, best_stats, results = result
    # plot_results(best_stats["losses"], "losses")


if __name__ == "__main__":
    main()
