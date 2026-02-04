import numpy as np
from matplotlib import pyplot as plt
import os


def plot_single_result(
    model_out, ground_truth, save_path, input_series=None, title=None
):
    if input_series is not None:
        ground_truth = np.concatenate([input_series, ground_truth])
        model_out = np.concatenate([input_series, model_out])
    plt.figure(figsize=(32, 24))
    plt.plot(ground_truth, label="grountruth", color="blue")
    plt.plot(model_out, label="model_out", color="orange")
    plt.legend(prop={"size": 20})
    if title:
        plt.title(title, fontdict={"size": 20})
    plt.savefig(save_path)


def plot_all_result(model_out, ground_truth, save_path):
    model_out = np.concatenate(model_out[:, :, -1])
    ground_truth = np.concatenate(ground_truth[:, :, -1])
    plot_single_result(model_out, ground_truth, save_path)
