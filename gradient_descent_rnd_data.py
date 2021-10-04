import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse


def loss(sample, label, points, labels, num_classes, alpha=0.001):
    x = np.where(labels == label)[0]
    anchor_idx = np.random.choice(x, size=1)
    anchor_sample = points[anchor_idx, :].reshape(2, 1)
    L = 0
    grad = 0
    for negative in range(num_classes):
        """
        compute the loss and gradient for the current sample with respect to all negative samples
        """
        if negative != label:
            x = np.where(labels == negative)[0]
            n_idx = np.random.choice(x, size=1)
            n_sample = points[n_idx, :].reshape(2, 1)
            label_multiplier = np.min([np.abs(label - negative), np.abs(num_classes - np.abs(label - negative))])
            L += np.exp(label_multiplier * np.matmul(sample.T, n_sample) - np.matmul(sample.T, anchor_sample))
            grad += (label_multiplier * n_sample - anchor_sample) * np.exp(
                label_multiplier * np.matmul(sample.T, n_sample) - np.matmul(sample.T, anchor_sample))

    sample_updated = sample - alpha * grad  # perform gradient descent update
    sample_updated /= np.linalg.norm(sample_updated)  # normalise to stay on the unit circle
    return sample_updated.T, L


def run(num_classes, num_iterations, n_datapoints, plotting):
    losses = list()  # intermediate losses
    loss_per_iter = list()  # total loss
    points = np.random.randn(n_datapoints, 2)  # generate random datapoints
    points = points / np.linalg.norm(points, axis=1).reshape(n_datapoints, 1)  # project datapoints on the unit circle
    labels = np.random.randint(0, num_classes, size=(n_datapoints, 1))  # assign random labels to every datapoint
    updated_points = points  # copy original data to plot the difference later
    if plotting == "True":
        plot_scatter(data=points, labels=labels)
    for iter in range(num_iterations):
        for idx, sample in enumerate(points):
            sample = sample.reshape(2, 1)
            label = labels[idx][0]
            updated_points[idx], l = loss(sample, label, updated_points, labels, num_classes=num_classes)
            losses.append(l)
        loss_per_iter.append(np.mean(losses))
        if iter % 10 == 0:
            print(f"loss {np.mean(losses)} in iteration {iter}/{num_iterations}")
        losses = []

    if plotting == "True":
        plt.plot(loss_per_iter)
        plt.ylim(-1, 17)
        plt.show()
        plot_scatter(data=updated_points, labels=labels)
    return updated_points


def plot_scatter(data, labels):
    df = pd.DataFrame()
    df["x"] = data[:, 0]
    df["y"] = data[:, 1]
    df["labels"] = labels
    ax = sns.scatterplot(x="x", y="y", hue="labels", data=df, palette="deep")
    circle1 = plt.Circle(xy=(0, 0), radius=1, color='red', fill=False)
    ax.add_patch(circle1)
    ax.set(xlim=(-2, 2))
    ax.set(ylim=(-2, 2))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", help="number of classes", type=int , default=7)
    parser.add_argument("--num_data", help="number of datapoints", type=int, default=1000)
    parser.add_argument("--num_iter", help="number of iterations of gradient descent", type=int, default=1000)
    parser.add_argument("--plot", choices=('True','False'), help="Choose between True or False to see the plots", default="True")
    args = parser.parse_args()
    updated_points = run(args.num_classes, args.num_iter, args.num_data, args.plot)
