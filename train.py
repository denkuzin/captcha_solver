import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
import config
import preprocessing
from models import models

from scipy.sparse import coo_matrix
import seaborn as sns
import pandas as pd
import logging

logger = logging.getLogger()

def confusion_matrix(y_true, y_pred, num_classes):
    row = y_true
    col = y_pred
    data = np.ones_like(col)
    CM = coo_matrix((data, (row, col)), shape=(num_classes, num_classes))
    return CM.toarray()


def plot_conf_matrix(CM, name="models/confuse_matrix.png"):
    df_cm = pd.DataFrame(CM, index=[i for i in config.possible_characters],
                         columns=[i for i in config.possible_characters])
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig(name)
    logger.info("the figure {} is saved".format(name))


def train_model(model, data_loader):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rates[0])

    # Train the Model
    model.train()
    total_conf_matrix = None
    total = correct = 0

    for i, (images, labels) in enumerate(data_loader):
        # figure out what the current learning rate is
        num_rates = len(config.learning_rates)
        index_learning_rate = int(i/(config.num_steps/num_rates + 1))
        for g in optimizer.param_groups:
            g['lr'] = config.learning_rates[index_learning_rate]

        # data input flows
        images = Variable(images)
        labels = Variable(labels)  # OHE
        _, labels = labels.max(dim=1)  # to indexes

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % max(1, int(config.num_steps/10)) == 0:
            logger.info('Step [%d/%d], Example [%d/%d] Loss: %.5f Accuracy %0.2f %%'
                  % (
                  i + 1, config.num_steps, (i + 1) * config.batch_size,
                  config.num_steps * config.batch_size, loss.data[0], 100 * correct / total))
            total = correct = 0

        # at the end of training, start to collect statistic for confusion matrix
        if i > config.num_steps * 0.9:
            if total_conf_matrix is None:
                total_conf_matrix = confusion_matrix(labels.data, predicted, config.num_classes)
            else:
                total_conf_matrix += confusion_matrix(labels.data, predicted, config.num_classes)

        if i > config.num_steps:
            plot_conf_matrix(total_conf_matrix, name="models/confuse_matrix_train.png")
            break


def eval_model(model, data_loader):
    correct = total = 0
    total_conf_matrix = None
    model.eval()
    for images, labels in data_loader:
        _, labels = labels.max(dim=1)
        images = Variable(images)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        if total_conf_matrix is None:
            total_conf_matrix = confusion_matrix(labels, predicted, config.num_classes)
        else:
            total_conf_matrix += confusion_matrix(labels, predicted, config.num_classes)

    logger.info('Test Accuracy of the model on the %d test images: %.2f %%' %
                    (total, 100 * correct / total))
    logger.info("precision for 5 correct answers: %.2f" % (correct / total) ** 5)
    plot_conf_matrix(total_conf_matrix, name="models/confuse_matrix_test.png")


def train():
    train_loader = DataLoader(dataset=preprocessing.TrainLoader(),
                         batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=preprocessing.Test_Loader(),
                         batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = models.CNN(num_classes=config.num_classes)
    logger.info("number of parameters is {}".format(model.count_parameters()))
    if torch.cuda.is_available():
        model = model.cuda()

    train_model(model, train_loader)
    eval_model(model, test_loader)
    torch.save(model.state_dict(), config.model_path)


if __name__ == '__main__':
    train()
