import time
from collections import OrderedDict

import numpy as np
from tabpfn import TabPFNClassifier

import torch
import torch.nn as nn

from common.utils import is_resume
from utils import MetricLogger, save_checkpoint, save_checkpoint_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta_trainer(P, train_func, test_func, model, train_loader, test_loader, logger, feature_subsets=None):
    metric_logger = MetricLogger(delimiter="  ")

    """ resume option """
    is_best, start_step, best, acc = False, 1, 100.0, 0.0

    """ define loss function """
    criterion = nn.CrossEntropyLoss()

    """ training start """
    logger.log_dirname(f"Start training")
    total_accuracy = 0
    total_tasks = 0

    for step in range(P.outer_steps):
        try:
            print(step)
            train_batch = next(train_loader)
            support_inputs, support_targets = train_batch['train']
            query_inputs, query_targets = train_batch['test']

            batch_size = support_inputs.shape[0]
            num_ways = P.num_ways
            num_shots = P.num_shots
            feature_dim = support_inputs.shape[-1]

            support_inputs = support_inputs.reshape(batch_size * num_shots * num_ways, feature_dim)
            support_targets = support_targets.reshape(batch_size * num_shots * num_ways)
            query_inputs = query_inputs.reshape(batch_size * num_ways * P.num_shots_test, feature_dim)
            query_targets = query_targets.reshape(batch_size * num_ways * P.num_shots_test)

            if isinstance(model, list):
                all_preds = []
                for i, submodel in enumerate(model):
                    submodel.fit(support_inputs[:, feature_subsets[i]], support_targets)
                    preds = submodel.predict(query_inputs[:, feature_subsets[i]])
                    all_preds.append(preds)
                all_preds = np.array(all_preds).T
                final_preds = []
                for pred_row in all_preds:
                    counts = np.bincount(pred_row)
                    final_preds.append(np.argmax(counts))
                correct = sum(1 for i in range(len(final_preds)) if final_preds[i] == query_targets[i])
                acc = correct / len(query_targets)
                total_accuracy += acc
                total_tasks += 1
            else:
                model.fit(support_inputs, support_targets)
                y_eval, p_eval = model.predict(query_inputs, return_winning_probability=True)

                correct = sum(1 for i in range(len(y_eval)) if y_eval[i] == query_targets[i])
                acc = correct / len(y_eval)

                total_accuracy += acc
                total_tasks += 1

            if step % 100 == 0:
                avg_accuracy = total_accuracy / total_tasks
                print("Avg accuracies =", avg_accuracy)
                logger.log("STEP" + str(step) + ", Avg Accuracies = " + str(avg_accuracy))
        except Exception as e:
            print('no niezle', e)
