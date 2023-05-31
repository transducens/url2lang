
import os
import logging

import url2lang.url2lang as url2lang

import numpy as np
import torch
import matplotlib.pyplot as plt
import sklearn.metrics

logger = logging.getLogger("url2lang")

# TODO check if values from sklearn are the same that the ones we calculate. If so, change own calculated values with sklearn
DEBUG = bool(int(os.environ["U2L_DEBUG"])) if "U2L_DEBUG" in os.environ else False

if DEBUG:
    logger.info("DEBUG is enabled")

def get_confusion_matrix(outputs_argmax, labels, classes=2):
    tp, fp, fn, tn = np.zeros(classes), np.zeros(classes), np.zeros(classes), np.zeros(classes)
    conf_mat = np.array([[torch.sum(torch.logical_and(outputs_argmax == c1, labels == c2)) for c1 in range(classes)] for c2 in range(classes)])

    if DEBUG:
        sklearn_conf_mat = sklearn.metrics.confusion_matrix(labels, outputs_argmax, labels=list(range(classes)))

        if (conf_mat != sklearn_conf_mat).any():
            logger.error("Own confusion matric is different from the one calculated in sklearn: %s vs %s", conf_mat["conf_mat"], sklearn_conf_mat)

    for c in range(classes):
        # Multiclass confusion matrix
        # https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/
        tp[c] = int(torch.sum(torch.logical_and(labels == c, outputs_argmax == c)))
        fp[c] = int(torch.sum(torch.logical_and(labels != c, outputs_argmax == c)))
        fn[c] = int(torch.sum(torch.logical_and(labels == c, outputs_argmax != c)))
        tn[c] = int(torch.sum(torch.logical_and(labels != c, outputs_argmax != c)))

    if DEBUG:
        # Check
        def check(a, b, desc):
            if a != b:
                logger.error("%s: %d vs %d", desc, a, b)

        for c in range(classes):
            idxs = np.arange(classes)

            check(tp[c], conf_mat[c][c], f"class {c} -> TP != confusion matrix")
            check(fp[c], np.sum(conf_mat[idxs != c][:,c]), f"class {c} -> FP != confusion matrix")
            check(fn[c], np.sum(conf_mat[c][idxs != c]), f"class {c} -> FN != confusion matrix")
            check(tn[c], np.sum([conf_mat[c1][c2] for c1 in range(classes) for c2 in range(classes) if c1 != c and c2 != c]), f"class {c} -> TN != confusion matrix")

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "conf_mat": conf_mat,
    }

def get_metrics(outputs_argmax, labels, current_batch_size, classes=2, batch_idx=-1, log=False, task="UNKNOWN"):
    if classes < 2:
        raise Exception(f"Classes has to be greater or equal than 2: {classes}")
    #elif classes > 2:
    #    logger.warning("Some metrics might not work as expected since they have been designed to work for a binary classification")

    sklearn_mcc = sklearn.metrics.matthews_corrcoef(labels, outputs_argmax)

    if DEBUG:
        sklearn_precision_recall_fscore_support = \
            sklearn.metrics.precision_recall_fscore_support(labels, outputs_argmax, labels=list(range(classes)), zero_division=0)

    acc = (torch.sum(outputs_argmax == labels) / current_batch_size).cpu().detach().numpy()

    no_values_per_class = np.zeros(classes)
    acc_per_class = np.zeros(classes)
    precision, recall, f1 = np.zeros(classes), np.zeros(classes), np.zeros(classes)
    macro_f1 = 0.0
    mcc = np.zeros(classes)

    conf_mat = get_confusion_matrix(outputs_argmax, labels, classes=classes)
    tp, fp, fn, tn = conf_mat["tp"], conf_mat["fp"], conf_mat["fn"], conf_mat["tn"]

    for c in range(classes):
        no_values_per_class[c] = torch.sum(labels == c)

        # How many times have we classify correctly the target class taking into account all the data? -> we get how many percentage is from each class
        acc_per_class[c] = torch.sum(torch.logical_and(labels == c, outputs_argmax == c)) / current_batch_size

        # Metrics
        # http://rali.iro.umontreal.ca/rali/sites/default/files/publis/SokolovaLapalme-JIPM09.pdf
        # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
        precision[c] = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) != 0 else 0.0
        recall[c] = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) != 0 else 0.0
        f1[c] = 2 * ((precision[c] * recall[c]) / (precision[c] + recall[c])) if not np.isclose(precision[c] + recall[c], 0.0) else 0.0

        # MCC
        mcc[c] = (tp[c] + fp[c]) * (tp[c] + fn[c]) * (tn[c] + fp[c]) * (tn[c] + fn[c])
        mcc[c] = (tp[c] * tn[c] - fp[c] * fn[c]) / np.sqrt(mcc[c]) if mcc[c] != 0 else (tp[c] * tn[c] - fp[c] * fn[c])

        if DEBUG:
            # Sklearn
            if not np.isclose(sklearn_precision_recall_fscore_support[0][c], precision[c]):
                logger.warning("Own precision is different from the one calculated in sklearn: class %d -> %f vs %f", c, sklearn_precision_recall_fscore_support[0][c], precision[c])
            if not np.isclose(sklearn_precision_recall_fscore_support[1][c], recall[c]):
                logger.warning("Own recall is different from the one calculated in sklearn: class %d -> %f vs %f", c, sklearn_precision_recall_fscore_support[1][c], recall[c])
            if not np.isclose(sklearn_precision_recall_fscore_support[2][c], f1[c]):
                logger.warning("Own f1 is different from the one calculated in sklearn: class %d -> %f vs %f", c, sklearn_precision_recall_fscore_support[2][c], f1[c])

    #assert outputs.shape[-1] == acc_per_class.shape[-1], f"Shape of outputs does not match the acc per class shape ({outputs.shape[-1]} vs {acc_per_class.shape[-1]})"
    if not np.isclose(np.sum(acc_per_class), acc):
        raise Exception(f"Acc and the sum of acc per classes should match: {acc} vs {np.sum(acc_per_class)}")

    macro_f1 = np.sum(f1) / f1.shape[0]

    if DEBUG:
        if classes == 2:
            # Check that all values in mcc are the same since the value is symetric
            for idx in range(len(mcc) - 1):
                if not np.isclose(mcc[idx], mcc[idx + 1]):
                    logger.error("MCC of classes %d and %d are different: %f vs %f", idx, idx + 1, mcc[idx], mcc[idx + 1])

            mcc = mcc[0]

            if not np.isclose(mcc, sklearn_mcc):
                logger.error("Own MCC is different from the one calculated in sklearn: %f vs %f", mcc, sklearn_mcc)

    mcc = sklearn_mcc # Version from sklearn works for multiclass

    if log:
        logger.debug("[train:batch#%d] Acc (task '%s'): %.2f %% (%s)",
                     batch_idx + 1, task, acc * 100.0, "; ".join([f"{_lang}: {acc_per_class[_id] * 100.0:.2f} %" for _lang, _id in url2lang._lang2id.items()]))
        logger.debug("[train:batch#%d] Values per class (task '%s'; precision|recall|f1): "
                     "(%s | %s | %s)", batch_idx + 1, task,
                     "; ".join([f"{no_values_per_class[_id]} -> {_lang}: {precision[_id] * 100.0:.2f} %" for _lang, _id in url2lang._lang2id.items()]),
                     "; ".join([f"{no_values_per_class[_id]} -> {_lang}: {recall[_id] * 100.0:.2f} %" for _lang, _id in url2lang._lang2id.items()]),
                     "; ".join([f"{no_values_per_class[_id]} -> {_lang}: {f1[_id] * 100.0:.2f} %" for _lang, _id in url2lang._lang2id.items()]))
        logger.debug("[train:batch#%d] Macro F1 (task '%s'): %.2f %%", batch_idx + 1, task, macro_f1 * 100.0)
        logger.debug("[train:batch#%d] MCC (task '%s'): %.2f %%", batch_idx + 1, task, mcc * 100.0)

    return {
        "acc": acc,
        "acc_per_class": acc_per_class,
        "no_values_per_class": no_values_per_class,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }

def get_metrics_task_specific(task, outputs, labels, current_batch_size, classes=2, batch_idx=-1, log=False):
    if task in ("urls_classification", "language-identification", "langid-and-urls_classification"):
        metrics = get_metrics(outputs, labels, current_batch_size, classes=classes, batch_idx=batch_idx, log=log, task=task)
    elif task == "mlm":
        # TODO how can we evaluate MLM beyond the loss?
        logger.warning("Task '%s' is not being evaluated: returning empty metrics", task)

        metrics = {}
    else:
        raise Exception(f"Unknown task: {task}")

    return metrics

def plot_statistics(args, path=None, time_wait=5.0, freeze=False):
    plt_plot_common_params = {"marker": 'o', "markersize": 2,}
    plt_scatter_common_params = {"marker": 'o', "s": 2,}
    plt_legend_common_params = {"loc": "center left", "bbox_to_anchor": (1, 0.5), "fontsize": "x-small",}

    plt.clf()

    plt.subplot(3, 2, 1)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_loss"]))))), args["batch_loss"], label="Train loss", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 5)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc"]))))), args["batch_acc"], label="Train acc", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][0]))))), args["batch_acc_classes"][0], label="Train F1: no p.", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_acc_classes"][1]))))), args["batch_acc_classes"][1], label="Train F1: para.", **plt_plot_common_params)
    plt.plot(list(map(lambda x: x * args["show_statistics_every_batches"], list(range(len(args["batch_macro_f1"]))))), args["batch_macro_f1"], label="Train macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 2)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_loss"], label="Train loss", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_loss"], label="Dev loss", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 3)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc"], label="Train acc", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc_classes"][0], label="Train F1: no p.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_acc_classes"][1], label="Train F1: para.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_train_macro_f1"], label="Train macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plt.subplot(3, 2, 4)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc"], label="Dev acc", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc_classes"][0], label="Dev F1: no p.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_acc_classes"][1], label="Dev F1: para.", **plt_plot_common_params)
    plt.plot(list(range(1, args["epoch"] + 1)), args["epoch_dev_macro_f1"], label="Dev macro F1", **plt_plot_common_params)
    plt.legend(**plt_legend_common_params)

    plot_final = True if args["final_dev_acc"] else False

    plt.subplot(3, 2, 6)
    plt.scatter(0 if plot_final else None, args["final_dev_acc"] if plot_final else None, label="Dev acc", **plt_scatter_common_params)
    plt.scatter(0 if plot_final else None, args["final_test_acc"] if plot_final else None, label="Test acc", **plt_scatter_common_params)
    plt.scatter(1 if plot_final else None, args["final_dev_macro_f1"] if plot_final else None, label="Dev macro F1", **plt_scatter_common_params)
    plt.scatter(1 if plot_final else None, args["final_test_macro_f1"] if plot_final else None, label="Test macro F1", **plt_scatter_common_params)
    plt.legend(**plt_legend_common_params)

    #plt.tight_layout()
    plt.subplots_adjust(left=0.08,
                        bottom=0.07,
                        right=0.8,
                        top=0.95,
                        wspace=1.0,
                        hspace=0.4)

    if path:
        plt.savefig(path, dpi=1200)
    else:
        if freeze:
            plt.show()
        else:
            plt.pause(time_wait)
