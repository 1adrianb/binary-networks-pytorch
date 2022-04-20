import torch
from functools import partial
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)

# a wrapper function
def add_target(f, TARGET):
    return partial(f, TARGET=TARGET)


# Define main functions


def rmse(outputs, batch, TARGET):
    outputs = outputs["preds"].cpu()
    return (
        mean_squared_error(batch[TARGET].detach().cpu(), outputs.detach().cpu())
        ** 0.5
    )


def mse_loss(outputs, batch, TARGET):
    return torch.nn.functional.mse_loss(
        outputs["preds"].view_as(batch[TARGET]), batch[TARGET]
    )


def cross_entropy(outputs, batch, TARGET):
    entropy = torch.nn.CrossEntropyLoss()
    return entropy(outputs["preds"], batch[TARGET])


def f1_macro(outputs, batch, TARGET):
    _, preds = torch.max(outputs["preds"], 1)
    preds = preds.cpu()
    return f1_score(batch[TARGET].cpu(), preds, average="macro")


def f1_weighted(outputs, batch, TARGET):
    _, preds = torch.max(outputs["preds"], 1)
    preds = preds.cpu()
    return f1_score(batch[TARGET].cpu(), preds, average="weighted")


def accuracy(outputs, batch, TARGET):
    _, preds = torch.max(outputs["preds"], 1)
    preds = preds.cpu()
    return accuracy_score(batch[TARGET].cpu(), preds)

def r2(outputs, batch, TARGET):
    return(r2_score(batch[TARGET].cpu(), outputs.cpu()))


class AucScore:
    def __init__(self, TARGET):
        self.TARGET = TARGET
        self.history_preds = []
        self.history_yhat = []

    def reset(self):
        self.history_preds = []
        self.history_yhat = []

    def update(self, outputs, batch):
        preds = outputs["preds"][:, 1]
        preds = preds.detach().cpu().tolist()
        yhat = batch[self.TARGET].cpu().tolist()
        self.history_preds += preds
        self.history_yhat += yhat
        try:
            score = roc_auc_score(self.history_yhat, self.history_preds)
        except Exception as e:
            print(e)
            score = 0.5

        return score


LOSSES = {"CrossEntropyLoss": cross_entropy, "MSELoss": mse_loss}


METRICS = {
    "f1_macro": (f1_macro, "average"),
    "f1_weighted": (f1_weighted, "average"),
    "accuracy": (accuracy, "average"),
    "auc": (AucScore, "full"),
    "rmse": (rmse, "average"),
    "r2": (r2, "average"),
}


def get_metrics_and_loss(loss_name, metric_names, target_name):
    loss_fn = add_target(LOSSES[loss_name], target_name)

    metrics = []

    for m_name in metric_names:
        f, t = METRICS[m_name]
        if "auc" in m_name:
            metrics.append((m_name, add_target(f, target_name)(), t))
        else:
            metrics.append((m_name, add_target(f, target_name), t))

    return loss_fn, metrics
