class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, metric, name, metric_type):
        self.metric_type = metric_type
        self.metric = metric
        self.name = name
        self.history = []
        self.avg = 0
        self._reset()

    def _reset(self):
        self.history.append(self.avg)
        """Reset all statistics"""
        self.last_value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.metric_type == "full":
            self.metric.reset()

    def _update_avg(self, val, n=1):
        """Update statistics"""
        self.last_value = val
        if self.name == "loss":
            val = val.detach().item()
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self, outputs, batch):
        if self.metric_type == "average":
            val = self.metric(outputs, batch)

            for k in batch:
                pass
            self._update_avg(val, len(batch[k]))

        if self.metric_type == "full":
            self.avg = self.metric.update(outputs, batch)
