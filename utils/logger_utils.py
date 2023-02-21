import wandb

class Logger:
    def __init__(self, writer, test_fn, test_metric='accuracy'):
        self.writer = writer
        self.test_metric = test_metric
        self.test_fn = test_fn

    def log(self, step, model):
        metric = self.test_fn(model)
        if len(metric) == 1:
            t_accuracy = metric[0]
        elif len(metric) == 2:
            t_accuracy, t_loss = metric
        elif len(metric) == 4:
            t_accuracy, t_loss, tr_accuracy, tr_loss = metric
        else:
            raise NotImplementedError

        if self.test_metric == 'accuracy':
            self.writer.add_scalar("correct rate vs round/test", t_accuracy, step)
            if 't_loss' in locals(): self.writer.add_scalar("loss vs round/test", t_loss, step)
            if 'tr_accuracy' in locals(): self.writer.add_scalar("correct rate vs round/train", tr_accuracy, step)
            if 'tr_loss' in locals(): self.writer.add_scalar("loss vs round/train", tr_loss, step)

        elif self.test_metric == 'class_wise_accuracy':
            n_classes = len(t_accuracy)
            for i in range(n_classes):
                # the i th element is the accuracy for the test data with label i
                self.writer.add_scalar(f"class-wise correct rate vs round/test/class_{i}", t_accuracy[i], step)
                if 't_loss' in locals(): self.writer.add_scalar(f"class-wise loss vs round/test/class_{i}", t_loss[i], step)
                if 'tr_accuracy' in locals(): self.writer.add_scalar(f"class-wise correct rate vs round/test/class_{i}", tr_accuracy[i], step)
                if 'tr_loss' in locals(): self.writer.add_scalar(f"class-wise loss vs round/test/class_{i}", tr_loss[i], step)
        elif self.test_metric == 'model_monitor':
            self.writer.add_scalar("model param norm vs round", metric[0], step)
        else:
            raise NotImplementedError

class wandbLogger:
    def __init__(self, writer, test_fn, test_metric='accuracy'):
        self.writer = writer
        self.test_metric = test_metric
        self.test_fn = test_fn

    def log(self, step, model):
        metric = self.test_fn(model)
        if len(metric) == 1:
            t_accuracy = metric[0]
        elif len(metric) == 2:
            t_accuracy, t_loss = metric
        elif len(metric) == 4:
            t_accuracy, t_loss, tr_accuracy, tr_loss = metric
        else:
            raise NotImplementedError

        if self.test_metric == 'accuracy':
            wandb.log({"correct rate vs round/test": t_accuracy, "step":step})
            if 't_loss' in locals(): wandb.log({"loss vs round/test": t_loss, "step":step})
            if 'tr_accuracy' in locals(): wandb.log({"correct rate vs round/train":tr_accuracy, "step":step})
            if 'tr_loss' in locals(): wandb.log({"loss vs round/train":tr_loss, "step":step})

        elif self.test_metric == 'class_wise_accuracy':
            n_classes = len(t_accuracy)
            for i in range(n_classes):
                # the i th element is the accuracy for the test data with label i
                wandb.log({f"class-wise correct rate vs round/test/class_{i}":t_accuracy[i], "step":step})
                if 't_loss' in locals(): wandb.log({f"class-wise loss vs round/test/class_{i}": t_loss[i], "step":step})
                if 'tr_accuracy' in locals(): wandb.log({f"class-wise correct rate vs round/test/class_{i}":tr_accuracy[i], "step":step})
                if 'tr_loss' in locals(): wandb.log({f"class-wise loss vs round/test/class_{i}":tr_loss[i], "step":step})
        elif self.test_metric == 'model_monitor':
            wandb.log({"model param norm vs round": metric[0], "step":step})
        else:
            raise NotImplementedError