from torch.optim.lr_scheduler import LambdaLR


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
