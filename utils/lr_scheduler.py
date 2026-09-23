"""Learning-rate schedules that depend on optimizer-step counts."""

from torch.optim.lr_scheduler import LambdaLR


class WarmupPolynomialLR(LambdaLR):
    """Linear warmup followed by polynomial decay to zero.

    ``total_steps`` is the maximum number of optimizer updates, not epochs.
    The scheduler is stepped only after a successful optimizer update.
    """

    def __init__(self, optimizer, total_steps, warmup_ratio=0.05, power=1.0,
                 last_epoch=-1):
        total_steps = int(total_steps)
        warmup_ratio = float(warmup_ratio)
        power = float(power)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if power <= 0.0:
            raise ValueError("power must be positive")

        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = (
            max(1, round(total_steps * warmup_ratio))
            if warmup_ratio > 0.0 else 0
        )
        self.power = power

        def multiplier(step):
            if self.warmup_steps and step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            progress = (step - self.warmup_steps) / decay_steps
            progress = min(max(progress, 0.0), 1.0)
            return (1.0 - progress) ** self.power

        super().__init__(optimizer, multiplier, last_epoch=last_epoch)
