import math


def lr_cosine_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    num_steps: int,
    warmup_fraction: float,
    annealing_fraction: float,
) -> float:
    warmup_last_step = round(num_steps * warmup_fraction)
    annealing_last_step = round(num_steps * annealing_fraction)

    if step < warmup_last_step:
        return (step / warmup_last_step) * max_lr
    elif warmup_last_step <= step < annealing_last_step:
        annealing_num_steps = annealing_last_step - warmup_last_step
        annealing_step = step - warmup_last_step
        return min_lr + 0.5 * (1 + math.cos(annealing_step / annealing_num_steps * math.pi)) * (max_lr - min_lr)
    else:
        return min_lr
