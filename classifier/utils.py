import os
import json

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def append_to_logfile(log_filepath: str, data: dict):
    if not os.path.exists(log_filepath):
        with open(log_filepath, 'w') as f:
            json.dump([], f)
    
    with open(log_filepath, 'r+') as f:
        logs = json.load(f)
        logs.append(data)
        f.seek(0)
        json.dump(logs, f, indent=4)
