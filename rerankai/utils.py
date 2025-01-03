from torch.utils.data.dataloader import default_collate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def recursive_collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: recursive_collate_fn([b[key] for b in batch]) for key in batch[0]}
    else:
        return default_collate(batch)
    
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def calculate_metrics(predictions, ground_truth):
    binary_predictions = [1 if pred == 0 else 0 for pred in predictions]
    binary_ground_truth = [1 if true == 0 else 0 for true in ground_truth]
    
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    precision = precision_score(binary_ground_truth, binary_predictions)
    recall = recall_score(binary_ground_truth, binary_predictions)
    f1 = f1_score(binary_ground_truth, binary_predictions)
    
    log_message = (
        "==== 模型性能评估 ====\n"
        f"准确率 (Accuracy):  {accuracy:.4f}\n"
        f"精确率 (Precision): {precision:.4f}\n"
        f"召回率 (Recall):    {recall:.4f}\n"
        f"F1 得分 (F1-Score): {f1:.4f}\n"
        "=====================\n"
    )
    
    return log_message, accuracy, precision, recall, f1