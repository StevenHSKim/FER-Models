import os
import torch
import tqdm
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

def control_random_seed(seed):
    """
    random seed를 고정하는 함수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_model_parameters(model):
    """
    모델 파라미터 수 세는 함수
    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    return parameters

def poster_load_pretrained_weights(model, checkpoint):
    """
    POSTER 코드에서 test를 위해 pretrain weight를 가져오는 함수
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model


##### While train #####

def save_best_model(epoch, acc, balanced_acc, model, dataset_name, optimizer, iteration, checkpoint_dir):
    """
    best model의 checkpoint를 저장하는 함수
    동일한 iteration 내에서 이전의 checkpoint는 삭제
    따라서 각 iteration 별로 한 개의 best model checkpoint만 저장
    """
    # 이전 checkpoint 찾기 및 삭제
    previous_checkpoint = None
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(f"{dataset_name}_iter{iteration+1}_epoch") and filename.endswith(".pth"):
            previous_checkpoint = os.path.join(checkpoint_dir, filename)
            break

    if previous_checkpoint and os.path.exists(previous_checkpoint):
        os.remove(previous_checkpoint)

    # 새로운 checkpoint 저장
    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{dataset_name}_iter{iteration+1}_epoch{epoch}_acc{acc}_bacc{balanced_acc}.pth",
    )
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_checkpoint_path)

    tqdm.write('New best model saved, previous checkpoint removed.')
    return best_checkpoint_path


def check_early_stopping(val_loss, best_loss, patience_counter, patience):
    """
    train 시 early stopping 구현 함수
    """
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        return True, best_loss, patience_counter

    return False, best_loss, patience_counter

def setup_optimizer(model_params, optimizer_type, lr, weight_decay):
    """
    Optimizer 설정 함수
    """
    if optimizer_type == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.6, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def setup_scheduler(optimizer, scheduler_type, t_max, step_size, gamma):
    """
    Learning Rate Scheduler 설정 함수
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


##### Results #####

def poster_plot_confusion_matrix(cm, labels_name, title, acc, output_path=None):
    """
    POSTER 코드에서 가져온 Confusion Matrix 출력 함수 <-- 수정 필요 (legend bar, 감정 순서, 감정 abbv)
    """
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))
    plt.xticks(num_class, labels_name, rotation=90)
    plt.yticks(num_class, labels_name)
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join('./Confusion_matrix', title)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    plt.savefig(os.path.join(output_path, "acc" + str(acc) + ".png"), format='png')
    plt.show()


def calculate_metrics(all_accuracies, all_balanced_accuracies, all_val_losses, all_test_losses):
    """
    Metric(Acc, Bal Acc)과 Loss(Val Loss, Test Loss)를 계산하는 함수
    """
    metrics = {
        'accuracy': (np.mean(all_accuracies), np.std(all_accuracies)),
        'balanced_accuracy': (np.mean(all_balanced_accuracies), np.std(all_balanced_accuracies)),
        'val_loss': (np.mean(all_val_losses), np.std(all_val_losses)),
        'test_loss': (np.mean(all_test_losses), np.std(all_test_losses))
    }
    return metrics


def save_iteration_results(args, metrics, results):
    """
    개별 실험 모델 결과 저장
    """
    results_df = pd.DataFrame(results, columns=['Iteration', 'Test Accuracy', 'Balanced Accuracy', 'Val Loss', 'Test Loss'])
    summary = pd.DataFrame([{
        'Iteration': 'Mean±Std',
        'Test Accuracy': f"{metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.4f}",
        'Balanced Accuracy': f"{metrics['balanced_ccuracy'][0]:.4f}±{metrics['balanced_accuracy'][1]:.4f}",
        'Val Loss': f"{metrics['val_loss'][0]:.4f}±{metrics['val_loss'][1]:.4f}",
        'Test Loss': f"{metrics['test_loss'][0]:.4f}±{metrics['test_loss'][1]:.4f}",
    }])
    
    results_df = pd.concat([results_df, summary])
    results_df.to_csv(f'{args.model_name}_{args.dataset}_test_results.csv', index=False)


def update_total_results(args, metrics, current_time, parameters):
    """
    total_result.csv 파일에 모든 결과를 저장하는 함수
    """
    # Prepare the new result row
    new_result = {
        'Experiment Time': current_time,
        'Train Time': current_time,
        'Iteration': args.iterations,
        'Dataset Name': args.dataset.upper(),
        'Data Split': f'{int((1-args.test_size)*(1-args.val_size)*100)}:{int((1-args.test_size)*args.val_size*100)}:{int(args.test_size*100)}',
        'Model Name': args.model_name,
        'Total Parameters': f"{parameters:.3f}M",
        'Val Loss': f"{metrics['val_loss'][0]:.4f}±{metrics['val_loss'][1]:.4f}",
        'Test Loss': f"{metrics['test_loss'][0]:.4f}±{metrics['test_loss'][1]:.4f}",
        'Acc': f"{metrics['accuracy'][0]:.4f}±{metrics['accuracy'][1]:.4f}",
        'Balanced_Acc': f"{metrics['balanced_ccuracy'][0]:.4f}±{metrics['balanced_accuracy'][1]:.4f}"
    }
    
    expected_columns = ['Experiment Time', 'Train Time', 'Iteration', 'Dataset Name', 'Data Split', 'Model Name', 'Total Parameters', 'Val Loss', 'Test Loss', 'Acc']
    
    results_path = '/userHome/userhome1/automl_undergraduate/FER_Models/total_results.csv'
    
    try:
        total_results_df = pd.read_csv(results_path)
        if total_results_df.empty or not all(col in total_results_df.columns 
                                           for col in expected_columns):
            total_results_df = pd.DataFrame(columns=expected_columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        total_results_df = pd.DataFrame(columns=expected_columns)
    
    total_results_df = pd.concat([total_results_df, pd.DataFrame([new_result])], 
                               ignore_index=True)
    total_results_df.to_csv(results_path, index=False)
    print(f"\nResults have been appended to {results_path}")
    
    
def print_final_results(args, metrics, best_accuracies):
    """
    실험 끝내고 마지막 결과 출력
    """
    print("\nBest Accuracies over all iterations:")
    for i, acc in enumerate(best_accuracies, 1):
        print(f"Iteration {i}: {acc:.4f}")
    
    print(f"\nResults over {args.iterations} iterations:")
    print(f"Test Accuracy: {metrics['accuracy'][0]:.4f} ± {metrics['accuracy'][1]:.4f}")
    print(f"Balanced Accuracy': {metrics['balanced_ccuracy'][0]:.4f}±{metrics['balanced_accuracy'][1]:.4f}")
    print(f"Validation Loss: {metrics['val_loss'][0]:.4f} ± {metrics['val_loss'][1]:.4f}")
    print(f"Test Loss: {metrics['test_loss'][0]:.4f} ± {metrics['test_loss'][1]:.4f}")
    
    
