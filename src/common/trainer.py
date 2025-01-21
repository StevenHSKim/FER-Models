import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score

from utils.utils import save_best_model, check_early_stopping
from config.dataset_configs import AdaDF_DATASET_CONFIGS
from models.adadf.adadf import generate_adaptive_LD

class BaseTrainer:
    """Base trainer class with common training and testing logic"""
    
    def __init__(self, model, criterion, optimizer, scheduler, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def _initialize_metrics(self):
        """Initialize tracking metrics for training"""
        return {
            'running_loss': 0.0,
            'correct_sum': 0,
            'iter_cnt': 0,
            'sample_cnt': 0,
            'y_true': [],
            'y_pred': []
        }
    
    def _compute_metrics(self, metrics, dataset_size):
        """Compute accuracy and loss metrics"""
        acc = metrics['correct_sum'] / dataset_size
        loss = metrics['running_loss'] / metrics['iter_cnt'] if metrics['iter_cnt'] > 0 else 0
        
        if metrics['y_true'] and metrics['y_pred']:
            y_true = np.concatenate(metrics['y_true']) if isinstance(metrics['y_true'][0], np.ndarray) else metrics['y_true']
            y_pred = np.concatenate(metrics['y_pred']) if isinstance(metrics['y_pred'][0], np.ndarray) else metrics['y_pred']
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            return acc, loss, balanced_acc
        
        return acc, loss, None

    def _save_best_model(self, epoch, acc, balanced_acc, args, iteration):
        """Save the best model checkpoint"""
        checkpoint_dir = f'/userHome/userhome1/automl_undergraduate/FER_Models/FER_Models/{args.model_name}/checkpoints/{args.dataset}'
        return save_best_model(epoch, acc, balanced_acc, self.model, args.dataset, self.optimizer, iteration, checkpoint_dir)

class POSTERTrainer(BaseTrainer):
    """POSTER specific trainer implementation"""
    
    def train_epoch(self, train_loader, epoch):
        metrics = self._initialize_metrics()
        self.model.train()
        
        for imgs, targets in train_loader:
            metrics['iter_cnt'] += 1
            self.optimizer.zero_grad()
            
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            outputs, features = self.model(imgs)
            CE_loss = self.criterion['CE'](outputs, targets)
            lsce_loss = self.criterion['lsce'](outputs, targets)
            loss = 2 * lsce_loss + CE_loss
            
            loss.backward()
            self.optimizer.step()
            
            metrics['running_loss'] += loss.item()
            _, predicts = torch.max(outputs, 1)
            metrics['correct_sum'] += torch.eq(predicts, targets).sum().item()
            
        return metrics
    
    def validate_epoch(self, val_loader):
        metrics = self._initialize_metrics()
        self.model.eval()
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                outputs, features = self.model(imgs)
                loss = self.criterion['CE'](outputs, targets)
                
                metrics['running_loss'] += loss.item()
                metrics['iter_cnt'] += 1
                _, predicts = torch.max(outputs, 1)
                metrics['correct_sum'] += torch.eq(predicts, targets).sum().cpu().item()
                
                metrics['y_true'].extend(targets.cpu().tolist())
                metrics['y_pred'].extend(predicts.cpu().tolist())
                
        return metrics

    def test(self, test_loader, checkpoint_path):
        """POSTER specific test implementation"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        metrics = self.validate_epoch(test_loader)
        
        test_acc, test_loss, balanced_acc = self._compute_metrics(metrics, len(test_loader.dataset))
        return test_acc * 100, balanced_acc * 100, test_loss

class DANTrainer(BaseTrainer):
    """DAN specific trainer implementation"""
    
    def train_epoch(self, train_loader, epoch):
        metrics = self._initialize_metrics()
        self.model.train()
        
        for imgs, targets in train_loader:
            metrics['iter_cnt'] += 1
            self.optimizer.zero_grad()
            
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            out, feat, heads = self.model(imgs)
            loss = (self.criterion['cls'](out, targets) + 
                   self.criterion['af'](feat, targets) + 
                   self.criterion['pt'](heads))
            
            loss.backward()
            self.optimizer.step()
            
            metrics['running_loss'] += loss.item()
            _, predicts = torch.max(out, 1)
            metrics['correct_sum'] += torch.eq(predicts, targets).sum().item()
            
        return metrics

    def validate_epoch(self, val_loader):
        metrics = self._initialize_metrics()
        self.model.eval()
        
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                out, feat, heads = self.model(imgs)
                loss = (self.criterion['cls'](out, targets) + 
                       self.criterion['af'](feat, targets) + 
                       self.criterion['pt'](heads))
                
                metrics['running_loss'] += loss.item()
                metrics['iter_cnt'] += 1
                _, predicts = torch.max(out, 1)
                metrics['correct_sum'] += torch.eq(predicts, targets).sum().cpu().item()
                
                metrics['y_true'].append(targets.cpu().numpy())
                metrics['y_pred'].append(predicts.cpu().numpy())
                
        return metrics

    def test(self, test_loader, checkpoint_path):
        """DAN specific test implementation"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        metrics = self.validate_epoch(test_loader)
        
        test_acc, test_loss, balanced_acc = self._compute_metrics(metrics, len(test_loader.dataset))
        return test_acc * 100, balanced_acc * 100, test_loss

class DDAMFNTrainer(BaseTrainer):
    """DDAMFN specific trainer implementation"""
    
    def train_epoch(self, train_loader, epoch):
        metrics = self._initialize_metrics()
        self.model.train()
        
        for imgs, targets in train_loader:
            metrics['iter_cnt'] += 1
            self.optimizer.zero_grad()
            
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            out, feat, heads = self.model(imgs)
            loss = self.criterion['cls'](out, targets) + 0.1 * self.criterion['at'](heads)
            
            loss.backward()
            self.optimizer.step()
            
            metrics['running_loss'] += loss.item()
            _, predicts = torch.max(out, 1)
            metrics['correct_sum'] += torch.eq(predicts, targets).sum().item()
            
        return metrics

    def test(self, test_loader, checkpoint_path):
        """DDAMFN specific test implementation"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        metrics = self.validate_epoch(test_loader)
        
        test_acc, test_loss, balanced_acc = self._compute_metrics(metrics, len(test_loader.dataset))
        return test_acc * 100, balanced_acc * 100, test_loss

class AdaDFTrainer(BaseTrainer):
    """AdaDF specific trainer implementation"""
    
    def __init__(self, model, criterion, optimizer, scheduler, args, device='cuda'):
        super().__init__(model, criterion, optimizer, scheduler, device)
        self.args = args
        self.dataset_config = AdaDF_DATASET_CONFIGS[args.dataset]
        self.LD = self._initialize_LD()
        
    def _initialize_LD(self):
        """Initialize Label Distribution matrix"""
        LD = torch.zeros(self.dataset_config['num_classes'], 
                        self.dataset_config['num_classes']).cuda()
        for i in range(self.dataset_config['num_classes']):
            LD[i] = (torch.zeros(self.dataset_config['num_classes'])
                    .fill_((1 - self.args.threshold) / 
                          (self.dataset_config['num_classes'] - 1))
                    .scatter_(0, torch.tensor(i), self.args.threshold))
        
        if self.args.sharpen:
            LD = torch.pow(LD, 1 / self.args.T) / torch.sum(torch.pow(LD, 1 / self.args.T), dim=1)
        return LD

    def _calculate_alpha(self, epoch):
        """Calculate alpha values for loss weighting"""
        if self.args.alpha is not None:
            return self.args.alpha, 1 - self.args.alpha
        else:
            if epoch <= self.args.beta:
                alpha_1 = np.exp(-(1 - epoch / self.args.beta) ** 2)
                alpha_2 = 1
            else:
                alpha_1 = 1
                alpha_2 = np.exp(-(1 - self.args.beta / epoch) ** 2)
            return alpha_1, alpha_2

    def train_epoch(self, train_loader, epoch):
        metrics = self._initialize_metrics()
        self.model.train()
        alpha_1, alpha_2 = self._calculate_alpha(epoch)
        
        outputs_list, targets_list, weights_list = [], [], []
        
        for images, labels, idxs in train_loader:
            metrics['iter_cnt'] += 1
            self.optimizer.zero_grad()
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            idxs = idxs.to(self.device)
            
            outputs_1, outputs_2, attention_weights = self.model(images)
            
            # Calculate losses
            loss = self._calculate_losses(outputs_1, outputs_2, labels, attention_weights, alpha_1, alpha_2)
            
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics['running_loss'] += loss.item() * images.size(0)
            _, predicted = torch.max(outputs_1.data, 1)
            metrics['correct_sum'] += (predicted == labels).sum().item()
            metrics['sample_cnt'] += images.size(0)
            
            # Store for LD update
            outputs_list.append(outputs_1.detach())
            targets_list.append(labels.detach())
            weights_list.append(attention_weights.detach())
            
        # Update LD matrix
        self._update_LD(outputs_list, targets_list)
        
        return metrics

    def _calculate_losses(self, outputs_1, outputs_2, labels, attention_weights, alpha_1, alpha_2):
        """Calculate all losses for AdaDF"""
        batch_size = labels.size(0)
        
        # CE Loss
        loss_ce = self.criterion['cls'](outputs_1, labels).mean()
        
        # Attention weights processing
        attention_weights = attention_weights.squeeze(1)
        attention_weights = ((attention_weights - attention_weights.min()) /
                           (attention_weights.max() - attention_weights.min())) * \
                          (self.args.max_weight - self.args.min_weight) + self.args.min_weight
        attention_weights = attention_weights.unsqueeze(1)
        
        # KLD Loss
        targets = (1 - attention_weights) * F.softmax(outputs_1, dim=1) + \
                 attention_weights * self.LD[labels]
        loss_kld = self.criterion['kld'](F.log_softmax(outputs_2, dim=1), targets).sum() / batch_size
        
        # Total loss
        return alpha_2 * loss_ce + alpha_1 * loss_kld

    def _update_LD(self, outputs_list, targets_list):
        """Update Label Distribution matrix"""
        if outputs_list and targets_list:
            outputs = torch.cat(outputs_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            self.LD = generate_adaptive_LD(outputs, targets, 
                                         self.dataset_config['num_classes'],
                                         self.args.threshold, 
                                         self.args.sharpen, 
                                         self.args.T)

    def test(self, test_loader, checkpoint_path):
        """AdaDF specific test implementation"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        metrics = self.validate_epoch(test_loader)
        
        test_acc, test_loss, balanced_acc = self._compute_metrics(metrics, len(test_loader.dataset))
        return test_acc, balanced_acc, test_loss

def get_trainer(model_name, model, criterion, optimizer, scheduler, args, device='cuda'):
    """Factory function to get appropriate trainer instance"""
    trainers = {
        'POSTER': POSTERTrainer,
        'DAN': DANTrainer,
        'DDAMFN': DDAMFNTrainer,
        'AdaDF': AdaDFTrainer
    }
    
    trainer_class = trainers.get(model_name.upper())
    if trainer_class is None:
        raise ValueError(f"Unknown model name: {model_name}")
        
    if model_name.upper() == 'ADADF':
        return trainer_class(model, criterion, optimizer, scheduler, args, device)
    return trainer_class(model, criterion, optimizer, scheduler, device)

def train(trainer, train_loader, val_loader, args, iteration):
    """Common training loop for all models"""
    best_acc = 0
    best_val_loss = float('inf')
    best_checkpoint_path = None
    patience_counter = 0
    
    for epoch in tqdm(range(1, args.epochs + 1)):
        # Training phase
        train_metrics = trainer.train_epoch(train_loader, epoch)
        train_acc, train_loss, _ = trainer._compute_metrics(train_metrics, len(train_loader.dataset))
        
        # Log training metrics
        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        tqdm.write(
            f"[{current_time}] [Epoch {epoch}/{args.epochs}] "
            f"Training accuracy: {train_acc:.4f}, "
            f"Loss: {train_loss:.3f}, "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Validation phase
        val_metrics = trainer.validate_epoch(val_loader)
        val_acc, val_loss, val_balanced_acc = trainer._compute_metrics(
            val_metrics, 
            len(val_loader.dataset)
        )
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Log validation metrics
        tqdm.write(
            f"[{current_time}] [Epoch {epoch}/{args.epochs}] "
            f"Validation accuracy: {val_acc:.4f}, "
            f"Val Balanced accuracy: {val_balanced_acc:.4f}, "
            f"Validation Loss: {val_loss:.3f}"
        )
        
        # Save best model
        if val_acc > best_acc:
            best_checkpoint_path = trainer._save_best_model(epoch, val_acc, val_balanced_acc, args, iteration)
            best_acc = val_acc
            
        # Early stopping check
        stop, best_val_loss, patience_counter = check_early_stopping(val_loss, best_val_loss, patience_counter, args.early_stopping_patience)
        if stop:
            tqdm.write(f"Early stopping triggered after {epoch} epochs")
            break
            
    return best_checkpoint_path, best_acc, best_val_loss