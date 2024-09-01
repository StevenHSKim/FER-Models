'''
모델의 훈련, 검증, 테스트를 담당하는 트레이너 클래스를 정의
'''


import os
import torch
import matplotlib.pyplot as plt
from time import time
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from sklearn.metrics import confusion_matrix

from utils.utils import *
from utils.metrics import accuracy_, balanced_accuracy_, plot_confusion_matrix


class Trainer:
    """
    모델 훈련과 검증을 관리하는 클래스
    """
    def __init__(self, config, model, device):
        """
        Config를 이용하여 트레이너 객체를 초기화
        """
        self.config = config
        self.model = model.to(device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.best_acc = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.save_path = None
        self.train_loss_dict = {}
        self.use_data_parallel = config.get('use_data_parallel', False)
        if self.use_data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            
            
    def _build_optimizer(self):
        """
        optimizer 함수를 설정
        """
        optimizer_type = self.config['optimizer'].lower()
        if optimizer_type == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        elif optimizer_type == 'adam':
            optimizer = Adam(self.model.parameters(), lr=self.config['lr'])
        elif optimizer_type == 'sgd':
            optimizer = SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config.get('momentum', 0.9))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return optimizer


    def _build_scheduler(self):
        """
        learning rate scheduler을 설정
        """
        scheduler_type = self.config['lr_scheduler'].lower()
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config['t_max'])
        elif scheduler_type == 'step':
            scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'exp':
            scheduler = ExponentialLR(self.optimizer, gamma=0.98)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return scheduler


    def train(self, train_loader, val_loader, epochs=None, patience=None):
        """
        모델 훈련 함수
        """
        epochs = epochs or self.config['epochs']
        patience = patience or self.config['early_stopping_patience']

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss, correct_sum, iter_cnt = 0.0, 0, 0
            start_time = time()

            for imgs, targets in train_loader:
                iter_cnt += 1
                self.optimizer.zero_grad()
                imgs, targets = imgs.to(self.config['device']), targets.to(self.config['device'])
                outputs = self.model(imgs)
                loss = self.model.calculate_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                _, predicts = torch.max(outputs, 1)
                correct_sum += torch.eq(predicts, targets).sum().item()

            train_acc = accuracy_(predicts, targets)
            train_loss /= iter_cnt
            elapsed = (time() - start_time) / 60
            print(f'[Epoch {epoch}] Train time: {elapsed:.2f}, Training accuracy: {train_acc:.4f}, Loss: {train_loss:.3f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            self.scheduler.step()

            if self._validate(val_loader, epoch):
                print("Early stopping triggered.")
                break

        return self.model, self.best_acc, self.save_path


    def _validate(self, val_loader, epoch):
        """
        train 함수에서 사용하는 내장 validation 함수
        """
        self.model.eval()
        val_loss, bingo_cnt = 0.0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(self.config['device']), targets.to(self.config['device'])
                outputs = self.model(imgs)
                loss = self.model.calculate_loss(outputs, targets)
                val_loss += loss.item()
                _, predicts = torch.max(outputs, 1)
                bingo_cnt += torch.eq(predicts, targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = accuracy_(predicts, targets)
        print(f"[Epoch {epoch}] Validation accuracy: {val_acc:.4f}, Loss: {val_loss:.3f}")

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            current_time = get_local_time()  # Use local time for file naming
            self.save_path = os.path.join(self.config.get('checkpoint_dir', '.'), f"epoch{epoch}_acc{val_acc:.4f}_{current_time}.pth")
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, self.save_path)
            print(f'Model saved at {current_time}.')

        # Early stopping check
        self.best_loss, self.patience_counter, should_stop = early_stopping(val_loss, self.best_loss, self.patience_counter, self.config['early_stopping_patience'])
        return should_stop


    def test(self, test_loader, checkpoint_path):
        """
        모델 테스트 함수
        """
        print("Loading pretrained weights...", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        total_correct, total_loss = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs, targets = imgs.to(self.config['device']), targets.to(self.config['device'])
                outputs = self.model(imgs)
                loss = self.model.calculate_loss(outputs, targets)
                total_loss += loss.item()
                _, predicts = torch.max(outputs, 1)
                total_correct += torch.eq(predicts, targets).sum().item()
                y_pred.extend(predicts.cpu().numpy())
                y_true.extend(targets.cpu().numpy())

        test_acc = total_correct / len(test_loader.dataset)
        test_loss = total_loss / len(test_loader)
        balanced_acc = balanced_accuracy_(y_pred, y_true)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=test_loader.dataset.classes, normalize=True, title='Confusion Matrix', save_path='path/to/save')

        print(f"Test accuracy: {test_acc:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Running Loss: {test_loss:.3f}")

        return test_acc, balanced_acc, test_loss, cm


    def plot_train_loss(self, show=True, save_path=None):
        """
        학습 loss를 시각화
        """
        epochs = list(self.train_loss_dict.keys())
        losses = [self.train_loss_dict[epoch] for epoch in sorted(epochs)]
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, losses, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)
