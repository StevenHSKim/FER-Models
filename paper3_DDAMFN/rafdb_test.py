import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from networks.DDAM import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/datasets/raf-basic/', help='Raf-DB dataset path.')
    parser.add_argument('--model_path', default = './checkpoints/rafdb.pth')
    
    parser.add_argument('--gpu', type=str, default='0', help='Assign a single GPU by its number.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    
    # DDAMFN
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    
    # ------------------------------- 하이퍼 파라미터 정리 (default 값 사용) -------------------------------
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    # ------------------------------------------------------------------------------------------------
    
    return parser.parse_args()

               
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt)+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,names=['name','label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']  
def run_test():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet(num_class=7,num_head=args.num_head)
    # checkpoint = torch.load(args.model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to(device)
    
    # 모델이 DataParallel로 저장되고 있으므로 아래의 전처리 필요 ('module.' 을 모두 제거)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()   

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   
  
    val_dataset = RafDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
  
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        out,feat,heads = model(imgs)

        _, predicts = torch.max(out, 1)
        correct_num  = torch.eq(predicts,targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out.size(0)
        
        if iter_cnt == 0:
            all_predicted = predicts
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicts),0)
            all_targets = torch.cat((all_targets, targets),0)                  
        iter_cnt+=1        

    acc = bingo_cnt.float()/float(sample_cnt)
    acc = np.around(acc.numpy(),4)

    print("Validation accuracy:%.4f. " % ( acc))
                
    # Compute confusion matrix
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    # Plot normalized confusion matrix
    plot_confusion_matrix(matrix, classes=class_names, normalize=True, title= 'RAF-DB Confusion Matrix (acc: %0.2f%%)' %(acc*100))
     
    plt.savefig(os.path.join('/userHome/userhome1/automl_undergraduate/FER_Models/DDAMFN/checkpoints', "rafdb"+"_acc"+str(acc)+"_bacc"+".png"))
    plt.close()
        
if __name__ == "__main__":        
    run_test()
    