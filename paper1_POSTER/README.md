# POSTER
논문 링크: [POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition](https://arxiv.org/pdf/2204.04083.pdf)


## Preparation
### Data Preparation

Make sure RAF-DB dataset have a structure like following:
 
	```
	- data/raf-basic/
		 EmoLabel/
		     list_patition_label.txt
		 Image/aligned/
		     train_00001_aligned.jpg
		     test_0001_aligned.jpg
		     ...
	```

### Pretrained model weights
Dowonload pretrain weights (Image backbone and Landmark backbone) from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing). Put entire `pretrain` folder under `models` folder.

	```
	- models/pretrain/
		 ir50.pth
		 mobilefacenet_model_best.pth.tar
		     ...
	```
