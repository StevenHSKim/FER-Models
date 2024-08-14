# DAN

논문 링크: [Distract Your Attention: Multi-head Cross Attention Network for Facial Expression Recognition](https://arxiv.org/pdf/2109.07270.pdf)
  
## Preparation
### Data Preparation

Make sure RAF-DB dataset have a structure like following:
 
	```
	- datasets/raf-basic/
		 EmoLabel/
		     list_patition_label.txt
		 Image/aligned/
		     train_00001_aligned.jpg
		     test_0001_aligned.jpg
		     ...
	```

### Pretrained model weights
Dowonload pre-trained model of [MSCeleb](https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view?usp=sharing). Move the file to `./models`

	```
	- models/
		resnet18_msceleb.pth
	```
