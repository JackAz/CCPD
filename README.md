# CCPD: Chinese City Parking Dataset

## forked from https://github.com/detectRecog/CCPD

日常搬砖，端到端的车牌检测项目，RPnet训练自CCPD数据集（图片数超过30万），代码根据自己需求做了些修改：

```

  python image_detect.py -i [demo/rs1.jpg] -m [***/fh02.pth]
    
  python video_detect.py -m [***/fh02.pth]  
  
  python train_Bbox.py -i [***/ccpd/train]
  
  python train_rpnet.py -i [***/ccpd/train] -b 5 -se 0 -f [***/ccpd/rpnet] -t [***/ccpd/dev]
  
  python Eval.py -m [***/fh02.pth] -i [***/ccpd/test] -s [***/ccpd/failure_save]
  
```

CCPD的图片分辨率是720*1160，如果检测不同尺寸的图片，检测效果是不理想的。感觉一方面是进入cnn前先手动做了resize，另一方面可能训练的图片是这个尺寸，训练出来可能还是只能看这个尺寸吧，菜鸡的分析。所以在落地应用的时候，打算改一下网络结构，加个自适应池化，这样后面连FC层也无压力了。重新训练一下，emmm，过拟合了调参好麻烦，就放弃了。

曲线救国，首先可以规定输入的尺寸，比如我摄像头输入就是1920*1280的，但是opencv的设置好像还有点问题。然后对输入图片做切割，循环取 ` img_data[580*j:580*j+1160, 360*i:360*i+720]`
多个检测结果综合一下，这部分代码在image_detect.py中，还在完善。

数据集制作很辛苦，感谢大佬的开源！美中不足是安徽车居多，很多时候检测出来都是皖。

## CCPD2019 is now publicly available, is much more challenging with over 300k images and all annotations are refined. (If you are benefited from this dataset, please cite our paper.) It can be downloaded from:
 - [Google Drive the first part](https://drive.google.com/open?id=1AX2U3K9V-UpB8TjiVH8pL3tetyPt3f0p) , [Google Drive the second part](https://drive.google.com/open?id=1Zg3MtIvDoi83B2bkT0hionMxPNceHUpV) 
 
 - [BaiduYun Drive](https://pan.baidu.com/s/1z1HWBe671Gn2ZAOApf9huA)

This repository is designed to provide an open-source dataset for license plate detection and 
recognition, described in _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. This dataset is open-source under MIT license. More details about this dataset are avialable at our ECCV 2018 paper (also available in this github) _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_. If you are benefited from this paper, please cite our paper as follows:

```
@inproceedings{xu2018towards,
  title={Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline},
  author={Xu, Zhenbo and Yang, Wei and Meng, Ajin and Lu, Nanxue and Huang, Huan},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={255--271},
  year={2018}
}
```

## Downloads(Dataset and models):

### The google drive link for directly downloading the whole dataset: [google drive 12GB](https://drive.google.com/open?id=1fFqCXjhk7vE9yLklpJurEwP9vdLZmrJd). 

### The baiduyun link for directly downloading the whole dataset: [.zip(14GB)](https://pan.baidu.com/s/1SFUy5HlImM9w-Tw9kVuLZw), [.tar.bz2(12GB)](https://pan.baidu.com/s/1FH6pFOFF2MwyWiqn6vCzGA).

### The nearly well-trained model for testing and fun (Short of time, trained only for 5 epochs, but enough for testing): 

- Location module wR2.pth [google_drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-), [baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)
- rpnet model fh02.pth [google_drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv), [baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ).


## Training instructions

Input parameters are well commented in python codes(python2/3 are both ok, the version of pytorch should be >= 0.3). You can increase the batchSize as long as enough GPU memory is available.

#### Enviorment (not so important as long as you can run the code): 

- python: pytorch(0.3.1), numpy(1.14.3), cv2(2.4.9.1). 
- system: Cuda(release 9.1, V9.1.85)

#### For convinence, we provide a trained wR2 model and a trained rpnet model, you can download them from google drive or baiduyun.

First train the localization network (we provide one as before, you can download it from [google drive](https://drive.google.com/open?id=1l_tIt7D3vmYNYZLOPbwx8qJpPVM82CP-) or [baiduyun](https://pan.baidu.com/s/1Q3fPDHFYV5uibWwIQxPEOw)) 

After wR2 finetunes, we train the RPnet (we provide one as before, you can download it from [google drive](https://drive.google.com/open?id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv) or [baiduyun](https://pan.baidu.com/s/1sA-rzn4Mf33uhh1DWNcRhQ)) Please specify the variable wR2Path (the path of the well-trained wR2 model) in rpnet.py.


## Dataset Annotations

Annotations are embedded in file name.

A sample image name is "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg". Each name can be splited into seven fields. Those fields are explained as follows.

- **Area**: Area ratio of license plate area to the entire picture area.

- **Tilt degree**: Horizontal tilt degree and vertical tilt degree.

- **Bounding box coordinates**: The coordinates of the left-up and the right-bottom vertices.

- **Four vertices locations**: The exact (x, y) coordinates of the four vertices of LP in the whole image. These coordinates start from the right-bottom vertex.

- **License plate number**: Each image in CCPD has only one LP. Each LP number is comprised of a Chinese character, a letter, and five letters or numbers. A valid Chinese license plate consists of seven characters: province (1 character), alphabets (1 character), alphabets+digits (5 characters). "0_0_22_27_27_33_16" is the index of each character. These three arrays are defined as follows. The last character of each array is letter O rather than a digit 0. We use O as a sign of "no character" because there is no O in Chinese license plate characters.
```
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
```

- **Brightness**: The brightness of the license plate region.

- **Blurriness**: The Blurriness of the license plate region.



## Acknowledgement

If you have any problems about CCPD, please contact detectrecog@gmail.com.



Please cite the paper _《Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline》_, if you benefit from this dataset.

我只是搬砖（手动滑稽）