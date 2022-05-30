# DeepLabv3Plus-Pytorch-ROS


このパッケージは、[DeepLabV3Plus-Pytorchに](https://github.com/VainF/DeepLabV3Plus-Pytorch)基づくパクってつくたROSパケージです。

## インストール 
このリポジトリをcatkin_wsのsrcにクローンして
```bash
$ git clone https://github.com/Tsumoridesu/deeplabv3_plus_pytorch_ros.git
```
ビルド
```bash
$ cd ../
$ catkin build deeplabv3_plus_pytorch_ros
$ source ~/catkin_ws/devel/setup.bash 
```

##  Quick Start
### 1. 必要パケージのインストール
```bash
$ pip install -r requirements.txt
```

### 2. 起動
```bash
$ roslaunch deeplabv3_plus_pytorch_ros segmentation.launch
```

### 3. パラメータ調整
#### dataset:
VOCデータセットとCityscapes２種類があります

VOCの場合：```year 2012_aug```

Cityscapesの場合 : ```cityscapes```

#### model:
|      DeepLabV3       |        DeepLabV3+        |
|:--------------------:|:------------------------:|
|  deeplabv3_resnet50  |  deeplabv3plus_resnet50  |
| deeplabv3_resnet101  | deeplabv3plus_resnet101  |
| deeplabv3_mobilenet  | deeplabv3plus_mobilenet  ||
| deeplabv3_hrnetv2_48 | deeplabv3plus_hrnetv2_48 |
| deeplabv3_hrnetv2_32 | deeplabv3plus_hrnetv2_32 |

好きのモデルを選択すればいい

#### ckpt：
| Model                   | Batch Size | FLOPs | train/val OS | mIoU  |                                               Dropbox                                                |                Tencent Weiyun                 | 
|:------------------------|:----------:|:-----:|:------------:|:-----:|:----------------------------------------------------------------------------------------------------:|:---------------------------------------------:|
| DeepLabV3-MobileNet     |     16     | 6.0G  |    16/16     | 0.701 |   [Download](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/A4ubD1DD) |
| DeepLabV3-ResNet50      |     16     | 51.4G |    16/16     | 0.769 |   [Download](https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=0)    | [Download](https://share.weiyun.com/33eLjnVL) |
| DeepLabV3-ResNet101     |     16     | 72.1G |    16/16     | 0.773 |   [Download](https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/iCkzATAw) |
| DeepLabV3Plus-MobileNet |     16     | 17.0G |    16/16     | 0.711 | [Download](https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=0) | [Download](https://share.weiyun.com/djX6MDwM) |
| DeepLabV3Plus-ResNet50  |     16     | 62.7G |    16/16     | 0.772 | [Download](https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=0)  | [Download](https://share.weiyun.com/uTM4i2jG) |
| DeepLabV3Plus-ResNet101 |     16     | 83.4G |    16/16     | 0.783 | [Download](https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0) | [Download](https://share.weiyun.com/UNPZr3dk) |

好きのウェイトをダンロードもしくはカスタマイズのウェイトを```cheak_points```に入れる

#### ほかのパラメータ：
[本家](https://github.com/VainF/DeepLabV3Plus-Pytorch)に参照して適当に変える

ほぼ同じだから多分大丈夫

ノードと話題なら適当に変えばいい、多分

## Note
Issuesは中国語と日本語対応、英語じゃなくでもいい

Ubuntu 20.04　Python3.9で動けるはず、動けない場合なら多分自力で解決するほうが早い

