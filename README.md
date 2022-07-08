# DeepLabv3Plus-Pytorch-ROS


このパッケージは、[DeepLabV3Plus-Pytorchに](https://github.com/VainF/DeepLabV3Plus-Pytorch)基づくパクってつくたROSパケージです。

### add_cmd_velブランチに対しての説明を追加しています。


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
VOCデータセットとCityscapesの２種類があります

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

好きなモデルを選択する

#### ckpt：
| Model                   | Batch Size | FLOPs | train/val OS | mIoU  |                                               Dropbox                                                |                Tencent Weiyun                 | 
|:------------------------|:----------:|:-----:|:------------:|:-----:|:----------------------------------------------------------------------------------------------------:|:---------------------------------------------:|
| DeepLabV3-MobileNet     |     16     | 6.0G  |    16/16     | 0.701 |   [Download](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/A4ubD1DD) |
| DeepLabV3-ResNet50      |     16     | 51.4G |    16/16     | 0.769 |   [Download](https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=0)    | [Download](https://share.weiyun.com/33eLjnVL) |
| DeepLabV3-ResNet101     |     16     | 72.1G |    16/16     | 0.773 |   [Download](https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=0)   | [Download](https://share.weiyun.com/iCkzATAw) |
| DeepLabV3Plus-MobileNet |     16     | 17.0G |    16/16     | 0.711 | [Download](https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=0) | [Download](https://share.weiyun.com/djX6MDwM) |
| DeepLabV3Plus-ResNet50  |     16     | 62.7G |    16/16     | 0.772 | [Download](https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=0)  | [Download](https://share.weiyun.com/uTM4i2jG) |
| DeepLabV3Plus-ResNet101 |     16     | 83.4G |    16/16     | 0.783 | [Download](https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=0) | [Download](https://share.weiyun.com/UNPZr3dk) |

好きなcheck pointをダウンロードもしくはカスタマイズのcheak pointを```cheak_points```に入れる

#### ほかのパラメータ：
[本家](https://github.com/VainF/DeepLabV3Plus-Pytorch)を参照して適当に変える

ほぼ同じだから多分大丈夫

ノードと話題なら適当に変えればいい

## sementation to cmd_vel
### Quick Start
```
$ roslaunch deeplabv3_plus_pytorch_ros seg2cmd.launch 
```
起動する時、セグメンテーションも自動的に起動する。
### 新しいパラメータ
```zone_status_out```：セグメンテーションのノードが区域の状態を出力するかどうか


```zone_status_sub```：セグメンテーションのノードをpublishするtopic名


```cmd_vel_pub```：cmd_velのtopic名


他には速度とかの設定、気軽に変更してもよい。


```change_mode```：これを変更すれば　```segmentation_cmd_vel.py```の中のcmd_vel変換関数を自分で書くことができる

### 注意
640x480の画像を使うので、画像サイズを変えると動かなくなる。

## 今確認したのエラー
```
ImportError: dynamic module does not define module export function (PyInit_`cv_bridge`_boost ) 
```
Ubuntu 20.04以下の環境では動く時のエラーです。

python３のcv_bridgeを使うと解決できる。



```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```
起動する時出る可能性があるエラーです。

無視してしばらく安定したら大丈夫。

## Note
Issuesは中国語と日本語対応、英語じゃなくてもいい

Ubuntu 20.04　Python3.9で動作確認済み、動かない場合は多分自力で解決するほうが早い

