#!/usr/bin/env python3
# codding = utf-8

# Python imports
import numpy as np
import cv2
import os

# ROS imports
import rospy

# Deep Learning imports
import torch
import torch.nn as nn
from torchvision import transforms as T
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from deeplabv3_plus_pytorch_ros.msg import zone_status
import network
import utils
from datasets import VOCSegmentation, Cityscapes, cityscapes


class image_segmentation:
    def __init__(self):
        # Name of training set
        self.dataset = rospy.get_param('~dataset', 'cityscapes')
        rospy.loginfo("Use Dataset name %s", self.dataset)

        # Deeplab Options
        available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                                  not (name.startswith("__") or name.startswith('_')) and callable(
            network.modeling.__dict__[name])
                                  )
        # Model name
        self.model_mode = rospy.get_param('~model', 'deeplabv3plus_mobilenet')
        rospy.loginfo("Use Model name %s", self.model_mode)

        # Apply separable conv to decoder and aspp
        self.separable_conv = rospy.get_param('~separable_conv', False)
        rospy.loginfo("Separable conv to decoder and aspp: %s" % self.separable_conv)

        # Output stride
        self.output_stride = rospy.get_param('output_stride', 16)
        rospy.loginfo("Output stride: %s" % self.output_stride)

        # Train options
        # Resume from checkpoint(weights)
        self.ckpt = rospy.get_param('~ckpt')
        rospy.loginfo("Checkpoint path %s" % self.ckpt)

        self.save_val_results_to = None
        self.crop_val = False
        self.val_batch_size = 4
        self.crop_size = 513

        # Gpu ID
        # self.gpu_id = rospy.get_param('~gpu_id', 0)
        # rospy.loginfo("Gpu ID %s " % self.gpu_id)

        # Ros options
        self.image_pub_topic = rospy.get_param('~image_publish_topic', 'segmentation_image')
        self.image_pub = rospy.Publisher(self.image_pub_topic, Image, queue_size=1)
        self.zone_status_pub = rospy.Publisher('%s/zone_status' % self.image_pub_topic, zone_status, queue_size=1)

        self.bridge = CvBridge()

        self.image_sub_topic = rospy.get_param('~image_subscribe_topic', '/camera/image_raw')
        self.image_sub = rospy.Subscriber(self.image_sub_topic, Image, self.segmentation)

        # Predict options
        if self.dataset.lower() == 'voc':
            self.num_classes = 21
            self.decode_fn = VOCSegmentation.decode_target
        elif self.dataset.lower() == 'cityscapes':
            self.num_classes = 19
            self.decode_fn = Cityscapes.decode_target

        # os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % self.device)

        # Set up model (all models are 'constructed at network.modeling)
        self.model = network.modeling.__dict__[self.model_mode](num_classes=self.num_classes,
                                                                output_stride=self.output_stride)

        self.zone_status_out = rospy.get_param('~zone_status_out', False)

        if self.zone_status_out:
            # 左側
            left_mask = np.zeros([480, 640], np.uint8)
            points_left = np.array([[220, 240], [320, 240], [320, 480], [48, 480]])
            self.left_zone = cv2.fillPoly(left_mask, [points_left], color=255)
            self.sum_left_zone = np.sum(self.left_zone)

            # 右側
            right_mask = np.zeros([480, 640], np.uint8)
            points_right = np.array([[320, 240], [420, 240], [592, 480], [320, 480]])
            self.right_zone = cv2.fillPoly(right_mask, [points_right], color=255)
            self.sum_right_zone = np.sum(self.right_zone)

            # 直進危険領域
            danger_mask = np.zeros([480, 640], np.uint8)
            points_danger = np.array([[227, 360], [413, 360], [456, 480], [184, 480]])
            self.danger_zone = cv2.fillPoly(danger_mask, [points_danger], color=255)
            self.sum_danger_zone = np.sum(self.danger_zone)

            # 直進注意領域
            warning_mask = np.zeros([480, 640], np.uint8)
            points_warning = np.array([[270, 240], [370, 240], [413, 360], [227, 360]])
            self.warning_zone = cv2.fillPoly(warning_mask, [points_warning], color=255)
            self.sum_warning_zone = np.sum(self.warning_zone)

            # 左側探索領域
            turnleft_order_mask = np.zeros([480, 640], np.uint8)
            points_turnleft_order = np.array([[0, 480], [0, 360], [227, 360], [184, 480]])
            self.turnleft_order_zone = cv2.fillPoly(turnleft_order_mask, [points_turnleft_order], color=255)
            self.sum_turnleft_order_zone = np.sum(self.turnleft_order_zone)

            # 右側探索領域
            turnright_order_mask = np.zeros([480, 640], np.uint8)
            points_turnright_order = np.array([[640, 480], [640, 360], [413, 360], [456, 480]])
            self.turnright_order_zone = cv2.fillPoly(turnright_order_mask, [points_turnright_order], color=255)
            self.sum_turnright_order_zone = np.sum(self.turnright_order_zone)

            rospy.loginfo("zone_status_pub True")

        if self.separable_conv and 'plus' in self.model_mode:
            network.convert_to_separable_conv(self.model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)

        if self.ckpt is not None and os.path.isfile(self.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(self.ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(self.model)
            model.to(self.device)
            rospy.loginfo("Resume model from %s" % self.ckpt)
            del checkpoint
        else:
            rospy.loginfo("[!] Retrain")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
        with torch.no_grad():
            self.model = self.model.eval()

    def segmentation(self, data):
        if self.crop_val:
            T.Compose([
                T.Resize(self.crop_size),
                T.CenterCrop(self.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image = transform(cv_image).unsqueeze(0).to(self.device)
        pred = self.model(cv_image)
        pred = pred.max(1)[1].cpu().numpy()[0]
        colorized_preds = self.decode_fn(pred).astype('uint8')
        colorized_preds = cv2.cvtColor(np.asarray(colorized_preds), cv2.COLOR_RGB2BGR)

        if self.zone_status_out:
            # 通行可能領域抽出
            road_rgb = np.array([128, 64, 128])
            mask = cv2.inRange(colorized_preds, lowerb=road_rgb, upperb=road_rgb)

            # ノイズ消去
            kel_dil = np.ones((17, 17), np.uint8)
            kel_ero = np.ones((17, 17), np.uint8)
            # 膨張
            dil = cv2.dilate(mask, kel_dil)
            # 腐食
            ero = cv2.erode(dil, kel_ero)

            right_zone_img = cv2.bitwise_and(ero, ero, mask=self.right_zone)
            left_zone_img = cv2.bitwise_and(ero, ero, mask=self.left_zone)
            danger_zone_img = cv2.bitwise_and(ero, ero, mask=self.danger_zone)
            warning_zone_img = cv2.bitwise_and(ero, ero, mask=self.warning_zone)
            turnleft_order_zone_img = cv2.bitwise_and(ero, ero, mask=self.turnleft_order_zone)
            turnright_order_zone_img = cv2.bitwise_and(ero, ero, mask=self.turnright_order_zone)

            danger_zone_point = np.sum(danger_zone_img) / self.sum_danger_zone
            right_zone_point = np.sum(right_zone_img) / self.sum_right_zone
            left_zone_point = np.sum(left_zone_img) / self.sum_left_zone
            warning_zone_point = np.sum(warning_zone_img) / self.sum_warning_zone
            turnleft_order_zone_point = np.sum(turnleft_order_zone_img) / self.sum_turnleft_order_zone
            turnright_order_zone_point = np.sum(turnright_order_zone_img) / self.sum_turnright_order_zone

            msg = zone_status()
            msg.danger_zone_point = danger_zone_point
            msg.right_zone_point = right_zone_point
            msg.left_zone_point = left_zone_point
            msg.warning_zone_point = warning_zone_point
            msg.turnleft_order_zone_point = turnleft_order_zone_point
            msg.turnright_order_zone_point = turnright_order_zone_point

            self.zone_status_pub.publish(msg)

            right_zone_img = right_zone_img[:, :, np.newaxis]
            right_zone_img = right_zone_img.repeat([3], axis=2)
            right_zone_img *= np.uint8([50, 0, 0])
            cv2.addWeighted(colorized_preds, 1, right_zone_img, 0.3, 0, colorized_preds)

            left_zone_img = left_zone_img[:, :, np.newaxis]
            left_zone_img = left_zone_img.repeat([3], axis=2)
            left_zone_img *= np.uint8([50, 0, 0])
            cv2.addWeighted(colorized_preds, 1, left_zone_img, 0.3, 0, colorized_preds)

            danger_zone_img = danger_zone_img[:, :, np.newaxis]
            danger_zone_img = danger_zone_img.repeat([3], axis=2)
            danger_zone_img *= np.uint8([0, 0, 50])
            cv2.addWeighted(colorized_preds, 1, danger_zone_img, 0.2, 0, colorized_preds)

            warning_zone_img = warning_zone_img[:, :, np.newaxis]
            warning_zone_img = warning_zone_img.repeat([3], axis=2)
            warning_zone_img *= np.uint8([0, 0, 50])
            cv2.addWeighted(colorized_preds, 1, warning_zone_img, 0.5, 0, colorized_preds)

            turnleft_order_zone_img = turnleft_order_zone_img[:, :, np.newaxis]
            turnleft_order_zone_img = turnleft_order_zone_img.repeat([3], axis=2)
            turnleft_order_zone_img *= np.uint8([50, 50, 50])
            cv2.addWeighted(colorized_preds, 1, turnleft_order_zone_img, 0.3, 0, colorized_preds)

            turnright_order_zone_img = turnright_order_zone_img[:, :, np.newaxis]
            turnright_order_zone_img = turnright_order_zone_img.repeat([3], axis=2)
            turnright_order_zone_img *= np.uint8([50, 50, 50])
            cv2.addWeighted(colorized_preds, 1, turnright_order_zone_img, 0.3, 0, colorized_preds)




        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(colorized_preds, "bgr8"))
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    try:
        rospy.init_node("segmentation")
        rospy.loginfo("Satarting")
        image_segmentation()
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down")
