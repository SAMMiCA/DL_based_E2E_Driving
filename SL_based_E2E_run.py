#!/usr/bin/env python
import os
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from control_msgs.msg import VehicleState
from std_msgs.msg import Float32, Int16MultiArray

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from SL_based_E2E_model import Net

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = ""
model = Net().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

class e2e_sl:
    def __init__(self):
        self.pub_steer = rospy.Publisher('/e2e_steer', Float32, queue_size=10)
        self.sub_image = rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, self.callback_image)
        self.sub_steer = rospy.Subscriber('/vehicle_state', VehicleState, self.callback_steer)
        self.sub_indicator = rospy.Subscriber('/indicator', Int16MultiArray, self.callback_indicator)
        
        self.bridge = CvBridge()

        self.transform = transforms.Compose([transforms.Resize((64, 200)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5), (0.5))
                                             ])        

    def callback_indicator(self, data):
        indicator_tuple = data.data
        indicator_numpy = np.asarray([indicator_tuple])
        self.indicator = torch.from_numpy(indicator_numpy).float() * 1000

    def callback_steer(self, data):
        self.label_steer = data.steer_angle

    def callback_image(self, frame):
        time = frame.header

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(frame)
        except CvBridgeError as e:
            print(e)
        
        cv_roi = cv_image[375:480, 0:640].copy()
        cv_roi = cv2.cvtColor(cv_roi, cv2.COLOR_RGB2BGR)
        img_PIL = Image.fromarray(cv_roi)
        image = self.transform(img_PIL)
        image = image.unsqueeze(0)

        image = image.to(DEVICE)
        self.indicator = self.indicator.to(DEVICE)
        result_steer = model(image, self.indicator)
        result_steer = result_steer.item()

        steer_msg = Float32()
        steer_msg.data = result_steer
        self.pub_steer.publish(steer_msg)

        print("Result Angle : {:.4f}".format(result_steer))

if __name__ == "__main__":

    try:
        rospy.init_node('e2e_sl')
        e2e_sl()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
