# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection

##############################################################

# Included:
#     1. Crop (requiring modification of bbox)
#     2. Translation (requiring modification of bbox)
#     3. Adjust brightness
#     4. Add noise
#     5. Rotate angle (requiring modification of bbox)
#     6. Mirror (requiring modification of bbox) #     7. cutout
# Note:
#     random.seed() will ensure that the same seed value will result in the same sequence of random numbers!!

import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
from xml_helper import *
import time


def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    '''
    Input:
        img: Image array
        bboxes: List of all bounding boxes in the image, in the format [[x_min, y_min, x_max, y_max]...]
        names: Name corresponding to each box 
'''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
    cv2.namedWindow('pic', 0) 
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

# All images are read by cv2.
class DataAugmentForObjectDetection():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5, 
                crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                add_noise_rate=0.5, flip_rate=0.5, 
                cutout_rate=0.5, cut_out_length=50, cut_out_holes=1, cut_out_threshold=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate
        self.cutout_rate = cutout_rate

        self.cut_out_length = cut_out_length
        self.cut_out_holes = cut_out_holes
        self.cut_out_threshold = cut_out_threshold
    
    # Add noise
    def _addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        '''
        Input:
            img: Image array
            Output:
            Noisy image array after adding noise. Since the output pixels are within the range of [0, 1], they need to be multiplied by 255.
        '''
        # random.seed(int(time.time())) 
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        allowedtypes = [
            'gaussian',
            'localvar',
            'poisson',
            'salt',
            'pepper',
            's&p',
            'speckle']
        mode=random.choice(allowedtypes)
        return random_noise(img, mode=mode, clip=True)*255

    
    # 调整亮度
    # Adjust brightness
    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5) #flag>1 indicates dimming, while values less than 1 indicate brightening.
        return exposure.adjust_gamma(img, flag)
    
    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
        '''
        Original Version: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        Randomly mask out one or more patches from an image.
        Args:
            img : a 3D numpy array,(h,w,c)
            bboxes : The coordinates of the frame
            n_holes (int): Number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        
        def cal_iou(boxA, boxB):
            '''
            boxA and boxB are two boxes. Return the IoU.
            boxB为bouding box
            '''

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            if xB <= xA or yB <= yA:
                return 0.0

            # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            # iou = interArea / float(boxAArea + boxBArea - interArea)
            iou = interArea / float(boxBArea)

            # return the intersection over union value
            return iou

        # Obtain h and w
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            
            chongdie = True    #Check whether the cut area overlaps too much with the box.
            
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - length // 2, 0, h)    
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                chongdie = False
                for box in bboxes:
                    if cal_iou([x1,y1,x2,y2], box) > threshold:
                        chongdie = True
                        break
            
            mask[y1: y2, x1: x2, :] = 0.
        
        # mask = np.expand_dims(mask, axis=0)
        img = img * mask

        return img

    # 旋转
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        '''
        Reference: https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        Input:
        img: Image array, (h, w, c)
        bboxes: All bounding boxes contained in this image, a list, each element is [x_min, y_min, x_max, y_max], make sure they are numerical values
        angle: Rotation angle
        scale: Default 1
        Output:
        rot_img: Rotated image array
        rot_bboxes: Rotated bounding box coordinates list 
        '''
        #---------------------- Rotating Images ----------------------
        w = img.shape[1]
        h = img.shape[0]
        # Angle change radian
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        #---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        #---------------------- Correcting the bbox coordinates ----------------------
        # rot_mat is the final rotation matrix
        # Obtain the four midpoints of the original bbox, and then transform these four points to the coordinate system after rotation.
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
            # Merge np.array
            concat = np.vstack((point1, point2, point3, point4))
            # Change the array type
            concat = concat.astype(np.int32)
            # Obtain the coordinates after rotation
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max,bbox[4]])
        
        return rot_img, rot_bboxes

    # Crop
    def _crop_img_bboxes(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        '''
        The cropped image should contain all the bounding boxes.
        Input:
        img: Image array
        bboxes: All bounding boxes contained in this image, a list, each element is [x_min, y_min, x_max, y_max], make sure they are numerical values
        Output:
        crop_img: Cropped image array
        crop_bboxes: List of coordinates of the cropped bounding boxes 
        '''
        #---------------------- Crop Image ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   The smallest bounding box that has been cropped and contains all the target boxes.
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           
        d_to_right = w - x_max      
        d_to_top = y_min            
        d_to_bottom = h - y_max     

        # Randomly expand this minimum bounding box
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # Randomly expand this minimum bounding box to prevent it from being too small in the cutout operation.
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # Ensure not to exceed the boundaries
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        #---------------------- Crop bounding box ----------------------
        # Calculation of the bounding box coordinates after cropping
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min,bbox[4]])
        
        return crop_img, crop_bboxes
  
    # Translation:
    def _shift_pic_bboxes(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        '''
        After translation, the image should contain all the bounding boxes.
        Input:
            img: Image array
        bboxes: All bounding boxes contained in this image, a list, each element is [x_min, y_min, x_max, y_max], ensuring they are numerical values
        Output:
            shift_img: Translated image array
            shift_bboxes: List of coordinates of the translated bounding boxes
        '''
        
        #---------------------- Translation of Image by Shifting ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w   # The smallest bounding box that has been cropped and contains all the target boxes.
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        
        d_to_left = x_min           #The maximum leftward displacement of the bounding box that encompasses all the targets.
        d_to_right = w - x_max      #The maximum rightward displacement of the largest target box
        d_to_top = y_min            #The maximum upward displacement of the largest target box included
        d_to_bottom = h - y_max     #The maximum downward displacement distance that encompasses all target boxes

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])  #x represents the pixel value for moving left or right, positive for moving right and negative for moving left; y represents the pixel value for moving up or down, positive for moving down and negative for moving up.
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y,bbox[4]])

        return shift_img, shift_bboxes

    # Mirror image
    def _filp_pic_bboxes(self, img, bboxes):
        '''
            参考:https://blog.csdn.net/jningwei/article/details/78753607
            平移后的图片要包含所有的框
            输入:
                img:图像array
                bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            输出:
                flip_img:平移后的图像array
                flip_bboxes:平移后的bounding box的坐标list
        '''
        '''
        After translation, the image should contain all the bounding boxes.
        Input:
        img: Image array
        bboxes: All bounding boxes contained in this image, a list, each element is [x_min, y_min, x_max, y_max], ensuring it is numerical
        Output:
        flip_img: Translated image array
        flip_bboxes: List of coordinates of the translated bounding boxes 
        '''

        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:   
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon: 
            flip_img =  cv2.flip(flip_img, 1)   
        else:
            flip_img = cv2.flip(flip_img, 0)

       
        flip_bboxes = list()
        for box in bboxes:
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            if horizon:
                flip_bboxes.append([w-x_max, y_min, w-x_min, y_max,box[4]])
            else:
                flip_bboxes.append([x_min, h-y_max, x_max, h-y_min,box[4]])

        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        '''
        Image Enhancement
        Input:
        img: Image array
        bboxes: Coordinates of all bounding boxes in this image
        Output:
        img: Enhanced image
        bboxes: Corresponding bounding boxes of the enhanced image '''
        change_num = 0  # Frequency of Changes
        print('------')
        while change_num < 1:   
            if random.random() < self.crop_rate:        # Crop
                #print('裁剪')
                change_num += 1
                img, bboxes = self._crop_img_bboxes(img, bboxes)

            '''
            if random.random() > self.rotation_rate:    # Rotation
                #print('旋转')
                change_num += 1
                # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
            '''

            if random.random() < self.shift_rate:       # Translatio
                #print('平移')
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)
            
            if random.random() > self.change_light_rate: # Adjust Brightness
                #print('亮度')
                change_num += 1
                img = self._changeLight(img)
            
            if random.random() < self.add_noise_rate:    # Add noise
                #print('加噪声')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  #cutout
                #print('cutout')
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)

            # if random.random() < self.flip_rate:    # Inversion
            #     print('翻转')
            #     change_num += 1
            #     img, bboxes = self._filp_pic_bboxes(img, bboxes)

            #print('\n')
        # print('------')
        return img, bboxes
            

if __name__ == '__main__':

    need_aug_num = 10 # Number of Image Enlargements per Picture
    out_root_path = r"F:\Code_Dataset\Fish_C\yolov5\DataAugForObjectDetection\data\train" # Output Path

    source_pic_root_path = r"F:\Code_Dataset\Fish_C\yolov5\DataAugForObjectDetection\data\dataset\train\images"
    source_xml_root_path = r"F:\Code_Dataset\Fish_C\yolov5\DataAugForObjectDetection\data\dataset\train\labels"
    dataAug = DataAugmentForObjectDetection()
    if not os.path.exists(out_root_path):
        os.makedirs(out_root_path)

    s=time.time()
    for parent, _, files in os.walk(source_pic_root_path):
        for i,file in enumerate(files):
            cnt = 0
            while cnt < need_aug_num:
                pic_path = os.path.join(parent, file)
                xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
                coords = parse_xml(xml_path)        
                #coords = [coord[:5] for coord in coords]


                img = cv2.imread(pic_path)
                #show_pic(img, coords)    

                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                cnt += 1
                print(i+1,'/',len(files),file,cnt)


                aug_img_name = 'aug_' + str(cnt) + '_' + file


                #auged_bboxes=[box.append('table') for box in auged_bboxes]
                #for box in auged_bboxes:
                #    box.append('table')

                generate_xml(aug_img_name,auged_img, auged_bboxes, out_root_path)

                out_img_path=os.path.join(out_root_path,'images')
                if not os.path.exists(out_img_path):
                    os.makedirs(out_img_path)
                cv2.imwrite(os.path.join(out_img_path,aug_img_name),auged_img)
                #show_pic(auged_img, auged_bboxes)  


    e=time.time()
    print('耗时：',e-s)
