# -*- coding=utf-8 -*-
##############################################################
# description:
#     data augmentation for obeject detection

#Multi-process version
##############################################################



import time
import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
from xml_helper import parse_xml,generate_xml

from multiprocessing import Pool
import multiprocessing as mp
import time

def show_pic(img, bboxes=None):

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
    
    # 加噪声
    def _addNoise(self, img):

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

    

    def _changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5) 
        return exposure.adjust_gamma(img, flag)
    
    # cutout
    def _cutout(self, img, bboxes, length=100, n_holes=1, threshold=0.5):
      
        
        def cal_iou(boxA, boxB):
    

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

       
        if img.ndim == 3:
            h,w,c = img.shape
        else:
            _,h,w,c = img.shape
        
        mask = np.ones((h,w,c), np.float32)

        for n in range(n_holes):
            
            chongdie = True   
            
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

    
    def _rotate_img_bbox(self, img, bboxes, angle=5, scale=1.):

      
        w = img.shape[1]
        h = img.shape[0]
       
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
        
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

     
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
         
            concat = np.vstack((point1, point2, point3, point4))
         
            concat = concat.astype(np.int32)
          
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx+rw
            ry_max = ry+rh
           
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max,bbox[4]])
        
        return rot_img, rot_bboxes

   
    def _crop_img_bboxes(self, img, bboxes):

        w = img.shape[1]
        h = img.shape[0]
        x_min = w   
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

        
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

       
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
      
        crop_bboxes = list()
        for bbox in bboxes:
            crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min,bbox[4]])
        
        return crop_img, crop_bboxes
  

    def _shift_pic_bboxes(self, img, bboxes):

     
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  
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

        x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
        y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
        
        M = np.float32([[1, 0, x], [0, 1, y]])  
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

      
        shift_bboxes = list()
        for bbox in bboxes:
            shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y,bbox[4]])

        return shift_img, shift_bboxes


    def _filp_pic_bboxes(self, img, bboxes):
       
        import copy
        flip_img = copy.deepcopy(img)
        if random.random() < 0.5:   
            horizon = True
        else:
            horizon = False
        h,w,_ = img.shape
        if horizon: #水平翻转
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

        change_num = 0  
        print('------')
        while change_num < 1:   
            if random.random() < self.crop_rate:       
                #print('裁剪')
                change_num += 1
                img, bboxes = self._crop_img_bboxes(img, bboxes)

            '''
            if random.random() > self.rotation_rate:   
                #print('旋转')
                change_num += 1
                # angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                angle = random.sample([90, 180, 270],1)[0]
                scale = random.uniform(0.7, 0.8)
                img, bboxes = self._rotate_img_bbox(img, bboxes, angle, scale)
            '''

            if random.random() < self.shift_rate:       
                #print('平移')
                change_num += 1
                img, bboxes = self._shift_pic_bboxes(img, bboxes)
            
            if random.random() > self.change_light_rate:
                #print('亮度')
                change_num += 1
                img = self._changeLight(img)
            
            if random.random() < self.add_noise_rate:    
                #print('加噪声')
                change_num += 1
                img = self._addNoise(img)

            if random.random() < self.cutout_rate:  #cutout
                #print('cutout')
                change_num += 1
                img = self._cutout(img, bboxes, length=self.cut_out_length, n_holes=self.cut_out_holes, threshold=self.cut_out_threshold)

            # if random.random() < self.flip_rate:    
            #     print('翻转')
            #     change_num += 1
            #     img, bboxes = self._filp_pic_bboxes(img, bboxes)

            #print('\n')
        # print('------')
        return img, bboxes


def gen(parent,file,dataAug,need_aug_num,source_xml_root_path,out_root_path):
    print(file)
    cnt = 0
    while cnt < need_aug_num:

        pic_path = os.path.join(parent, file)
        xml_path = os.path.join(source_xml_root_path, file[:-4] + '.xml')

        coords = parse_xml(xml_path)  
        # coords = [coord[:5] for coord in coords]


        img = cv2.imread(pic_path)
        # show_pic(img, coords)    

        auged_img, auged_bboxes = dataAug.dataAugment(img, coords)

        cnt += 1

        #print(i + 1, '/', len(files), file, cnt)

        aug_img_name = 'aug_' + str(cnt) + '_' + file

        # auged_bboxes=[box.append('table') for box in auged_bboxes]
        # for box in auged_bboxes:
        #    box.append('table')

        generate_xml(aug_img_name, auged_img, auged_bboxes, out_root_path)

        out_img_path = os.path.join(out_root_path, 'images')
        if not os.path.exists(out_img_path):
            os.makedirs(out_img_path)
        cv2.imwrite(os.path.join(out_img_path, aug_img_name), auged_img)
        # show_pic(auged_img, auged_bboxes)  


if __name__ == '__main__':

    need_aug_num = 10 
    out_root_path = '../datasets' 

    source_pic_root_path = './data/images'
    source_xml_root_path = './data/Annotations'
    dataAug = DataAugmentForObjectDetection()
    if not os.path.exists(out_root_path):
        os.makedirs(out_root_path)
    s=time.time()

    process=mp.cpu_count()
    p = Pool(process)
    for parent, _, files in os.walk(source_pic_root_path):
        for i,file in enumerate(files):
            p.apply_async(gen, args=(parent,file,dataAug,need_aug_num,source_xml_root_path,out_root_path))
    p.close()
    p.join()

    e = time.time()
    print('耗时：',e-s)


