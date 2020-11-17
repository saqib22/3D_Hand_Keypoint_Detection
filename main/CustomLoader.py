import cv2
import os
import numpy as np
from config import cfg
import torch
import torchvision.transforms as transforms
import os.path as osp

from utils.preprocessing import load_skeleton
from utils.vis import vis_keypoints, vis_3d_keypoints
from utils.transforms import world2cam, cam2pixel, pixel2cam
from utils.preprocessing import load_skeleton, trans_point2d, augmentation
from utils.preprocessing import load_img as load_image

class CustomLoader():

    def __init__(self, transform, test_dir, batch_size=32):
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.filenames = os.listdir(self.test_dir)
        self.batch_index = 0
        self.transform = transform
        self.skeleton = load_skeleton(osp.join('/home/ubuntu/3d-testing/InterHand2.6M/data/InterHand2.6M/annotations', 'skeleton.txt'), 21*2)
        self.bboxs = open("bbs.txt", "r")
        self.files = open("my_hand_files.txt", "r")
        self.index = -1

        self.fs = self.files.readlines()
        self.bs = self.bboxs.readlines()

    def load_img(self, p, order='RGB'):
        #img = cv2.imread(self.test_dir + "/" + path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        p = osp.join('/home/ubuntu/3d-testing/InterHand2.6M/main/', p).strip()

        img = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
        
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)

        if order=='RGB':
            img = img[:,:,::-1].copy()
        
        img = img.astype(np.float32)
        return img.reshape(3, cfg.input_img_shape[0], cfg.input_img_shape[1])

    def get_batch_from_txt_files(self):
        batch = np.empty(shape=(self.batch_size, 3, 256, 256))
        bboxs = []

        fs = self.files.readlines()
        bs = self.bboxs.readlines()
        
        #for i, f in enumerate(fs):
        batch[0,:,:,:] = self.load_img(fs[0])
        batch = torch.from_numpy(batch.astype(np.float32))/255.

        box = bs[0].split(",")
        bbox = []
        for b in box:
            bbox.append(float(b.replace("\n", "")))

        # image = cv2.imread(osp.join('/home/ubuntu/3d-testing/InterHand2.6M/main/', f).strip())
        # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[0]) + int(bbox[3])), (255,0,0), 3)
        # print(cv2.imwrite("bbox_testing/" + str(i)+".jpg", image))

        bboxs.append(bbox)

        return batch, osp.join('/home/ubuntu/3d-testing/InterHand2.6M/main/', fs[0]).strip(), bbox

    def get_batch_from_txt_files_(self):        
        self.index += 1
        #for i, f in enumerate(fs):
        img = load_image(osp.join('/home/ubuntu/3d-testing/InterHand2.6M/main/', self.fs[self.index]).strip())

        box = self.bs[self.index].split(",")
        bbox = []
        for b in box:
            bbox.append(float(b.replace("\n", "")))

        img, inv_trans = augmentation(img, np.array(bbox), None, None, None, 'test', None)
        img = self.transform(img.astype(np.float32))/255.

        # image = cv2.imread(osp.join('/home/ubuntu/3d-testing/InterHand2.6M/main/', self.fs[self.index]).strip())
        # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[0]) + int(bbox[3])), (255,0,0), 3)
        # print(cv2.imwrite("bbox_testing/" + str(self.index)+".jpg", image))

        return img, osp.join('/home/ubuntu/3d-testing/InterHand2.6M/main/', self.fs[self.index]).strip(), inv_trans

    def get_batch(self, filename):
        batch = np.empty(shape=(self.batch_size, 3, 256, 256))

        #for i, img_name in enumerate(self.filenames):
            #batch[i,:,:,:] = self.load_img(img_name)/255.
        batch[0,:,:,:] = self.load_img(filename)

        batch = torch.from_numpy(batch.astype(np.float32))/255.
        return batch

    def visualize(self, preds, filename, img_path, inv_trans):
        joint_valid = [1.0]*21 + [1.0]*21 #change 1.0 to 0 if that hand is not resent right hand is comes first in output
        focal = [1500, 1500] # x-axis, y-axis
        princpt = [256/2, 256/2] 
        root_joint_idx = {'right': 20, 'left': 41}

        preds_joint_coord, preds_rel_root_depth, preds_hand_type = preds['joint_coord'], preds['rel_root_depth'], preds['hand_type']

        print(preds_hand_type)


        # img = load_image(img_path)
        
        # inv_trans = augmentation(img, 
        #                         np.array(bbox), 
        #                         preds_joint_coord, 
        #                         joint_valid, 
        #                         preds_hand_type, 
        #                         'test', 
        #                         None)
        
        pred_joint_coord_img = preds_joint_coord[0].copy()
        pred_joint_coord_img[:,0] = pred_joint_coord_img[:,0]/cfg.output_hm_shape[2]*cfg.input_img_shape[1]
        pred_joint_coord_img[:,1] = pred_joint_coord_img[:,1]/cfg.output_hm_shape[1]*cfg.input_img_shape[0]
        for j in range(21*2):
            pred_joint_coord_img[j,:2] = trans_point2d(pred_joint_coord_img[j,:2], inv_trans)
        pred_joint_coord_img[:,2] = (pred_joint_coord_img[:,2]/cfg.output_hm_shape[0] * 2 - 1) * (cfg.bbox_3d_size/2)

        # if preds_hand_type[0][0] == 0.9 and preds_hand_type[0][1] == 0.9:  #change threshold to execute this parth if both handa are present
        #     pred_rel_root_depth = (preds_rel_root_depth[0]/cfg.output_root_hm_shape * 2 - 1) * (cfg.bbox_3d_size_root/2)

        #     pred_left_root_img = pred_joint_coord_img[root_joint_idx['left']].copy()
        #     pred_left_root_img[2] +=  pred_rel_root_depth
        #     pred_left_root_cam = pixel2cam(pred_left_root_img[None,:], focal, princpt)[0]

        #     pred_right_root_img = pred_joint_coord_img[root_joint_idx['right']].copy()
        #     pred_right_root_cam = pixel2cam(pred_right_root_img[None,:], focal, princpt)[0]
            
        #     pred_rel_root = pred_left_root_cam - pred_right_root_cam

        # pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, focal, princpt)
        
        # joint_type = {'right': np.arange(0,21), 'left': np.arange(21,21*2)}
        # for h in ('right', 'left'):
        #     pred_joint_coord_cam[joint_type[h]] = pred_joint_coord_cam[joint_type[h]] - pred_joint_coord_cam[root_joint_idx[h],None,:]

        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(pred_joint_coord_cam.shape)
        
        print(img_path)
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        _img = cvimg[:,:,::-1].transpose(2,0,1)
        vis_kps = pred_joint_coord_img.copy()
        vis_valid = joint_valid.copy()
        
        filename = img_path.replace(".jpg", "2D.jpg").replace("main/custom_data", "custom_output")
        vis_keypoints(_img, vis_kps, joint_valid, self.skeleton, filename)
        filename = img_path.replace(".jpg", "3D.jpg").replace("main/custom_data", "custom_output")
        vis_3d_keypoints(pred_joint_coord_img, joint_valid, self.skeleton, filename)

        print("Finished Processing Image!!!!" + "\n")


if __name__ == '__main__':
    loader = CustomLoader(transforms.ToTensor(), '../custom_data')
    inputs = loader.get_batch()

    print(inputs.shape)