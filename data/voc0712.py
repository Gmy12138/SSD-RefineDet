"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from data.config import HOME
from utils.augmentations import SSDAugmentation
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import glob
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
      'crazing',   'inclusion',  'patches',
      'pitted_surface',  'rolled-in_scale',  'scratches')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "data/NEU-DET/")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712',phase='train'):
        self.root = root
        # self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        # self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        # self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.phase=phase
        paths = osp.join(self.root, self.phase, 'label')
        # print(annopath)
        # imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = sorted(glob.glob("%s/*.*" % paths))
        self._imgpath = [path.replace("label", "image").replace(".xml", ".jpg") for path in self._annopath]
        self.ids = list()
        rootpath = osp.join(self.root, 'valid', 'image')
        a = sorted(glob.glob("%s/*.*" % rootpath))
        imagenames = [path.split('/')[-1].split('.')[0].strip() for path in a]
        for line in imagenames:
            self.ids.append((rootpath, line.strip()))

        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self._annopath)

    def pull_item(self, index):
        # img_id = self.ids[index]

        target = ET.parse(self._annopath[index]).getroot()
        img = cv2.imread(self._imgpath[index])
        # print(self._annopath[index])
        if len(img.shape) == 2:
            # np.expand_dims: 用于扩展数组的形状
            img = np.expand_dims(img, -1)
            # np.concatenate：用于数组连接
            img = np.concatenate([img, img, img], axis=-1)

        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

if __name__ == '__main__':


   def detection_collate(batch):

       targets = []
       imgs = []
       for sample in batch:
           # print(sample[1])
           imgs.append(sample[0])
           targets.append(torch.FloatTensor(sample[1]))
       return torch.stack(imgs, 0), targets
   #
   #
   dataset = VOCDetection(root=VOC_ROOT,transform=SSDAugmentation(300,(104, 117, 123)))
   data_loader = data.DataLoader(dataset, 1,
                                 num_workers=0,
                                 shuffle=True, collate_fn=detection_collate,
                                 pin_memory=True)
   batch_iterator = iter(data_loader)
   images, targets = next(batch_iterator)
   targets = [ann for ann in targets]
   print(images,targets)

