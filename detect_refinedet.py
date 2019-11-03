import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from PIL import Image
import time
import datetime
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from models.refinedet import build_refinedet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform, BaseTransform
from data import VOC_CLASSES as labels

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=320):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor

        img = cv2.imread(img_path)
        Augmentation = BaseTransform(320,(104, 117, 123))
        img, _, _ = Augmentation(img)
        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2, 0, 1)

        # img = transforms.ToTensor()(Image.open(img_path))
        # # Pad to square resolution
        # img, _ = pad_to_square(img, 0)
        # Resize
        # img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/RefineDet320_591.pth',type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='output/', type=str,help='Dir to save results')
parser.add_argument('--dataset_root', default='data/samples', help='Dataset root directory path')
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=320, help="size of each image dimension")
args = parser.parse_args()


os.makedirs("refinedet_output", exist_ok=True)
net = build_refinedet('test', 320, 7)
net.load_weights(args.trained_model)


dataloader = DataLoader(
        ImageFolder(args.dataset_root, img_size=args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
    )
TIME=0
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    # input_imgs = Variable(input_imgs.type(Tensor))
    prev_time = time.time()
    # print(img_paths[0])
    xx = input_imgs  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)

    current_time = time.time()
    inference_time = current_time - prev_time

    if batch_i != 0:
        TIME += inference_time

    prev_time = current_time
    print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))


    top_k = 10

    # plt.figure(figsize=(10, 10))
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]
    # plt.imshow(img)  # plot the image for matplotlib
    # currentAxis = plt.gca()

    img = Image.open(img_paths[0])
    img = img.resize((320, 320))
    img = np.array(img)
    # print(img.shape)
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(img.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.55:
            score = detections[0, i, j, 0]
            # print(i)
            label_name = labels[i - 1]
            # print(score,label_name)
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            # print(pt)
            color = colors[i-1]
            # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            # j += 1
            # Create a Rectangle patch
            bbox = patches.Rectangle(*coords, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                pt[0],
                pt[1],
                s=label_name,
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
            j+=1
            # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = img_paths[0].split("/")[-1].split(".")[0]
    plt.savefig(f"refinedet_output/{filename}.jpg", bbox_inches="tight",pad_inches=0.0)
    plt.close()
print("FPS: %s" % (1/(TIME/5)))

