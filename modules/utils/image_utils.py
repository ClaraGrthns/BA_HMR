import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import torch
from matplotlib import pyplot as plt

from .data_utils_3dpw import get_relevant_keypoints
#from .render import Renderer
from .geometry import rotation_matrix_to_angle_axis
import cv2


# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
class SquarePad_tensor:
    def __call__(self, img):
        h, w = img.shape[1:]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp,  hp, vp, vp)
        return torch.nn.ZeroPad2d(padding)(img)
    
def crop_box(img_tensor, pose2d, border_scale=1.3, padding = True):
    h, w = img_tensor.shape[-2:]
    relevant_pose2d = torch.tensor(get_relevant_keypoints(pose2d))

    if relevant_pose2d.nelement() == 0:
        x_min, x_max, y_min, y_max =  0, w, 0, h
    else:
        # get min and max coordinates for bounding box
        box_min = (relevant_pose2d.min(dim=0)[0])
        box_max = (relevant_pose2d.max(dim=0)[0])
        # scale box
        box_wh = box_max - box_min
        delta = box_wh.max() * ((border_scale-1) / 2)
        box_min = (box_min - delta).to(torch.int64)
        box_min = torch.max(box_min, torch.tensor([0,0])) #limited to the image borders
        box_max = (box_max + delta).to(torch.int64)
        
        center_xy = torch.tensor((box_max + box_min).detach().numpy().copy())//2
        
        x_min, y_min = box_min
        x_max, y_max = box_max  

    if padding:
        w_new = abs(x_max-x_min)
        h_new = abs(y_max-y_min)
        max_wh = np.max([w_new,h_new])
        hp = int((max_wh - w_new) / 2)
        vp = int((max_wh - h_new) / 2)
        #distance from bbox to image border (smaller one)
        limit_pad = abs(torch.min(abs(center_xy - torch.tensor([w,h])), center_xy) - torch.tensor([w_new//2, h_new//2]))
        #padding s.t. it doesn't go beyond the image borders
        hp, vp = torch.min(limit_pad, torch.tensor([hp, vp]))
        x_min, y_min = torch.max(torch.tensor([x_min-hp,y_min-vp ]), torch.tensor([0,0]))
        x_max, y_max = x_max+hp, y_max+vp

    return img_tensor[:, y_min:y_max, x_min:x_max], (x_min, x_max, y_min, y_max)


def to_tensor(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim == 2:
         return torch.from_numpy(img.astype(np.float32))/255.
    else:
        return torch.from_numpy(img.astype(np.float32)).permute(2,0,1)/255.



def transform(img, img_size=224):
    trans = transforms.Compose([  
                        SquarePad_tensor(),         
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    return trans(img)

def transform_visualize(img, img_size=224):
    trans = transforms.Compose([  
                        SquarePad_tensor(),
                        transforms.Resize(img_size),
                        transforms.CenterCrop(img_size),
                        ])
    return trans(img)

def plot_tensor(img_tensor):
    plt.figure(figsize = (9,16))
    plt.axis('off')
    plt.imshow(img_tensor.numpy().transpose(1,2,0),)
    plt.show()
    
    
# INPUT: betas: torch.Tensor([1, 10]), poses: torch.Tensor([1,72]), trans: torch.Tensor([1,3])
# cam_pose: torch.Tensor([4,4]), cam_intr: torch.Tensor([3, 3])
def visualize_mesh(img, cam_intr, smpl, beta=None, pose=None, trans=None, vertices=None):
    cam_intr = cam_intr.detach().numpy()
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1,2,0)

    ## get smpl faces and vertices ##
    faces = smpl.faces.cpu().numpy()
    if vertices is None:
        vertices = smpl(pose=pose, beta=beta) + trans
    vertices = vertices.detach().numpy()
    ## camera: rotation matrix, t, f and center
        #cam_rot = rotation_matrix_to_angle_axis(cam_pose[None, :3, :3]).detach().numpy().ravel()
    cam_rot = rotation_matrix_to_angle_axis(torch.eye(3)[None]).detach().numpy().ravel() 
    cam_t = np.zeros(3) #cam_pose[0:3,3]
    cam_f = np.array([cam_intr[0,0],cam_intr[1,1]])
    cam_center = cam_intr[0:2,2]
    ## Visualize Mesh 
    renderer = Renderer(faces=faces)
    color= 'pink'
    focal_length = 1000
    rend_img = renderer.render(vertices,
                               cam_t= cam_t,
                               cam_rot= cam_rot,
                               cam_center= cam_center,
                               cam_f = cam_f,
                               img= img, 
                               use_bg = True,
                               focal_length = focal_length,
                               body_color = color)
    #return torch.Tensor
    rend_img = rend_img.transpose(2,0,1)
    rend_img = torch.from_numpy(rend_img.copy())
    return rend_img

def lcc(mask):
    num_labels, labels_im = cv2.connectedComponents(np.byte(mask), connectivity=4)
    labels = np.eye(num_labels)[labels_im][:,:,1:]
    arg = labels.sum(axis=0).sum(axis=0).argmax()
    return labels[:,:,arg]