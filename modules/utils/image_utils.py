import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import torch
from matplotlib import pyplot as plt

from .data_utils import get_relevant_keypoints
from .render import Renderer
from .geometry import rotation_matrix_to_angle_axis
from ..smpl_model.smpl import SMPL, SMPL_MODEL_DIR, get_smpl_faces


# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/4
class SquarePad_tensor:
    def __call__(self, img):
        h, w = img.shape[1:]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp,  hp, vp, vp)
        return torch.nn.ZeroPad2d(padding)(img)
    
def crop_box(img_tensor, pose2d, border_scale=1.3):
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
        delta = box_wh.max() * ((1.3-1) / 2)
        box_min = (box_min - delta).to(torch.int64)
        box_max = (box_max + delta).to(torch.int64)
        
        x_min, y_min = box_min
        x_max, y_max = box_max

    # Ensure that box does not exceed image width or height.
    x_min, y_min = torch.max(torch.zeros(2, dtype=torch.int64), torch.tensor([x_min, y_min], dtype=torch.int64))
    x_max, y_max = torch.min(torch.tensor(img_tensor.shape[-2:][::-1], dtype=torch.int64), torch.tensor([x_max, y_max], dtype=torch.int64))
    crop = img_tensor[:, y_min:y_max, x_min:x_max]
    
    return crop, (x_min, x_max, y_min, y_max)

def to_tensor(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    return torch.from_numpy(img.astype(np.float32)).permute(2,0,1)/255.



def transform(img, size=224):
    trans = transforms.Compose([  
                        SquarePad_tensor(),         
                        transforms.Resize(size),
                        transforms.CenterCrop(size),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    return trans(img)

def transform_visualize(img, size=224):
    trans = transforms.Compose([  
                        SquarePad_tensor(),     
                        transforms.Resize(size),
                        transforms.CenterCrop(size),
                        ])
    return trans(img)

def plot_tensor(img_tensor):
    plt.figure(figsize = (9,16))
    plt.axis('off')
    plt.imshow(img_tensor.numpy().transpose(1,2,0),)
    plt.show()
    
    
# INPUT: betas: torch.Tensor([1, 10]), poses: torch.Tensor([1,72]), trans: torch.Tensor([1,3])
# cam_pose: torch.Tensor([1,4,4]), cam_intr: torch.Tensor([3, 3])
def visualize_mesh(img, beta, pose, cam_pose, cam_intr, trans = None):
    cam_intr = cam_intr.detach().numpy()

    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1,2,0)
   

 ## get smpl faces and vertices ##

    #SMPLX Model: 
    smpl = SMPL(SMPL_MODEL_DIR)
    body_pose = pose[:,3:]
    global_orient = pose[:,:3]
    out = smpl(beta, body_pose, global_orient, trans) ## SMPLX with translation
    vertices = out.vertices
    vertices = vertices[0].detach().numpy()
    faces = get_smpl_faces()
    
    # _SMPL from METRO:
    #smpl = SMPL()
    #vertices = smpl(pose = pose, beta = beta[:10]) + trans
    #faces = smpl.faces.cpu().numpy()
    

    # camera: rotation matrix, t, f and center
    cam_rot = rotation_matrix_to_angle_axis(cam_pose[None, :3, :3]).detach().numpy().ravel() 
    cam_t = cam_pose[0:3,3]
    cam_f = np.array([cam_intr[0,0],cam_intr[1,1]])
    cam_center = cam_intr[0:2,2]



    # Visualize Mesh 
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

