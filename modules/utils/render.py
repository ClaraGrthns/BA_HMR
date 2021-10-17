import numpy as np
# https://github.com/microsoft/MeshTransformer/blob/main/metro/utils/renderer.py
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight

def rotateY(points, angle):
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


class Renderer(object):

    def __init__(self, width=800, height=600, near=0.5, far=1000, faces=None):
        self.colors = {'hand': [.9, .9, .9], 'pink': [.9, .7, .7], 'light_blue': [0.65098039, 0.74117647, 0.85882353] }
        self.width = width
        self.height = height
        self.faces = faces
        self.renderer = ColoredRenderer()
        
        
    def render(self, vertices, faces=None, img=None,
               cam_t=np.zeros([3], dtype=np.float32),
               cam_rot=np.zeros([3], dtype=np.float32),
               cam_center=None,
               cam_f = None,
               use_bg=False,
               bg_color=(0.0, 0.0, 0.0),
               body_color=None,
               focal_length=5000,
               disp_text=False,
               gt_keyp=None,
               pred_keyp=None,
               **kwargs):
             
        
        if img is not None:
            height, width = img.shape[:2]
        else:
            height, width = self.height, self.width
        if faces is None:
            faces = self.faces

        if cam_center is None:
            cam_center = np.array([width * 0.5, height * 0.5])

        if body_color is None:
            color = self.colors['light_blue']
        else:
            color = self.colors[body_color]

        if cam_f is None:
            f = np.array([width, width]) / 2.     

        self.renderer.camera = ProjectPoints(rt= cam_rot,
                                             t = cam_t,
                                             f = cam_f,
                                             c = cam_center,
                                             k = np.zeros(5))
        dist = np.abs(self.renderer.camera.t.r[2] - np.mean(vertices, axis=0)[2])
        far = dist+20

        self.renderer.frustum = {'near': 0.5, 'far': far, 'width': width, 'height': height}

        if img is not None:
            if use_bg:
                self.renderer.background_image = img
            else:
                self.renderer.background_image = np.ones_like(img) * np.array(bg_color)

        
              
        self.renderer.set(v=vertices, f=faces,
                          vc=color, bgcolor=np.ones(3))


        num_verts = self.renderer.v.shape[0]
        albedo = self.renderer.vc
        # Construct Back Light (on back right corner)
        yrot = np.radians(120)

        self.renderer.vc = LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts = num_verts,
            #light_pos= np.array([-1000,-1000,-1000]),
            light_pos = rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts= num_verts,
            light_pos= rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        #  Construct Right Light
        self.renderer.vc += LambertianPointLight(
            f=self.renderer.f,
            v=self.renderer.v,
            num_verts=num_verts,
            light_pos=rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))
        

        return self.renderer.r