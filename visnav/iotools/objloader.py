# adapted from http://www.pygame.org/wiki/OBJFileLoader
import os

import cv2
import numpy as np
from algo import tools


def MTL(filename):
   contents = {}
   mtl = None
   for line in open(filename, "r"):
       if line.startswith('#'): continue
       values = line.split()
       if not values: continue
       if values[0] == 'newmtl':
           mtl = contents[values[1]] = {}
       elif mtl is None:
           raise ValueError("mtl file doesn't start with newmtl stmt")
       elif values[0] == 'map_Kd':
           mtl[values[0]] = values[1]
           ## load the texture referred to by this declaration
           # surf = pygame.image.load(mtl['map_Kd'])
           # image = pygame.image.tostring(surf, 'RGBA', 1)
           # ix, iy = surf.get_rect().size
           # texid = mtl['texture_Kd'] = glGenTextures(1)
           # glBindTexture(GL_TEXTURE_2D, texid)
           # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
           #     GL_LINEAR)
           # glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
           #     GL_LINEAR)
           # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA,
           #     GL_UNSIGNED_BYTE, image)
       else:
           mtl[values[0]] = list(map(float, values[1:]))
   return contents


class ShapeModel:
    def __init__(self, fname=None, data=None):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.texfile = None
        
        if fname is not None:
            self.from_file(fname)
        elif data is not None:
            self.from_dict(data)
    
    def from_file(self, fname, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.texfile = None
        dir = os.path.abspath(os.path.dirname(fname))
 
        #material = None
        for line in open(fname, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                txc = list(map(float, values[1:3]))
                assert len(txc) == 2, 'wrong length texture coordinates at idx=%d'%len(self.texcoords)
                self.texcoords.append(txc)
            elif values[0] in ('usemtl', 'usemat'):
                pass
#                material = values[1]
            elif values[0] == 'mtllib':
                mtl = MTL(os.path.join(dir, values[1]))
                material = tuple(mtl.values())[0]
                if 'map_Kd' in material:
                    self.texfile = os.path.join(dir, material['map_Kd'])
            elif values[0] == 'f':
                vertices = []
                texcoords = []
#                norm = []
                for v in values[1:]:
                    w = v.split('/')
                    vertices.append(int(w[0])-1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
#                    if len(w) >= 3 and len(w[2]) > 0:
#                        norm.append(int(w[2]))
#                    else:
#                        norm.append(0)
                    #self.faces.append((face, norms, texcoords, material))
                if len(vertices)==3:
                    assert len(texcoords) == 0 or len(vertices) == len(texcoords), 'Some tex coords missing!'
                    self.faces.append((vertices, self._calc_norm(vertices), texcoords))
#                    self.triangles.append(tuple(face))
                else:
                    raise Exception('Not a triangle!')

    def from_dict(self, data):
        self.faces = data['faces']
        if len(self.faces[0]) < 3:
            #  backward compatibility
            self.faces = [(f[0], f[1], []) for f in self.faces]

        self.vertices = data['vertices']
        self.normals = data.get('normals', [])
        self.texcoords = data.get('texcoords', [])
        self.texfile = data.get('texfile', None)
        if len(self.normals) == 0:
            self.recalc_norms()
        
    def as_dict(self):
        return {'faces':self.faces, 'vertices':self.vertices, 'normals':self.normals,
                'texcoords':self.texcoords, 'texfile':self.texfile}
        
    def _calc_norm(self, face):
        n = tools.surf_normal(self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]])
        self.normals.append(n)
        return len(self.normals)-1
    
    def recalc_norms(self):
        self.normals.clear()
        self.faces = [(face, self._calc_norm(face), tx) for face, n, tx in self.faces]

    def export_smooth_faces(self):
        """
        compatible output for a moderngl ext obj
        """
        norms = np.zeros((len(self.vertices), 3))
        for f, n, t in self.faces:
            norms[f[0]] += self.normals[n]
            norms[f[1]] += self.normals[n]
            norms[f[2]] += self.normals[n]
        norms = norms / np.linalg.norm(norms, axis=1).reshape((-1, 1))
        faces = [(vx + 1, (txs or None) and txs[i], vx + 1)
                    for vxs, nrm, txs in self.faces
                        for i, vx in enumerate(vxs)]
        return self.vertices, [(tx, ty, 0) for tx, ty in self.texcoords], norms, faces

    def export_angular_faces(self):
        """
        compatible output for a moderngl ext obj
        """
        faces = [(vx + 1, (txs or None) and txs[i], nrm + 1)
                    for vxs, nrm, txs in self.faces
                        for i, vx in enumerate(vxs)]
        return self.vertices, [(tx, ty, 0) for tx, ty in self.texcoords], self.normals, faces

    def load_texture(self, normalize=True):
        if self.texfile is None:
            return None
        tex = cv2.imread(self.texfile, cv2.IMREAD_GRAYSCALE).astype('f4')
        if normalize:
            tex /= np.max(tex)  # normalize so that max relative albedo is 1
        return tex
