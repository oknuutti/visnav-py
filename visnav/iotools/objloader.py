# adapted from http://www.pygame.org/wiki/OBJFileLoader
import os

import cv2
import numpy as np

from visnav.algo import tools


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
        self.vertices = None
        self.normals = None
        self.texcoords = None
        self.faces = None
        self.texfile = None
        self._tex = None
        
        if fname is not None:
            self.from_file(fname)
        elif data is not None:
            self.from_dict(data)

    def from_file(self, fname, swapyz=False):
        """Loads a Wavefront OBJ file. """
        vertices = []
        normals = []
        texcoords = []
        faces = []
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
                vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                normals.append(v)
            elif values[0] == 'vt':
                txc = list(map(float, values[1:3]))
                assert len(txc) == 2, 'wrong length texture coordinates'
                texcoords.append(txc)
            elif values[0] in ('usemtl', 'usemat'):
                pass
#                material = values[1]
            elif values[0] == 'mtllib':
                mtl = MTL(os.path.join(dir, values[1]))
                material = tuple(mtl.values())[0]
                if 'map_Kd' in material:
                    self.texfile = os.path.join(dir, material['map_Kd'])
            elif values[0] == 'f':
                fvert = []
                ftext = []
#                norm = []
                for v in values[1:]:
                    w = v.split('/')
                    fvert.append(int(w[0])-1)
                    if len(w) >= 2 and len(w[1]) > 0:
                        ftext.append(int(w[1])-1)
#                    if len(w) >= 3 and len(w[2]) > 0:
#                        norm.append(int(w[2]))
#                    else:
#                        norm.append(0)
                    #self.faces.append((face, norms, texcoords, material))
                if len(fvert) == 3:
                    assert len(ftext) == 0 or len(fvert) == len(ftext), 'Some tex coords missing!'
                    # normals are calculated for each face => same indices as faces
                    faces.append((fvert, len(faces), ftext))   # v idx, n idx, t idx
#                    self.triangles.append(tuple(face))
                else:
                    assert False, 'Not a triangle!'

        nf = len(faces)
        faces = ShapeModel._face_massage(faces)
        self.faces = np.array(faces, dtype=np.uint32)
        assert self.faces.shape == (nf*3, 3),\
                                    'wrong shape "faces" array %s should be (nf*3, 3)' % (self.faces.shape,)

        self.vertices = np.array(vertices, dtype=np.float32)
        assert self.vertices.shape[1:] == (3,),\
                                    'wrong shape "vertices" array %s should be (-1, 3)' % (self.vertices.shape,)

        self.texcoords = np.array(texcoords, dtype=np.float32)
        assert self.texcoords.shape[1:] == (2,),\
                                    'wrong shape "texcoords" array %s should be (-1, 2)' % (self.texcoords.shape,)

        self.recalc_norms()
        assert len(self.normals) == 0 or self.normals.shape[1:] == (3,), \
                                    'wrong shape "normals" array %s should be (-1, 3)' % (self.normals.shape,)

    @staticmethod
    def _face_massage(faces):
        # (n faces, v&n&t, 3 x vertices) => (nf*3, v&n&t)
        faces = [(vx, i, (txs or 0) and txs[j])
                    for i, (vxs, nrm, txs) in enumerate(faces)
                        for j, vx in enumerate(vxs)]
        return faces

    def from_dict(self, data):
        self.faces = data['faces']
        self.vertices = data['vertices']
        self.normals = data.get('normals', [])
        self.texcoords = data.get('texcoords', [])
        self.texfile = data.get('texfile', None)
        self.tex = data.get('tex', None)

        # backwards compatibility
        if not isinstance(self.faces, np.ndarray):
            nf = len(self.faces)
            self.faces = np.array(ShapeModel._face_massage(self.faces), dtype=np.uint32)
            self.faces[:, 2] -= 1   # tx idxs started from 1
            assert self.faces.shape == (nf * 3, 3),\
                                'wrong shape "faces" array %s should be (nf*3, 3)' % (self.faces.shape,)

            self.vertices = np.array(self.vertices, dtype=np.float32)
            assert self.vertices.shape[1:] == (3,),\
                                'wrong shape "vertices" array %s should be (-1, 3)' % (self.vertices.shape,)

            self.texcoords = np.array(self.texcoords, dtype=np.float32)
            assert self.texcoords.shape[1:] == (2,),\
                                'wrong shape "texcoords" array %s should be (-1, 2)' % (self.texcoords.shape,)

            self.normals = np.array(self.normals, dtype=np.float32)
            assert len(self.normals) == 0 or self.normals.shape[1:] == (3,), \
                                'wrong shape "normals" array %s should be (-1, 3)' % (self.normals.shape,)

        if len(self.normals) == 0:
            self.recalc_norms()

        self.faces = self.faces.astype(np.uint32)
        self.vertices = self.vertices.astype(np.float32)
        self.texcoords = self.texcoords.astype(np.float32)
        self.normals = self.normals.astype(np.float32)

    def as_dict(self):
        return {'faces': self.faces, 'vertices': self.vertices, 'normals': self.normals,
                'texcoords': self.texcoords, 'texfile': self.texfile, 'tex': self.tex}

    def recalc_norms(self):
        """
        Recalculate normals so that each vertex of a face has the normal of the face. For optional smooth normals,
        would need to average normals across the faces each unique vertex belongs to and set faces[:, 1] = faces[:, 0]
        """
        # reshape faces to be (nf, 3v, v&n&t)
        f, v = self.faces.reshape((-1, 3, 3)), self.vertices
        v1, v2, v3 = v[f[:, 0, 0]], v[f[:, 1, 0]], v[f[:, 2, 0]]
        n = np.cross(v2 - v1, v3 - v1)
        self.normals = n / np.linalg.norm(n, axis=1).reshape((-1, 1))

    def pack_all(self):
        f, v, n, t = self.faces, self.vertices, self.normals, self.texcoords
        return np.hstack((v[f[:, 0], :], n[f[:, 1], :], t[f[:, 2], :])).astype(np.float32).tobytes()

    def pack_simple(self):
        f, v = self.faces, self.vertices
        return v[f[:, 0], :].astype(np.float32).tobytes()

    def texture_to_vertex_map(self):
        tx2vx = np.ones((len(self.texcoords),), dtype=np.int64) * -1
        for v, n, t in self.faces:
            tx2vx[t] = v
        return tx2vx

    def export_smooth_faces(self):
        """
        compatible output for a moderngl ext obj
        """
        assert False, 'not supported anymore'
        # norms = np.zeros((len(self.vertices), 3))
        # for f, n, t in self.faces:
        #     norms[f[0]] += self.normals[n]
        #     norms[f[1]] += self.normals[n]
        #     norms[f[2]] += self.normals[n]
        # norms = norms / np.linalg.norm(norms, axis=1).reshape((-1, 1))
        # faces = [(vx + 1, (txs or None) and txs[i]+1, vx + 1)
        #             for vxs, nrm, txs in self.faces
        #                 for i, vx in enumerate(vxs)]
        # return self.vertices, [(tx, ty, 0) for tx, ty in self.texcoords], norms, faces

    def export_angular_faces(self):
        """
        compatible output for a moderngl ext obj
        """
        texcoords = np.hstack((self.texcoords, np.zeros((len(self.texcoords), 1), dtype=np.float32)))
        return self.vertices, texcoords, self.normals, self.faces[:, (0, 2, 1)] + 1

    def load_texture(self, normalize=True):
        if self.tex is not None:
            return self.tex
        if self.texfile is None:
            return None
        self._tex = cv2.imread(self.texfile, cv2.IMREAD_GRAYSCALE).astype('f4')
        if normalize:
            self._tex /= np.max(self._tex)  # normalize so that max relative albedo is 1
        return self._tex

    @property
    def tex(self):
        return self._tex

    @tex.setter
    def tex(self, new_tex):
        self._tex = None if new_tex is None else new_tex.astype('f4')
