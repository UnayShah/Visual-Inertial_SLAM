import numpy as np
from pr3_utils import *

def get_Ks(K, b):
    Ks = np.vstack([K[:-1], K[:-1]])
    Ks = np.hstack([Ks, np.zeros([4, 1])])
    Ks[2, -1] = -K[0, 0]*b
    return Ks

def pixel_to_world(features, K, b, imu_T_cam, T):
    ul, vl, d = features[0], features[1], (features[0]-features[2])
    fsu, cu, fsv, cv = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
    x = b*(ul-cu)/d
    y = fsu*b*(vl-cv)/(d*fsv)
    z = fsu*b/d
    coords = np.vstack([x, y, z, np.ones([1, z.shape[0]])])
    return T.dot(imu_T_cam.dot(coords))


def predict_z(imu_T_cam, coords, Ks, T):
    return Ks.dot(projection(np.linalg.inv(imu_T_cam).dot(np.linalg.inv(T).dot(coords)).T).T)

def dot(pos):
  Nt = pos.shape[1]
  z = np.zeros([4, 6, Nt])
  skew = -axangle2skew(pos[:3,:].T)
  for i in range(Nt):
    z[:3, :3, i] = np.eye(3)
    z[:3, 3:6, i] = skew[i]
  return z