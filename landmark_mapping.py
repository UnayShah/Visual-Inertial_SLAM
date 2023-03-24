
import numpy as np
from tqdm import tqdm
from pr3_utils import *
from utils import *

def landmark_mapping(features, imu_T_cam, hidden_features, empty_features, K, b, Ks, T, t):
    P = np.hstack([np.eye(3), np.zeros([3, 1])])
    V = 3
    sigma = np.eye(3*features.shape[1]) # 3M x 3M
    mu = np.zeros([4, features.shape[1]])
    prev_features = np.where(np.all(features[..., 0]!=hidden_features, 0))[0]
    mu[:, prev_features] = pixel_to_world(features[:, prev_features, 0], K, b, imu_T_cam, T[..., 0])

    for timestamp in tqdm(range(1, t.shape[1])):
        visible_features_positions = np.where(np.all(features[..., timestamp]!=hidden_features, 0))[0]
        seen_features_positions = np.where(np.all(mu!=empty_features, 0))[0]
        seen_features_again_positions = np.intersect1d(visible_features_positions, seen_features_positions)
        new_features_positions = np.setdiff1d(visible_features_positions, seen_features_positions)

        seen_features_again = features[:, seen_features_again_positions, timestamp]
        seen_features_again_flatten_shape = seen_features_again.flatten('F').shape[0] # 4Nt
        seen_features_again_positions_shape = seen_features_again_positions.shape[0] # Nt
        new_features = features[:, new_features_positions, timestamp]
        
        mu[:, new_features_positions] = pixel_to_world(new_features, K, b, imu_T_cam, T[..., timestamp])

        z_tilda = predict_z(imu_T_cam, mu[:, seen_features_again_positions], Ks, T[..., timestamp])

        H = np.zeros([seen_features_again_flatten_shape, 3*seen_features_again_positions_shape])  # 4Nt x 3M
        h1 = projectionJacobian(np.linalg.inv(imu_T_cam).dot(np.linalg.inv(T[..., timestamp]).dot(mu[:, seen_features_again_positions])).T)
        h2 = np.linalg.inv(imu_T_cam).dot(np.linalg.inv(T[..., timestamp]).dot(P.T))
        h = np.einsum('aij,jk->aik',h1,h2) 
        h = np.einsum('ij,ajl->ail',Ks,h)

        for i in range(seen_features_again_positions.shape[0]):
            H[i*4: i*4+4, i*3:i*3+3] = h[i, :, :]

        IV = V*np.eye(seen_features_again_flatten_shape)
        new_sigma = np.zeros([3*seen_features_again_positions_shape, 3*seen_features_again_positions_shape]) #  3Nt x 3Nt

        for i in range(seen_features_again_positions_shape):
            new_sigma[3*i:3*i+3, 3*i:3*i+3] = sigma[3*seen_features_again_positions[i]:3*seen_features_again_positions[i]+3, 3*seen_features_again_positions[i]:3*seen_features_again_positions[i]+3]

        k_1 = new_sigma.dot(H.T)
        k_2 = H.dot(new_sigma.dot(H.T)) + IV
        kalman_gain = k_1.dot(np.linalg.inv(k_2))

        del_z = seen_features_again - z_tilda
        del_z = del_z.flatten('F') #  4Nt
        K_del_z = kalman_gain.dot(del_z)
        new_mu = mu[:3].flatten('F') #  3M

        for i in range(seen_features_again_positions.shape[0]):
            new_mu[3*seen_features_again_positions[i]:3*seen_features_again_positions[i]+3] = \
            new_mu[3*seen_features_again_positions[i]:3*seen_features_again_positions[i]+3] + K_del_z[3*i:3*i+3]

        new_sigma = (np.eye(3*seen_features_again_positions_shape) - kalman_gain.dot(H)).dot(new_sigma)

        for i in range(seen_features_again_positions_shape):
            sigma[3*seen_features_again_positions[i]:3*seen_features_again_positions[i]+3, 3*seen_features_again_positions[i]:3*seen_features_again_positions[i]+3] = \
            new_sigma[3*i:3*i+3, 3*i:3*i+3]
        
        mu = np.vstack([new_mu.reshape([3, features.shape[1]], order='F'), np.ones([1, features.shape[1]])])
    return T, mu