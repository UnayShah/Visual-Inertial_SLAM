import numpy as np
from tqdm import tqdm
from pr3_utils import *
from utils import *
from scipy import linalg
import transforms3d

def visual_inertial_slam(features, imu_T_cam, hidden_features, empty_features, K, b, Ks, T, t, linear_velocity, angular_velocity):

    P = np.hstack([np.eye(3), np.zeros([3, 1])])
    V = 4
    world_coordinates = np.zeros([4, features.shape[1]]) # 4 x M
    prev_features = np.where(np.all(features[..., 0]!=hidden_features, 0))[0]
    world_coordinates[:, prev_features] = pixel_to_world(features[:, prev_features, 0], 
                                                        K, b, imu_T_cam, T[..., 0])

    # Trajectory for SLAM, t-th value updated at t-th timestamp
    SLAM_T = np.zeros([4, 4, t.shape[1]])
    SLAM_T[..., 0] = np.eye(4)

    SLAM_sigma_T = np.zeros([6, 6, t.shape[1]])
    SLAM_sigma = np.eye(3*features.shape[1] + 6)
    # SLAM_sigma = np.random.random([3*features.shape[1] + 6, 3*features.shape[1] + 6])
    SLAM_robot_init = np.diag([0.001, 0.001, 0.001, 0.00001, 0.00001, 0.00001])
    SLAM_sigma[3*features.shape[1]:, 3*features.shape[1]:] = SLAM_robot_init
    W = np.diag([0.1,0.01,0.01,0.00001,0.00001,0.0001])
    M = features.shape[1]

    twist = np.vstack([linear_velocity, angular_velocity])
    sigma = np.zeros([6, 6, twist.shape[1]])

    for timestamp in tqdm(range(1, t.shape[1])):
        # Prediction of pose for current timestamp using motion model
        pose = twist2pose((t[0, timestamp]-t[0, timestamp-1])*
                axangle2twist(twist[:, timestamp]))
        SLAM_T[..., timestamp] = np.matmul(SLAM_T[..., timestamp-1], pose)
        A = linalg.expm(-(t[0, timestamp]-t[0, timestamp-1])*axangle2adtwist(twist[:,timestamp]))
        sigma[:,:,timestamp] = np.matmul(A,np.matmul(sigma[:,:,timestamp-1],A.T)) + W

        robot_pos_pred = SLAM_T[:3, -1, timestamp]
        robot_ori_pred = transforms3d.euler.mat2euler(SLAM_T[:3, :3, timestamp], 'sxyz')

        visible_features_positions = np.where(np.all(features[..., timestamp]!=hidden_features, 0))[0]
        seen_features_positions = np.where(np.all(world_coordinates!=empty_features, 0))[0]
        seen_features_again_positions = np.intersect1d(visible_features_positions, seen_features_positions)
        new_features_positions = np.setdiff1d(visible_features_positions, seen_features_positions)
        
        seen_features_again = features[:, seen_features_again_positions, timestamp]
        seen_features_again_flatten_shape = seen_features_again.flatten('F').shape[0] # 4Nt
        seen_features_again_positions_shape = seen_features_again_positions.shape[0] # Nt
        new_features = features[:, new_features_positions, timestamp]
        
        # Finding world coordinates for given pixels
        world_coordinates[:, new_features_positions] = pixel_to_world(new_features, 
                                                                    K, b, imu_T_cam, SLAM_T[..., timestamp])
        # Flatten the visible landmarks
        visible_landmarks = world_coordinates[:3].flatten('F')

        # Update step
        # Calculating H
        ph = np.linalg.inv(imu_T_cam).dot(np.linalg.inv(SLAM_T[..., timestamp])
                                        .dot(world_coordinates[:, seen_features_again_positions]))
        projJacobian = projectionJacobian(ph.T)
        h = np.einsum('aij,jk->aik', projJacobian, 
                    np.linalg.inv(imu_T_cam).dot(np.linalg.inv(SLAM_T[..., timestamp]).dot(P.T)))
        h = np.einsum('ij,ajl->ail', Ks, h)
        
        H = np.zeros([seen_features_again_flatten_shape, visible_landmarks.shape[0]])  # 4Nt x 3M
        for i in range(seen_features_again_positions.shape[0]):
            H[i*4: i*4+4, i*3:i*3+3] = h[i, :, :]
        
        h1 = np.linalg.inv(SLAM_T[..., timestamp]).dot(world_coordinates[:, seen_features_again_positions])
        h2 = np.einsum('ij,jkl->ikl', np.linalg.inv(imu_T_cam), dot(h1)) # 4 x 6 x NT
        h5 = np.einsum('ijk,jli->jli', projJacobian, h2)  # 4 x 6 x NT
        h5 = np.einsum('ij,jlk->ilk', -Ks, h5)  # 4 x 6 x NT
        H_jacobian = np.reshape(h5, [h5.shape[0]*h5.shape[2], 6]) # 4NT x 6

        SLAM_sigma[3*M: 3*M+6, 3*M:3*M+6] = sigma[:,:,timestamp] 
        SLAM_sigma[:3*M,3*M:3*M+6]=np.matmul(SLAM_sigma[:3*M,3*M:3*M+6],A.T)
        SLAM_sigma[3*M:3*M+6,:3*M]=SLAM_sigma[:3*M,3*M:3*M+6].T
        
        SLAM_combined_jacobian = np.hstack([H, H_jacobian])
        # Calculating Kalman Gain
        IV = V*np.eye(seen_features_again_flatten_shape)
        k1 = IV + SLAM_combined_jacobian.dot(SLAM_sigma.dot(SLAM_combined_jacobian.T))
        kalman_gain = SLAM_sigma.dot(SLAM_combined_jacobian.T.dot(np.linalg.inv(k1)))
        z_pred = predict_z(imu_T_cam, world_coordinates[:, seen_features_again_positions], 
                        Ks, SLAM_T[..., timestamp])
        del_z = (seen_features_again - z_pred).flatten('F')
        K_del_z = kalman_gain.dot(del_z)


        visible_landmarks[np.concatenate([3*seen_features_again_positions,
                                        3*seen_features_again_positions+1,
                                        3*seen_features_again_positions+2])] += K_del_z[
            np.concatenate([3*seen_features_again_positions,3*seen_features_again_positions+1,
                                3*seen_features_again_positions+2])] # 3M X1


        # visible_landmarks += K_del_z[:3*features.shape[1]]
        new_mu = np.reshape(visible_landmarks, [3, features.shape[1]])
        world_coordinates = np.vstack([new_mu, np.ones([1, features.shape[1]])])

        new_pose = axangle2twist(K_del_z[3*features.shape[1]:3*features.shape[1] + 6])
        SLAM_T[..., timestamp] = SLAM_T[..., timestamp].dot(linalg.expm(new_pose))


        SLAM_sigma = np.matmul((np.eye(3*M+6) - np.matmul(kalman_gain,SLAM_combined_jacobian)),SLAM_sigma)
        sigma[:,:,timestamp] = SLAM_sigma[3*M: 3*M+6, 3*M:3*M+6]
    return SLAM_T, world_coordinates