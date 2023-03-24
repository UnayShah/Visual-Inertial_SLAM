import numpy as np
from scipy import linalg
import transforms3d
import re
import sys
from tqdm import tqdm
from pr3_utils import *
from utils import *
from landmark_mapping import landmark_mapping
from visual_inertial_slam import visual_inertial_slam

import os


if __name__ == '__main__':
	if not os.path.exists('.\\outputs'):
		os.mkdir('.\\outputs')
	dataset_number = '10'

	if len(sys.argv) > 1 and re.match('[0-9]+', sys.argv[1]) and sys.argv[1] in ['03', '10']:
		dataset_number = sys.argv[1]
		print('Using dataset {}'.format(sys.argv[1]))
	else: print('Defaulting to dataset {}'.format(dataset_number))
    
	# Load the measurements
	filename = "./data/{}.npz".format(dataset_number)
	t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(
        filename)
	correction_R = transforms3d.euler.euler2mat(np.pi, 0, 0)
	correction_T = np.zeros([4, 4])
	correction_T[:3, :3] = correction_R
	correction_T[-1, -1] = 1
	imu_T_cam = np.dot(correction_T, imu_T_cam)
	downsample_features = True
	Ks = get_Ks(K, b)

	features = features.round().astype(int)
	if downsample_features:
		features = features[:, ::50]
	downsample_features = False
	hidden_features = np.ones([4, 1])*-1
	empty_features = np.zeros([4, 1])

	twist_matrix = np.zeros([angular_velocity.shape[1], 4, 4])
	twist_matrix[..., :3, :3] = axangle2skew(angular_velocity.T)
	twist_matrix[..., :-1, -1] = linear_velocity.T

	T = [np.eye(4)]
	mu = np.zeros(features.shape[:-1])
	for i in tqdm(range(twist_matrix.shape[0]-1)):
		T.append(
			np.dot(T[-1], linalg.expm((t[0, i+1]-t[0, i]) * twist_matrix[i])))
		visible_features_positions = np.where(
			np.all(features[..., i] != hidden_features, 0))[0]
		visible_features = features[:, visible_features_positions, i]
		mu[:, visible_features_positions] = pixel_to_world(
			visible_features, K, b, imu_T_cam, T[i])

	T = np.array(T)
	T = np.transpose(T, [1, 2, 0])

	plt.title('Dead Reckoning Trajectory and Landmarks Dataset {}'.format(dataset_number))
	plt.plot(T[0, -1], T[1, -1], c='b')
	plt.scatter(mu[0], mu[1], c='y')
	plt.savefig('.\\outputs\\dead_reckoning_dataset_{}'.format(dataset_number))
	plt.show()

	# (b) Landmark Mapping via EKF Update
	T, mu = landmark_mapping(features, imu_T_cam, hidden_features, empty_features, K, b, Ks, T, t)
	plt.title('Landmark Mapping via EKF Dataset {}'.format(dataset_number))
	plt.plot(T[0, -1], T[1, -1], c='b')
	plt.scatter(mu[0], mu[1], c='y')
	plt.savefig('.\\outputs\\landmark_mapping_dataset_{}'.format(dataset_number))
	plt.show()

	# (c) Visual-Inertial SLAM
	T, mu = visual_inertial_slam(features, imu_T_cam, hidden_features, empty_features, K, b, Ks, T, t, linear_velocity, angular_velocity)
	plt.title('Visual Inertial Mapping via EKF Dataset {}'.format(dataset_number))
	plt.plot(T[0, -1], T[1, -1], c='b')
	plt.scatter(mu[0], mu[1], c='y')
	plt.savefig('.\\outputs\\visual_inertial_slam_dataset_{}'.format(dataset_number))
	plt.show()
