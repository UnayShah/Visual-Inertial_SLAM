# ECE 276A: Visual Inertial SLAM
## Introduction
This is project 3 of the course [ECE 276A: Sensing & Estimation in Robotics](https://natanaso.github.io/ece276a/) at UCSD, being taught by professor [Nikolay Atanisov](https://natanaso.github.io/).

This project is based on SLAM using data provided at [this link]('https://natanaso.github.io/ece276a/ref/ECE276A_PR3.zip)

IMU data is provided for a car that has camera sensors mounted on it. These include LiDAR and stereo cameras. Features are extractred from the camera and provided in the dataset.

The task is to perform SLAM on the data provided to correct landmark positions and trajectory of the car using Extended Kalman Filter.

## Running the code
1. Place the data in a folder labelled 'data' outside the folder containing the code.

        ├── data
        │   ├── 03.npz
        │   ├── 10.npz
        ├── code
        │   ├── landmark_mapping.py
        │   ├── pr3_utils.py
        │   ├── utils.py
        │   ├── visual_inertial_slam.py
        │   └── main.py

3. Install the required packages:

        pip install -r requirements.txt

4. Alternatively you can run this command:

        pip install numpy, scipy, matplotlib, tqdm, transforms3d

5. Run the file main.py. The code can be run with or without a specifying the dataset. If it is not specified, the code runs on dataset 10 by default.

        python main.py 10

## Results
The following results were obtained for dataset 10. The actual visual inertial SLAM results are incorrect, and require tuning in the updation step.

For dead reckoning:

![Dead Reckoning for Dataset 10](https://github.com/UnayShah/Visual-Inertial_SLAM/blob/master/outputs/dead_reckoning_dataset_10.png)

For Landmark Mapping with EKF:

![Landmark Mapping for Dataset 10](https://github.com/UnayShah/Visual-Inertial_SLAM/blob/master/outputs/landmark_mapping_dataset_10.png)

For Visual Inertial SLAM:

![Visual SLAM for Dataset 10](https://github.com/UnayShah/Visual-Inertial_SLAM/blob/master/outputs/visual_inertial_slam_dataset_10.png)

## References:
* [Kalman Filter]('https://natanaso.github.io/ece276a/ref/ECE276A_9_KalmanFilter.pdf#page=12')
* [Extended Kalman Filter]('https://natanaso.github.io/ece276a/ref/ECE276A_10_EKF_UKF.pdf#page=11')
* [Project Algorithm Breakdown]('https://natanaso.github.io/ece276a/ref/ECE276A_11_VI_SLAM.pdf#page=6')
