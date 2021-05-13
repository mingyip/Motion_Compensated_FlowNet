# Motion Compensated FlowNet


<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/optical_flow.gif?raw=true" width="420px">
      </p>
      Event Data Optical Flow Estimation
    </th>
    <th class="tg-0lax">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/outdoor1_trajectory.png?raw=true" width="420px">
      </p>
      Odometry Estimation
    </th>
  </tr>
</thead>
</table>

## Abstract
We developed an algorithm to estimate the optical flow of a scene and correspond camera odometry from a sequence of **event data**. This idea was adapted from the literature *"Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion."* Alex Zihao Zhu, Liangzhe Yuan, Kenneth Chaney, Kostas Daniilidis. [ArXiV](https://arxiv.org/pdf/1812.08156.pdf) 2018. [1]

Also, check out our **event interest point detection** project. https://github.com/mingyip/pytorch-superpoint


## Installation
The environment is run in python 3.6, Pytorch 1.5.0 and ROS. We ran our code with Ubuntu 18.04 and ROS Melodic. Installation instructions for *ROS* can be found [here](http://wiki.ros.org/melodic/Installation/Ubuntu). To generate syntheic event data, we used "ESIM: an Open Event Camera Simulator". You may find installation details of *ESIM* [here](https://github.com/uzh-rpg/rpg_esim).

#### To install conda env
```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt # install pytorch
```

#### To install Ros Melodic 
```
sudo apt-get update
sudo apt-get install ros-melodic-desktop-full
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
```
After installed Ros, don't forget to install the Event Camera Simulator. 


## Dataset
we used data sequences (in ros format) from [MVSEC](https://daniilidis-group.github.io/mvsec/) [2] and [IJRR](http://rpg.ifi.uzh.ch/davis_data.html) (ETH event dataset) [1] to further train our network. This code processes the events in HDF5 format. To convert the rosbags to this format, open a new terminal and source a ROS workspace. We command to use packages from https://github.com/TimoStoff/event_cnn_minimal
```
source /opt/ros/kinetic/setup.bash
python events_contrast_maximization/tools/rosbag_to_h5.py <path/to/rosbag/or/dir/with/rosbags> --output_dir <path/to/save_h5_events> --event_topic <event_topic> --image_topic <image_topic>
```

## Usage

To train the network with the dataset. To set training parameters, go to `config.py` file.
```
python train.py --load_path data/outdoor_day1_data.h5
```

To Evaluate the Relative Pose Error run with associate.py

The program output an tragetory of the estimated path together with the error rate.
```
python associate.py gt.txt estimated.txt
```

## Result

#### Odometry Estimation

The evaluation is done under our evaluation scripts. We evaluated our algorithm using relative pose error corresponds to the drift of the trajectory. Moreover, we also calculated the percentange of outliners *>0.5* and *>1.0*, where *2.0* is the max error rate

<p align="center">
  <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/rpe1.png?raw=true" width="420px">
</p>
<p align="center">
  <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/rpe2.png?raw=true" width="420px">
</p>


<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/outdoor1_trajectory.png?raw=true" width="420px">
      </p>
    </th>
    <th class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/night1_indoor4.png?raw=true" width="420px">
      </p>
    </th>
  </tr>
</thead>
</table>

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Dataset</th>
    <th class="tg-0pky">Sequence</th>
    <th class="tg-0pky">RPE (median)</th>
    <th class="tg-0pky">RPE (mean)</th>
    <th class="tg-0pky">% Outlier (&gt;0.5)</th>
    <th class="tg-0pky">% Outlier (&gt;1.0)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">MVSEC</td>
    <td class="tg-0pky">Indoor_flying1</td>
    <td class="tg-0pky">0.3856</td>
    <td class="tg-0pky">0.5213</td>
    <td class="tg-0pky">38.63</td>
    <td class="tg-0pky">15.00</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Indoor_flying2</td>
    <td class="tg-0pky">0.3820</td>
    <td class="tg-0pky">0.5333</td>
    <td class="tg-0pky">39.79</td>
    <td class="tg-0pky">15.56</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Indoor_flying3</td>
    <td class="tg-0pky">0.4045</td>
    <td class="tg-0pky">0.5684</td>
    <td class="tg-0pky">39.36</td>
    <td class="tg-0pky">20.47</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Indoor_flying4</td>
    <td class="tg-0pky">0.5217</td>
    <td class="tg-0pky">0.5919</td>
    <td class="tg-0pky">52.41</td>
    <td class="tg-0pky">17.74</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Outdoor_day1*</td>
    <td class="tg-0pky">0.1039</td>
    <td class="tg-0pky">0.1363</td>
    <td class="tg-0pky">1.44</td>
    <td class="tg-0pky">1.44</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Outdoor_day2**</td>
    <td class="tg-0pky">0.1301</td>
    <td class="tg-0pky">0.3527</td>
    <td class="tg-0pky">21.78</td>
    <td class="tg-0pky">16.83</td>
  </tr>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Outdoor_night</td>
    <td class="tg-0pky">0.1270</td>
    <td class="tg-0pky">0.2725</td>
    <td class="tg-0pky">15.88</td>
    <td class="tg-0pky">10.90</td>
  </tr>
  <tr>
    <td class="tg-0pky">IRJJ</td>
    <td class="tg-0pky">Poster_translation</td>
    <td class="tg-0pky">0.2678</td>
    <td class="tg-0pky">0.6211</td>
    <td class="tg-0pky">41.12</td>
    <td class="tg-0pky">34.67</td>
  </tr>
</tbody>
</table>
* tested on no sunlight scene

** training Set






#### Estimated Optical Flow
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Orignal Event Images</th>
    <th class="tg-0pky">Deblured Images</th>
    <th class="tg-0pky">Estimated Optical Flow</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/blur1.png?raw=true" width="360px">
      </p>
    </td>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/deblur2.png?raw=true" width="360px">
      </p>
    </td>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/opticalflow1.png?raw=true" width="360px">
      </p>
    </td>
  </tr>
  <tr>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/blur2.png?raw=true" width="360px">
      </p>
    </td>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/deblur2.png?raw=true" width="360px">
      </p>
    </td>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/opticalflow2.png?raw=true" width="360px">
      </p>
    </td>
  </tr>
  <tr>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/blur3.png?raw=true" width="360px">
      </p>
    </td>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/deblur3.png?raw=true" width="360px">
      </p>
    </td>
    <td class="tg-0pky">
      <p align="center">
        <img src="https://github.com/mingyip/Motion_Compensated_FlowNet/blob/master/img/opticalflow3.png?raw=true" width="360px">
      </p>
    </td>
  </tr>
</tbody>
</table>

## Reference
[1] Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers, D. (2012). A benchmark for the evaluation of RGB-D SLAM systems. 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 573-580.

[2] Zhu, A.Z., Yuan, L., Chaney, K., & Daniilidis, K. (2019). Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 989-997.

[3] Zhu, A.Z., Thakur, D., Özaslan, T., Pfrommer, B., Kumar, V., & Daniilidis, K. (2018). The Multivehicle Stereo Event Camera Dataset: An Event Camera Dataset for 3D Perception. IEEE Robotics and Automation Letters, 3, 2032-2039.

[4] Mueggler, E., Rebecq, H., Gallego, G., Delbrück, T., & Scaramuzza, D. (2017). The event-camera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM. The International Journal of Robotics Research, 36, 142 - 149.

[5] Stoffregen, T., Scheerlinck, C., Scaramuzza, D., Drummond, T., Barnes, N., Kleeman, L., & Mahony, R. (2020). Reducing the Sim-to-Real Gap for Event Cameras. ECCV.
