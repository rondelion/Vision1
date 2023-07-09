# Vision1
Basic Salience - Saccade model

## Features

Human vision uses saccades that direct the gaze to a salient part in the visual field.

This project includes a simple visual environment (`Vision1Env.py`) with [Gymnasium](https://gymnasium.farama.org).  It gives an observation of visual objects (circles) with varying sizes and brightness.  The environment receives saccade commands to move the gaze.  It uses [PyGame](https://www.pygame.org/news) to render the observation.  
<p align="center">
<img src="/Vision1.png" width="500px"/><br><strong>Fig.1</strong><br>
Observation Size = Scene Image<br> Size = Grid Size × Scene Size<br>
Stage Image Size = ½ Scene Image Size<br>
Visual objects are displayed in the Stage.</p>
  
The project also includes a simple agent (`Vision1.py`) that directs its gaze to the most salient object in the observation by a sending saccade command to the environment.  It uses [BriCA](https://github.com/wbap/BriCA1) (a framework for Brain-inspired Computing Architecture) and [OpenCV](https://pypi.org/project/opencv-python/) for image processing.

### Saliency calculation
`Periphery2Saliency`  
  
The weighted sum of the following:
* Brightness (intensity map)
* Spatial integral and differential in the following order:
    1. ½ size reduction (resize -- integral)
    2. edge detection (differential)
    3. size reduction (resize -- integral)
* Temporal differential  
brightness increase from the previous intensity map

### Saccade
`PriorityMap2Gaze`  
  
Saccade to the position in the saliency map after accumulation (temporal integral) with a given decay constant.


## How to Install
* Clone the repository

* Install [BriCA](https://github.com/wbap/BriCA1) and [BriCAL](https://github.com/wbap/BriCAL).

* Install gymnasium, tensorflow (for TensorBoard), cv2 (OpenCV), and pygame for Python

* Register the environment to Gym
    * Place `MinWMEnvA.py` file in `gymnasium/envs/myenv`  
    (wherever Gym to be used is installed)
    * Add to `__init__.py` (located in the same folder)  
      `from gymnasium.envs.myenv.Vision1Env import Vision1Env`
    * Add to `gymnasium/envs/__init__.py`  
```
register(
    id="Vision1Env-v0",
    entry_point="gymnasium.envs.myenv:Vision1Env",
    max_episode_steps=1000,
)
```

## Usage

Required files:
* `Vision1.json`: config file
* `Vision1.brical.json` : architecture description for BriCAL
