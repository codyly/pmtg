# PMTG Synthetic Control (Chpt 3.) example

### Usage
```python
# Training
python3 trainPMTG.py

# Testing
python3 evalPMTG.py
```

### Files
* figs: directory to store output images
* utils: utility scripts for loading trajectories and visualizing results
* weights: for saving weights
* trainPMTG.py: training script
* evalPMTG.py: testing script
* painter.py: a tkInter program for customizing target trajectories
* net.py: policy network architecture
* modulated_trajectory_generator: implementation of trajectory generator


### Photos
![Test Result](https://github.com/codyly/pmtg/blob/main/pmtg-synthetic-control/figs/original_trajectory.png) Original Trajectory to Synthesize
![Reward Curve](https://github.com/codyly/pmtg/blob/main/pmtg-synthetic-control/figs/Reward.png) Training Curve
![Test Result](https://github.com/codyly/pmtg/blob/main/pmtg-synthetic-control/figs/Trajecotry%20Eval.png) Test Result
