# Graph-Based Trajectory Recovery with Hierarchical Movement Dynamics
# The code for GTR-HMD
This repo contains the source code for the GTR-HMD model.
## Requirements
* `Python==3.8.18`
* `pytorch==2.1.0`
* `dgl==2.3.0+cu118`
* `scikit-learn==1.3.2`
* `networkx==3.1`
* `GDAL==2.3.3`
* `rtree==1.3.0`
* `numpy==1.24.4`
## Datasets
### Datasets Sourcess
The Chengdu dataset is derived from Didi vehicle trajectory data in Chengdu collected in November 2016, as detailed in [RNTrajRec](https://github.com/chenyuqi990215/RNTrajRec/tree/master/preprocess). The Harbin dataset includes data from over 13,000 taxis, recording more than 1 million trips over 5 days, which can be obtained from [DeepGTT](https://github.com/boathit/deepgtt/tree/master).
### Datasets preprocessing
Linear interpolation and Map matching can follow the approach in RNTrajRec. For dataset splitting, you can execute the following commands:
```
cd preprocess
python skip_data.py
```
## How to run the code for ReadGraph
```
python main.py --city Porto --keep ratio 0.125
```

