# STREAM

# The code for STREAM
This repo contains the source code for the STREAM model.



## Requirements

STREAM uses the following dependencies with Python 3.8

* `pytorch==2.1.0`
* `dgl==2.4.0+cu121`
* `scikit-learn==1.3.2`
* `networkx==2.8`
* `GDAL==2.3.3`
* `rtree==1.3.0`
* `numpy==1.24.4`

While the following scripts are recommended for install `dgl`.

```bash
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```



## Datasets
- This study evaluates the performance of all methods using two real-world trajectory datasets: the Chengdu and the Harbin dataset. 


- The Chengdu dataset is derived from Didi vehicle trajectory data in Chengdu collected in November 2016, with a time interval of 3.3 seconds on average, as detailed in [1]. 


- The Harbin dataset contains taxi trip data from Harbin, China, collected between January 3 to 7, 2015, with a 36.2-second time interval, which can be obtained from [2].




## Preprocessing

- The Chengdu dataset retains trajectories with lengths between 40 and 80 after map matching as following [3]. 


- The Harbin dataset includes trajectories with 20 to 300 points, where the time interval between consecutive points was less than 60 seconds, excluding any points that could not be matched to the road network. 


- Road network data was sourced from OpenStreetMap(OSM), you can download OSM data from [4] in `.osm.pbf` format directly. 


- And each dataset was split into training, validation, and test sets in a 7:1:2 ratio as in [3].


- Linear interpolation and Map matching can follow the approach in [3]. For dataset splitting, you can execute the following commands in the directory `preprocess`.

```bash
python skip_data.py
```
To obtain the single-step global transition frequency of road segments, run:

```bash
python road_trans.py
```

To obtain the global grid speed matrix for a time period, run:

```bash
python process_S.py

```
To get the local road segment speed information, run:
```bash
python get_speed.py
```

## How to run the code for STREAM

- Training & testing :  (You can also set the `keep_ratio` to 0.0625)

```bash
python multi_main.py --city Chengdu --keep_ratio 0.125
```



## Reference

[1] http://outreach.didichuxing.com/research/opendata/

[2] https://github.com/boathit/deepgtt/tree/master

[3] https://github.com/chenyuqi990215/RNTrajRec/tree/master/preprocess

[4] https://download.geofabrik.de/

