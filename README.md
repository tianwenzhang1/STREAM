# STREAM

# The code for STREAM
This repo contains the source code for the STREAM model.



---

## Requirements

---

GTR-HMD uses the following dependencies with Python 3.8

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



---

## Datasets
---

### Datasets Sourcess

This study evaluates the performance of all methods using two real-world trajectory datasets: the Chengdu and the Harbin dataset. The Chengdu dataset is derived from Didi vehicle trajectory data in Chengdu collected in November 2016, with a time interval of 3.3 seconds on average, as detailed in [1]. The Harbin dataset contains taxi trip data from Harbin, China, collected between January 3 -- 7, 2015, with a 36.2-second time interval, which can be obtained from [2].

### Datasets preprocessing

The Chengdu dataset retains trajectories with lengths between 40 and 80 after map matching as following [3]. The Harbin dataset includes trajectories with 20 to 300 points, where the time interval between consecutive points was less than 60 seconds, excluding any points that could not be matched to the road network. Road network data was sourced from OpenStreetMap. And each dataset was split into training, validation, and test sets in a 7:1:2 ratio as in [3].

Linear interpolation and Map matching can follow the approach in RNTrajRec. For dataset splitting, you can execute the following commands:
```
cd preprocess
python skip_data.py
```


---

## How to run the code for ReadGraph

---

```bash
python main.py --city chengdu --keep ratio 0.125
```



---

## Reference

---

[1] http://outreach.didichuxing.com/research/opendata/

[2] https://github.com/boathit/deepgtt/tree/master

[3] https://github.com/chenyuqi990215/RNTrajRec/tree/master/preprocess

