# Stop or Forward: Dynamic Layer Skipping for Efficient Action Recognition

![Framework](./architecture.pdf)

## Requirements
Our experiments are conducted on 4 Titan XP (48GB):
```bash
conda env create -n sof -f ./sofnet_env.yml
conda activate sofnet
pip install tensorboardX thop 
```

## Dataset preparation
1. Move the ActivityNet-v1.3 train/test splits (and classes file) from `/data` to  `/foo/bar/activity-net-v1.3`. Here `/foo/bar` is your directory to save the datasets.
2. Download ActivityNet-v1.3 videos from [here](http://activity-net.org/download.html) (contact [them](http://activity-net.org/people.html) if there is any missing video) and save to `/foo/bar/activity-net-v1.3/videos`
3. Extract frames using the script from the repository:
``` bash
cd ./ops
python video_jpg.py /foo/bar/activity-net-v1.3/videos /foo/bar/activity-net-v1.3/frames  --parallel
```

The frames will be saved to `/foo/bar/activity-net-v1.3/frames`.


## Training
To test the models on ActivityNet-v1.3, run:
```bash
sh sof_train.sh 
```
This might take around 1~2 day.

##  Evaluation
To test the models on ActivityNet-v1.3, run:
```bash
sh sof_test.sh 
```



Our code is based on [AR-Net](https://github.com/mengyuest/AR-Net.git)
