# deception_detection
# Description of the two datasets
## 'Court' dataset
The 'Court' dataset comes from literature[1], which is collected from court trial videos. There are 121 videos in total, 60 videos are labeled 'truth' and 61 are labeled 'lie'. The video file is in `court_video`, and the video frame image file is in `data/court`. In order to ensure the reliability of the results, we conduct 5 experiments for each model and take the average value to obtain the accuracy. In each experiment, 118 videos are randomly divided into 99 training set and 19 test set(3 videos with obvious flaws were discarded).
## 'Pandakill' dataset
We collect and publish a new dataset here called the 'Pandakill' dataset. We collected the video of the pandakill competition in the second season and edited each player's speech each time to form a video sample, which is available in https://www.iqiyi.com/a_19rrh9xjrl.html. There are a total of 200 video samples, and the duration of each video sample is 10-50 seconds. There are 15 players in all videos. If the player is a good person, it is marked as 'truth'. If the player is a werewolf, it is marked as 'lie'. In each experiment, 160 videos are randomly divided into training set and 40 test set. In order to ensure the reliability of the results, we conduct 5 experiments for each model and take the average value to obtain the accuracy. In each experiment, 160 videos are randomly divided into training set and 40 test set. The video frame image file is in `data/pandakill_row`, and the video frame image file is in `data/pandakill` after object detection.
# Source code
The source code of the Multi-stream-merge model is located in the folder `multi-stream`.

# Training
To train a new model, use the `main.py` script.

The template command to produce model of Multi-stream-merge on the dataset can be:

`python main.py  <train_list>  <test_list>  <result_path>  --<dataset> -b. –gpus`

The template command to test model of Multi-stream-merge on the dataset can be:

`python main.py  <train_list>  <test_list>  <result_path>  --<dataset> -b. –gpus -e --resume`

## Training Multi-stream-merge on 'Pandakill' dataset
For example, in order to train the Multi-stream-merge on 'Pandakill' datset:

The <train_list> is shown in `/data/train_pandakill_5.txt` and <test_list> is shown in `/data/test_pandakill_5.txt`.

We first need to extract local features through TSN:

`python main.py /data/train_pandakill_5.txt /data/test_pandakill_5.txt /data/pandakill_result/ /home/glq/exp/data/pandakill --dataset pandakill -b 2 --gpus 0 1`

Then fusion training to get Multi-stream-merge:

`python main.py /data/train_pandakill_5.txt /data/test_pandakill_5.txt /data/pandakill_result/ /data/pandakill --dataset pandakill -b 2 --gpus 0 1 -merge --resume /data/pandakill_result/_rgb_checkpoint.pth.tar`

The command to test Multi-stream-merge can be:
`python main.py /data/train_pandakill_5.txt /data/test_pandakill_5.txt /data/pandakill_result/ /home/glq/exp/data/pandakill --dataset pandakill -b 2 --gpus 0 1 -merge --resume /data/pandakill_result/_rgb_checkpoint_merge.pth.tar -e`

The log during the training and the testing resulting are shown in `data/result_pandakill_5`.



# Comparative Experiment
In addition to our proposed model, we have also reproduced the 3 baseline models (TSN, LSTM, 3D-CNN) in the folder `baseline`.

The command to produce model of TSN on the 'Court' dataset can be:

`python main.py pandakill RGB /data/train_pandakill.txt /data/test_pandakill.txt --arch resnet101 --num_segments 3 --lr 0.001 --lr_steps 30 60 -b 4 --epochs 40 --dropout 0.8 --gpus 0 1`

The command to produce model of LSTM on the 'Pandakill' dataset can be:

`CUDA_VISIBLE_DEVICES=0,1 python main.py --batch_size 16 --video_path pandakill --annotation_path pandakill.json --result_path result_lstm_pandakill --dataset pandakill --lstm Truth`

The command to produce model of 3D-CNN on the 'Pandakill' dataset can be:

`python main.py pandakill RGB /data/train_pandakill.txt /data/test_pandakill.txt --arch resnet101 --num_segments 3 --lr 0.001 --lr_steps 30 60 -b 4 --epochs 40 --dropout 0.8 --gpus 0 1`

# Reference
[1] V. Pe ŕ ez-Rosas, M. Abouelenien, R. Mihalcea, and M. Burzo. Deception detection using real- life trial data. In International Conference on Multimodal In- teraction, pages 59–66, 2015. 
