1) 
How to run Experiment
    This project is based off of this respository:

https://github.com/clovaai/stargan-v2

    The datasets and how to run the project should be there

    Here is the way to run on google colab:
    Run these statements


conda create -n stargan-v2 python=3.6.7
conda activate stargan-v2
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0


    mount the drive on the colab notebook

    run these statements

!pip install munch
!pip install ffmpeg

    run the following command to perform the training and produce output results, along with interpolation videos


python main.py --mode sample --num_domains 2 --resume_iter 100000 --w_hpf 1 \
               --checkpoint_dir expr/checkpoints/celeba_hq \
               --result_dir expr/results/celeba_hq \
               --src_dir assets/representative/celeba_hq/src \
               --ref_dir assets/representative/celeba_hq/ref

    run the following command to output the evalutation metrics

python main.py --mode eval --num_domains 2 --w_hpf 1 \
               --resume_iter 100000 \
               --train_img_dir data/celeba_hq/train \
               --val_img_dir data/celeba_hq/val \
               --checkpoint_dir expr/checkpoints/celeba_hq \
               --eval_dir expr/eval/celeba_hq


In order to run the experiments, you should also download the checkpoints in the dataset section below. These are the pretrained network options to the download.sh file, and they should create a folder called "expr/"
2)
The parts that were changed follows the instructions given on this page:
https://v-hramchenko.medium.com/modifying-stargan-v2-using-modulated-convolutions-13dc5796cd6e

Some of the changes made in the steps to convert pytorch to tensorflow broke the code before I could fix it in time, so I left these changes out.

The specific changes I used were in the model.py folder: I added a convolutional neural network block GenResBlk to replace AdainResBLk in the generator.
This is the block that should allow training on images of size 512x512. The code the author of the article used was from
https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py

added or modified code:
model.py 123-155
model.py 209-240
model.py 192, 202

The article further specified that the following evaluation metrics code in utils.py should be taken out as well

code removed:
utils.py 133-145


3)

The datasets I used were the Celeba-hq and afhq datasets, accesible through the instructions in the stargan v2 github link earlier, or using the following commands with the given download.sh file:

bash download.sh celeba-hq-dataset
bash download.sh pretrained-network-celeba-hq
bash download.sh wing

bash download.sh afhq-dataset
bash download.sh pretrained-network-afhq
