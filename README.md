# DUNT

-------------------------------------------------------

Code for DUNT: Optimal Transport of Diverse Unsupervised Tasks for Robust Learning from Noisy Few-Shot Data

-------------------------------------------------------

This package contains source code for submission: MetaAux: Robust Meta-Learning for Few-Shot Learning with Noisy Labels.

The required packages are listed in the file: requirements.txt, please intall through the following command: 

pip install -r requirements.txt

The code is stored in the folder code.

To download the example noisy data of cub, please use the link: https://drive.google.com/file/d/10dZXw83fgFVrf_MQbum9rhvA2hWax54G/view?usp=sharing
and store it in the folder few_data/cub.

To run the algorithm, change the path to the current directory in the command window, and run the train.py file:

python train.py --dataset cub --dataset_aux cub --train_support_shots 1 --train_query_shots 3 --no_aug_support --noise True --pri True --aux True

For evaluation run the eval.py file:

python eval.py --dataset cub --eval_support_shots 5 --load_path checkpoints/proto_metaAux_cub_noisyTrue_cub_conv4_euclidean_1supp_3query_5bs_mixupFalse_priTrue_auxTrue

the code is cited from https://github.com/indy-lab/ProtoTransfer



