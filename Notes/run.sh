cd /media/MMVCNYLOCAL/MMVC_NY/David_jin/TextBoxes-TensorFlow/


     
DATASET_DIR=./data/ICDAR2013/  

export TF_CUDNN_USE_AUTOTUNE=0
CHECKPOINT_PATH=./logs_426/model.ckpt-48406

CHECKPOINT_PATH=./checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt
CHECKPOINT_PATH=./checkpoints/model.ckpt-182167
CHECKPOINT_PATH=./logs/train/adam_vgg/model.ckpt-75000
CHECKPOINT_PATH=./logs/train/momentum_batch_new_hard/model.ckpt-182167
DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/train/momentum_601
CUDA_VISIBLE_DEVICES=4,5,6,7 setsid python Textbox_train.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=120 \
	--save_interval_secs=1800 \
	--weight_decay=0.0005 \
	--optimizer=momentum \
	--learning_rate=0.001 \
	--batch_size=32 \
	--match_threshold=0.5 \
	--num_samples=400000 \
	--gpu_memory_fraction=0.95 \
	--max_number_of_steps=200000 \
	--num_clones=4 \
    --num_readers=4 \
    --num_preprocessing_threads=4 
	--checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --ignore_missing_vars=True \
    --use_batch=False
    --learning_rate_decay_factor=0.5 \


    

CHECKPOINT_PATH=./logs/train/momentum_529_bias
EVAL_DIR=./logs/evals/momentum_529_bias
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=1 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --wait_for_checkpoints=True \
    --batch_size=1 \
    --gpu_memory_fraction=0.05 \
    --use_batch=False \


CHECKPOINT_PATH=./logs/train/momentum_reg_bias
EVAL_DIR=./logs/evals/momentum_reg_bias
DATASET_DIR=./data/ICDAR2013/test
CUDA_VISIBLE_DEVICES=1 setsid python eval.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1 \
    --wait_for_checkpoints=True \
    --gpu_memory_fraction=0.06 \
    --use_whiten=False \
    --model_name=txtbox_512






################################ 单机模式
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt
CHECKPOINT_PATH=./checkpoints/model.ckpt-12325
DATASET_DIR=./data/sythtext/
TRAIN_DIR=./logs/
python train.py \
	--train_dir=${TRAIN_DIR} \
	--dataset_dir=${DATASET_DIR} \
	--save_summaries_secs=60 \
	--save_interval_secs=600 \
	--weight_decay=0.0005 \
	--learning_rate=0.001 \
	--batch_size=1 \
	--max_number_of_steps=400000 \