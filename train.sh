CUDA_VISIBLE_DEVICES=1 python -u train.py \
--database WPC \
--model_type swin \
--lr 0.0001 \
--epochs 50 \
--train_batch_size 32 \
--num_workers 8 \
--img_length_read 6 \
--images_dir path_to_the_wpc_6face_folder \
>> logs/WPC.log

