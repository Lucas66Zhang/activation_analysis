#! /usr/bin/env zsh
echo "========gru model========"
embed_dim=256
gru_num_layers=4
decoder_num_layers=4
gru_hidden_size=256
decoder_hidden_size=256
device="mps"
batch_size=128
epoch=50
lr=0.0001

data_dir="./pair_9_npy"
log_dir="./log"

 classification task
target_dir="./gru_cls"

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_gru.py \
    --cls \
    --lr $lr \
    --epoch $epoch \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --gru_num_layers $gru_num_layers \
    --gru_hidden_size $gru_hidden_size \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/gru_cls_"$random_seed".log 2>&1 | tee $log_dir/gru_cls_"$random_seed".log
done

# regression task
target_dir="./gru_reg"
epoch=50

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_gru.py \
    --epoch $epoch \
    --lr $lr \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --gru_num_layers $gru_num_layers \
    --gru_hidden_size $gru_hidden_size \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/gru_reg_"$random_seed".log 2>&1 | tee $log_dir/gru_reg_"$random_seed".log
done

