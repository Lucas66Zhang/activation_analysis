#! /usr/bin/env zsh
nhead=8
embed_dim=256
dim_feedforward=2048
decoder_num_layers=4
decoder_hidden_size=256
device="mps"
batch_size=128
epoch=13

data_dir="./pair_9_npy"
log_dir="./log"

# classification task
target_dir="./transformer_cls"

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_transformer.py \
    --cls \
    --epoch $epoch \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --nhead $nhead \
    --dim_feedforward $dim_feedforward \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/transformer_cls_"$random_seed".log 2>&1 | tee $log_dir/transformer_cls_"$random_seed".log
done

# regression task
target_dir="./transformer_reg"
epoch=12

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_transformer.py \
    --epoch $epoch \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --nhead $nhead \
    --dim_feedforward $dim_feedforward \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/transformer_reg_"$random_seed".log 2>&1 | tee $log_dir/transformer_reg_"$random_seed".log
done


