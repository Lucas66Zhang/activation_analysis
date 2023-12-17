#! /usr/bin/env zsh
echo "========rnn model========"
embed_dim=256
rnn_hidden_size=256
rnn_num_layers=4
decoder_num_layers=4
decoder_hidden_size=256
device="mps"
batch_size=128
epoch=11
lr=0.001

data_dir="./pair_9_npy"
log_dir="./log"

# classification task
target_dir="./rnn_cls"

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_rnn.py \
    --cls \
    --lr $lr \
    --epoch $epoch \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --rnn_num_layers $rnn_num_layers \
    --rnn_hidden_size $rnn_hidden_size \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/rnn_cls_"$random_seed".log 2>&1 | tee $log_dir/rnn_cls_"$random_seed".log
done

# regression task
target_dir="./rnn_reg"
epoch=14

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_rnn.py \
    --epoch $epoch \
    --lr $lr \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --rnn_num_layers $rnn_num_layers \
    --rnn_hidden_size $rnn_hidden_size \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/rnn_reg_"$random_seed".log 2>&1 | tee $log_dir/rnn_reg_"$random_seed".log
done

