#! /usr/bin/env zsh
echo "========LSTM model========"
embed_dim=256
lstm_num_layers=4
decoder_num_layers=4
decoder_hidden_size=256
device="mps"
batch_size=128
epoch=7
lr=0.0001

data_dir="./pair_9_npy"
log_dir="./log"

# classification task
target_dir="./lstm_cls"

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_lstm.py \
    --cls \
    --lr $lr \
    --epoch $epoch \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --lstm_num_layers $lstm_num_layers \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/lstm_cls_"$random_seed".log 2>&1 | tee $log_dir/lstm_cls_"$random_seed".log
done

# regression task
target_dir="./lstm_reg"
epoch=45

for random_seed in {1..5}; do
  echo "random seed: $random_seed"
  python main_lstm.py \
    --epoch $epoch \
    --lr $lr \
    --random_seed $random_seed \
    --batch_size $batch_size \
    --embed_dim $embed_dim \
    --lstm_num_layers $lstm_num_layers \
    --decoder_num_layers $decoder_num_layers \
    --decoder_hidden_size $decoder_hidden_size \
    --device $device \
    --data_dir $data_dir \
    --target_dir $target_dir >> $log_dir/lstm_reg_"$random_seed".log 2>&1 | tee $log_dir/lstm_reg_"$random_seed".log
done

