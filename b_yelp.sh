#!/bin/bash


# lr=('0.001' '0.005' '0.01')
# lr=('0.01' '0.001' '0.0001')
lr=('0.001')
# lr=('0.01')
# reg_weight=('0.01' '0.001' '0.0001')
emb_size=(64)
# lr=('0.0005')
reg_weight=('0.001')
# log_regs=('0.5' '0.7' '1.0')
# log_regs=('0.9' '0.7' '0.5')
# log_regs=('1.0' '1.2' '1.5')
# lamb=('0.3' '0.5' '0.7')
# lamb=('0' '0.3' '0.5' '0.7' '1')
# beta=('0' '0.3' '0.7' '1')
lamb=('0.7')
beta=('0.7')
# lamb=('0.5')
# beta=('1.0')
# behaviors=("['pos']" "['tip', 'pos']" "['neutral', 'pos']" "['neg', 'pos']" "['tip', 'neutral', 'pos']" "['tip', 'neg', 'pos']" "['neutral', 'neg', 'pos']")
behaviors=("['tip', 'neutral', 'neg', 'pos']")

dataset=('yelp')
model_name='Tanh_dist'
device='cuda:3'
batch_size=1024
decay=('0')
gpu_no=1
# neg_count=(5 10 15 20)
neg_count=(10)
# model_name='base'
# model_name='lightGCN'

for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
            for dec in ${decay[@]}
            do
            for log_reg in ${lamb[@]}
            do
            for b in ${beta[@]}
            do
            for bhv in "${behaviors[@]}"
            do
            for n in "${neg_count[@]}"
            do
                echo 'start train: '$name

                    python main_x.py \
                        --lr ${l} \
                        --reg_weight ${reg} \
                        --lamb ${log_reg} \
                        --beta ${b} \
                        --data_name $name \
                        --embedding_size $emb \
                        --device $device \
                        --decay $dec \
                        --batch_size $batch_size \
                        --gpu_no $gpu_no \
                        --neg_count $n \
                        --behaviors "${bhv}" \
                        --model_name $model_name

                echo 'train end: '$name
            done
            done
            done
            done
            done
            done
        done
    done
done