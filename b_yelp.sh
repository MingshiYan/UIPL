#!/bin/bash


# lr=('0.001' '0.005' '0.01')
# lr=('0.01' '0.001' '0.0001')
lr=('0.001')
# lr=('0.01')
emb_size=(64)
reg_weight=('0.001')
kl_regs=('0.1')
# kl_regs=('0.1' '0.5' '1.0')
# kl_regs=('0.0' '0.1' '0.3' '0.5' '0.7' '1.0')
ort_regs=('0.001')
# ort_regs=('0.0' '0.0001' '0.001' '0.01' '0.1' '1.0')
log_regs=('0.1')
# log_regs=('0.01' '0.1' '0.5')
# log_regs=('0.0')
# log_regs=('0.01' '0.1' '0.5' '1.0')
nce_regs=('0.01')
# nce_regs=('0.01' '0.1')
# nce_regs=('0.0')
# nce_regs=('0.001' '0.01' '0.1' '0.5' '1.0')
bpr_regs=('1.0')
temperature=('0.5')
# temperature=('0.1' '0.3' '0.5' '0.7' '0.9')




dataset=('yelp')
model_name='model_nce_all_pt_item_linear1_loss'
log_name='model_nce_all_pt_item_linear1_only_bpr'
device='cuda:7'
batch_size=1024
decay=('0')
# neg_count=(5 10 15 20)
neg_count=(4)
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
            for kl_reg in ${kl_regs[@]}
            do
            for ort_reg in ${ort_regs[@]}
            do
            for log_reg in ${log_regs[@]}
            do
            for nce_reg in ${nce_regs[@]}
            do
            for bpr_reg in ${bpr_regs[@]}
            do
            for n in ${neg_count[@]}
            do
            for t in ${temperature[@]}
            do
                echo 'start train: '$name

                    python main.py \
                        --data_name $name \
                        --model_name $model_name \
                        --log_name $log_name \
                        --lr $l \
                        --kl_reg $kl_reg \
                        --ort_reg $ort_reg \
                        --log_reg $log_reg \
                        --nce_reg $nce_reg \
                        --bpr_reg $bpr_reg \
                        --reg_weight $reg \
                        --embedding_size $emb \
                        --device $device \
                        --decay $dec \
                        --batch_size $batch_size \
                        --neg_count $n \
                        --nce_temperature $t

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
        done
    done
done