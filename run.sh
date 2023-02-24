batch=200
group=$1
unseen_classes=oos
seed=2020
dataset=CLINC_OOD_full
train_class_num=n
detect_method="msp"

# train
if [ $group = 0 ];then
    ce_pre_epoches=200
    python main.py --dataset $dataset --mode train --unseen_classes $unseen_classes --seed $seed --batch_size $batch --cuda --train_class_num $train_class_num --ce_pre_epoches $ce_pre_epoches --experiment_No ce$ce_pre_epoches

# test
elif [ $group = 1 ];then
    model_dir="./output/CLINC_OOD_full-1_oos-n-2020-None-ce200/"

    python main.py --dataset $dataset --mode test --model_dir $model_dir --detect_method $detect_method  --unseen_classes $unseen_classes --seed $seed --batch_size $batch --cuda --train_class_num $train_class_num

# both
elif [ $group = 2 ];then
    ce_pre_epoches=200

    python main.py --dataset $dataset --mode both --unseen_classes $unseen_classes --detect_method $detect_method --seed $seed --batch_size $batch --cuda --train_class_num $train_class_num --ce_pre_epoches $ce_pre_epoches --experiment_No ce$ce_pre_epoches

# Bayes
elif [ $group = 3 ];then
    model_dir="./output/CLINC_OOD_full-1_oos-n-2020-None-ce200/"
    mc_dropout=True
    mc_pro=0.7
    mc_round=150
    dataset='CLINC_OOD_full'

    python main.py --mode test --model_dir $model_dir --mc_dropout --mc_round $mc_round --mc_pro $mc_pro --dataset $dataset --unseen_classes $unseen_classes --detect_method $detect_method --seed $seed --batch_size $batch --cuda --train_class_num $train_class_num
fi
