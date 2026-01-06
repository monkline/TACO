filenames=(AG/roberta_subst_10_roberta1_subst_10 SO/roberta_subst_10_roberta1_subst_10 \
 searchsnippets_trans_subst_10 biomedical_trans_subst_10 \
 TS_trans_subst_10 T_trans_subst_10 S_trans_subst_10 Tweet/trans_subst_10_trans_subst_10)
bertnames=(AG SO SS Bio G-TS G-T G-S Tweet)
datasize=(8000 20000 12340 20000 11109 11108 11108 2472)
n_classes=(4 20 8 20 152 152 152 89)
maxlen=(64 25 32 45 40 16 32 20)
batch_size=(500 1000 1000 1000 500 1000 500 2000)
accumulate_steps=(4 2 2 2 4 2 4 1)

id=(1)

echo 'begin running'

for i in ${id[*]}
do
echo "I am good at ${filenames[$i]} and ${bertnames[$i]}"

CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 python pretrain_transformers.py \
    --train_file ./augmented-datasets/${filenames[$i]}.csv \
    --val_file ./augmented-datasets/${filenames[$i]}.csv \
    --n_classes ${n_classes[$i]} \
    --model_name_or_path ./pretrained-models/distilbert-base-nli-stsb-mean-tokens \
    --output_dir ./pretrained-models/${bertnames[$i]} \
    --report_to tensorboard \
    --seed 0 \
    --num_train_epochs 10000 \
    --per_device_train_batch_size ${batch_size[$i]} \
    --per_device_eval_batch_size 1000 \
    --gradient_accumulation_steps ${accumulate_steps[$i]} \
    --checkpointing_epochs 50 \
    --with_tracking \
    --max_seq_length ${maxlen[$i]} \
    --early_stopping_loss_thresh 0.015 \
    --save_limit 3 \
    --wwm
done
