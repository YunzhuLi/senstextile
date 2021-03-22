CUDA_VISIBLE_DEVICES=0  \
python train.py \
    --data_path ../data_classification/glove_objclaassification_26obj/ \
    --n_obj 26 \
    --n_round 4 \
    --subsample 1

