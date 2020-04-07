CUDA_VISIBLE_DEVICES=0  \
python train.py \
    --data_path ../../../data/glove_objclaassification_26obj/newglove \
    --n_obj 26 \
    --n_round 4 \
    --subsample 1

