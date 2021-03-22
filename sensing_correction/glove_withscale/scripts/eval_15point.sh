CUDA_VISIBLE_DEVICES=0 \
python eval_15point.py          \
    --data_path /home/yunzhu/Documents/knit_calib/data \
    --knit_name glove_calibration_randompressing_withvision \
    --superres 1.       \
    --resume 0          \
    --epoch -1          \
    --iter -1           \
    --position_bias 1   \
    --lam_recon 0.1 	\
    --scale_factor 1.0	\
    --debug 0           \
    --eval 1            \
    --vis 0             \
    --store 0           \
    --eval_list files/glove_calibration_randompressing_withvision_pressing_shapes.txt \

