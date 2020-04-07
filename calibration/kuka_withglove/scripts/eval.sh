CUDA_VISIBLE_DEVICES=0 \
python eval.py          \
    --data_path /home/yunzhu/Documents/knit_calib/data \
    --knit_name kuka_calibration \
    --superres 1.       \
    --resume 0          \
    --epoch -1          \
    --iter -1           \
    --position_bias 1   \
    --lam_recon 0.1 	\
    --scale_factor 1.0	\
    --debug 0           \
    --eval 1            \
    --vis 1             \
    --store 0           \
    --eval_list files/kuka_videoForPaper_kuka_touching.txt \
    # --eval_list files/kuka_videoForPaper_kuka_cal.txt \

