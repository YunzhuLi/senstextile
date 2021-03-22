CUDA_VISIBLE_DEVICES=0 \
python eval.py          \
    --data_path /home/yunzhu/Documents/knit_calib/data \
    --knit_name sock_calibration_withscale/sock_calibration_right \
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
    --eval_list files/sock_withMocap_for_paper_right.txt
    # --eval_list files/sock_withMocap_final_left.txt \
    # --eval_list files/sock_videoForPaper_sock_data_updated_left.txt \
    # --eval_list files/sock_videoForPaper_sock_data_updated_right.txt \
    # --eval_list files/sock_videoForPaper_sock_data_right.txt \
    # --eval_list files/sock_videoForPaper_sock_data_left.txt \
    # --eval_list files/sock_classification_right.txt \
    # --eval_list files/sock_videoForPaper_sock_cal_updated.txt \
    # --eval_list files/sock_data_withMocap_left.txt \
    # --eval_list files/sock_calibration_left.txt \
    # --eval_list files/sock_videoForPaper.txt \
