CUDA_VISIBLE_DEVICES=0	\
python calib.py 	\
    --data_path ../data_sensing_correction/ \
    --knit_name kuka_calibration\
    --superres 1.	\
    --resume 0		\
    --epoch -1		\
    --iter -1		\
    --position_bias 1	\
    --lam_recon 0.1	\
    --scale_factor 1.0	\
    --debug 0		\
    --vis 0		\
    --eval 0

