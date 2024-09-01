python hsi_denoising_gauss_95_20.py --gpu-ids 4 -a irdnet18 --out_channel 16 \
 --grow 16 --grow_num 3 --out_num 5 --unfoldings 0 \
-p SVD_wawelets1_addsups0_unetlay3_WE0_cat_unfoldings0_eq3_batchsize8_lr1e-4_ICVL100_Corr_retry --kernel_size 3 --batchSize 8 --lr 1e-4 \
--dataroot /data/denoising/ICVL/ICVL64_31.db --testroot ../CVPR2022/ICVL/test \
-gr ../CVPR2022/ICVL/test_crop/ --outdir './checkpints/' --threads 16
