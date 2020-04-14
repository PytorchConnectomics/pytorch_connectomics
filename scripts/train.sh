CUDA_VISIBLE_DEVICES=0, python train.py -i /n/home11/mbelhamissi/data/SNEMI3D/super_resolution/\
    -o outputs/unetv3 -din train_downsampled.h5 -dln label_downsampled.h5 \
    -lr 1e-03 --iteration-total 100000 --iteration-save 1000 -dp 0,0,0\
    -mi 18,160,160 -ma unet_residual_3d -moc 3 \
    -to 2 -lo 1 -wo 1 -g 1 -c 4 -b 4

