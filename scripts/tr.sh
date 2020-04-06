CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py -i ~/data/SNEMI3D/super_resolution/ -din train_downsampled.h5 -dln train_normal.h5 -o outputs/unet_super\
  -lr 1e-03 --iteration-total 60000 --iteration-save 10000 \
  -mi 16,32,32 -mo 32,96,96 -dam 0 -moc 1\
  -to -1 -lo 0  -g 4 -c 0 -b 4  -ma super -dp 0,0,0 -dlm 255 -uvo 3,8
