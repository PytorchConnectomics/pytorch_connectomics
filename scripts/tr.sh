<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0 python -u train.py -i /n/home11/mbelhamissi/data/SNEMI3D/super_resolution/ -din train_downsampled.h5 -dln train_normal.h5 -o outputs/unet_super\
  -lr 1e-03 --iteration-total 60000 --iteration-save 100 \
  -mi 16,32,32 -mo 32,96,96 -dam 0 -moc 1 -uvo 3,8 -ulo 1,1,1 \
  -to -1 -lo 0  -g 1 -c 0 -b 4  -ma super -dp 0,0,0 -dlm 255
=======
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py -i ~/data/SNEMI3D/super_resolution/ -din train_downsampled.h5 -dln train_normal.h5 -o outputs/unet_super\
  -lr 1e-03 --iteration-total 60000 --iteration-save 10000 \
  -mi 16,32,32 -mo 32,96,96 -dam 0 -moc 1\
  -to -1 -lo 0  -g 4 -c 0 -b 4  -ma super -dp 0,0,0 -dlm 255 -uvo 3,8
>>>>>>> fd3d4c7e07740f9c1c8b624ca2239a74c4a8161d
