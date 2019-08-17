base_path='/media/htic/NewVolume3/Balamurali/polyp-segmentation/train_valid'
train_path=${base_path}'/train/image' 
val_path=${base_path}'/test/image' 
model_type='convmcd'
object_type='polyp'
save_path=${base_path}'/models'

sudo /home/htic/anaconda2/envs/torch4/bin/python -W ignore train.py --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} --object_type ${object_type} --save_path ${save_path}
