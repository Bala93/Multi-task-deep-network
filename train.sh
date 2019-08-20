base_path=''
train_path=${base_path}'/train/image' 
val_path=${base_path}'/test/image' 
model_type='convmcd'
object_type='polyp'
save_path=${base_path}'/models'
python train.py --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} --object_type ${object_type} --save_path ${save_path}
