root_path=`pwd`
echo $root_path
model_name=1
epochs=10
workernum=5
python mapfeat.py
#sh ./scripts/local.sh 1 1 $root_path/build/test/src/xflow_lr $root_path/data/small_train $root_path/data/small_train $model_name $epochs
sh ./scripts/local.sh 1 $workernum $root_path/build/test/src/xflow_lr $root_path/data/train.libsvm $root_path/data/test.libsvm $model_name $epochs

