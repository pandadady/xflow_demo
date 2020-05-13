root_path=`pwd`
echo $root_path
if [ -z $1 ]
then
    echo " please input model name "
    exit -1
fi
tmp=`date -d "0 day ago" +"%Y%m%d%H%M"`
modelname=$1.$tmp
modeltype=1
epochs=10
workernum=5
python mapfeat.py $workernum
#sh ./scripts/local.sh 1 1 $root_path/build/test/src/xflow_lr $root_path/data/small_train $root_path/data/small_train $model_name $epochs
sh ./scripts/local.sh 1 $workernum $root_path/build/test/src/xflow_lr $root_path/data/train.libsvm $root_path/data/test.libsvm $modeltype $epochs $modelname

cat model/model.$modelname.* | sort -nk 1 | uniq -c > model/model.all
