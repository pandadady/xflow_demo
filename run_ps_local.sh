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
epochs=5
workernum=5
testflag=$2
#python mapfeat.py $workernum $testflag
#sh ./scripts/local.sh 1 1 $root_path/build/test/src/xflow_lr $root_path/data/small_train $root_path/data/small_train $model_name $epochs
sh ./scripts/local.sh 1 $workernum $root_path/build/test/src/xflow_lr $root_path/data/small_train $root_path/data/small_train $modeltype $epochs $modelname $workernum

#cat model/model.$modelname.* |sort -k 1n -u> model/model.all
#wc -l model/model.all
