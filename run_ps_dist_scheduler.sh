root_path=`pwd`
export DMLC_NUM_WORKER=1
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI='10.173.63.96'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
sh ./scripts/start_scheduler.sh  $root_path/build/test/src/xflow_lr
