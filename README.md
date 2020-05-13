# xflow_demo
Distributed training model(LR, FM) demo using ps-lite.  FTRL and SGD Optimization Algorithm.
## 1. Introduction

Distributed LR With Parameter Server 

## 2. Install
        
### 2.1 build 
```
        sh work.sh make
```
### 2.2 deploy
```
        sh work.sh deploy
``` 

## 3.Run
### 3.1 Local 
```
sh run_ps_local.sh test
```

### 3.2 Distributed
```
# on scheduler node 
sh run_ps_dist_scheduler.sh

# on server node
sh run_ps_dist_server.sh

# on worker node
sh run_ps_dist_worker.sh
```


## 4. Acknowledge and Reference
- Referring the design from [xflow](https://github.com/xswang/xflow). 
