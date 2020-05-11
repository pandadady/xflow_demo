## 1. Introduction

Distributed LR With Parameter Server 

## 2. Install
        
### 2.1 build ps-lite
```
        cd ps-lite
        make -j
```
### 2.2 build xflow_demo
```
        # Back to current dir.
        cd ..
        mkdir build
        cd build 
        cmake ..
        make -j
``` 

## 3.Run
### 3.1 Local 
```
sh run_ps_local.sh
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



## 5. AD

欢迎关注公众号：**机器学习爱好者**

或者扫码：

![avatar](img/gongzhonghao.jpg)

学习更多的机器学习源码相关内容
