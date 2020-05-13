/*
 * server.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SRC_MODEL_SERVER_H_
#define SRC_MODEL_SERVER_H_

#include <time.h>
#include <string>
#include <iostream>

#include "ps/ps.h"
#include "src/optimizer/ftrl.h"
#include "src/optimizer/sgd.h"

namespace xflow {
class Server {
 public:
    static int current_realtime_s() {
        struct timespec tp;
        clock_gettime(CLOCK_REALTIME, &tp);
        return tp.tv_sec ;
    }
    Server() {
        server_w_ = new ps::KVServer<float>(0);
        server_w_->set_request_handle(FTRL::KVServerFTRLHandle_w());
        //server_w_->set_request_handle(SGD::KVServerSGDHandle_w());

        server_v_ = new ps::KVServer<float>(1);
        server_v_->set_request_handle(FTRL::KVServerFTRLHandle_v());
        //server_v_->set_request_handle(SGD::KVServerSGDHandle_v());
        std::cout << "init server success " << std::endl;
    }
    ~Server() {
    }


//    void dump_model(){
//    //std::unordered_map<ps::Key, ftrlentry_w> store;
//    //std::unordered_map<ps::Key, ftrlentry_v> store;
//        FTRL::KVServerFTRLHandle_w  Handle_w = (FTRL::KVServerFTRLHandle_w)server_w_->request_handle_;
//        FTRL::KVServerFTRLHandle_v  Handle_v = (FTRL::KVServerFTRLHandle_v)server_v_->request_handle_;
//        std::ofstream mld;
//        mld.open("model_" + std::to_string(current_realtime_s()) + ".txt");
//        if (!mld.is_open()) std::cout << "open pred file failure!" << std::endl;
//        std::cout << "w " << Handle_w.store.size() << " v " << Handle_v.store.size()<<std::endl;
//        for(auto& item : Handle_w.store){
//            mld << item.first << "\t" <<item.second.w[0] << "\t";
//            for (auto& val:Handle_v.store[item.first].w){
//                mld << val <<",";
//            }
//            mld << std::endl;
//        }
//        mld.close();
//    }
    ps::KVServer<float>* server_w_;
    ps::KVServer<float>* server_v_;

};
}    // namespace xflow
#endif    // SRC_MODEL_SERVER_H_
