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

    Server(int workernum ,std::string modelname) {
        server_w_ = new ps::KVServer<float>(0);
        server_v_ = new ps::KVServer<float>(1);
        FTRL::KVServerFTRLHandle_w w_h;;
        w_h.dump_start_count = 0;
        w_h.load_start_count = 0;
        w_h.workernum = workernum;
        FTRL::KVServerFTRLHandle_v v_h;;
        v_h.dump_start_count = 0;
        v_h.load_start_count = 0;
        v_h.workernum = workernum;
        server_w_->set_request_handle(w_h);
        //server_w_->set_request_handle(SGD::KVServerSGDHandle_w());
        server_v_->set_request_handle(v_h);
        //server_v_->set_request_handle(SGD::KVServerSGDHandle_v());

        std::cout << "init server success " << std::endl;
    }
    ~Server() {
        delete server_w_;
        delete server_v_;
    }



    ps::KVServer<float>* server_w_;
    ps::KVServer<float>* server_v_;

};
}    // namespace xflow
#endif    // SRC_MODEL_SERVER_H_
