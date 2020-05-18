/*
 * base.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "src/model/lr/lr_worker.h"
#include "src/model/fm/fm_worker.h"
#include "src/model/mvm/mvm_worker.h"
#include "src/model/server.h"

#include "ps/ps.h"

int main(int argc, char *argv[]) {
    if (ps::IsScheduler()) {
        std::cout << "start scheduler" << std::endl;
    }
    xflow::Server* server = NULL;
    if (ps::IsServer()) {
        std::cout << "start server" << std::endl;
        server = new xflow::Server();
        std::string modelname = argv[5];
        int workernum = std::atoi(argv[6]);
    }

    ps::Start();
    if (ps::IsWorker()) {
        //std::cout << "start worker" << std::endl;
        int epochs = std::atoi(argv[4]);
        std::string modelname = argv[5];
        std::cout << modelname << std::endl;
//        if (*(argv[3]) == '0') {
//            std::cout << "start LR " << std::endl;
//            xflow::LRWorker* lr_worker = new xflow::LRWorker(argv[1], argv[2]);
//            lr_worker->epochs = epochs;
//            lr_worker->train();
//        }
        if (*(argv[3]) == '1') {
            //std::cout << "start FM " << std::endl;
            xflow::FMWorker* fm_worker = new xflow::FMWorker(argv[1], argv[2]);
            fm_worker->epochs = epochs;
            fm_worker->modelname = modelname;
            fm_worker->train();
        }
//        if (*(argv[3]) == '2') {
//            std::cout<< "start MVM " << std::endl;
//            xflow::MVMWorker* mvm_worker = new xflow::MVMWorker(argv[1], argv[2]);
//            mvm_worker->epochs = epochs;
//            mvm_worker->train();
//        }
    }


    ps::Finalize();

}
