/*
 * fm_worker.cc
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <time.h>
#include <unistd.h>
#include <immintrin.h>

#include <algorithm>
#include <ctime>
#include <iostream>

#include <mutex>
#include <functional>
#include <random>
#include <string>
#include <memory>

#include "src/model/fm/fm_worker.h"

namespace xflow {

void FMWorker::calculate_pctr(int start, int end) {
    auto all_keys = std::vector<Base::sample_key>();
    auto unique_keys = std::vector<ps::Key>();
    int line_num = 0;
    for (int row = start; row < end; ++row) {
        int sample_size = test_data->fea_matrix[row].size();
        Base::sample_key sk;
        sk.sid = line_num;
        for (int j = 0; j < sample_size; ++j) {
            size_t idx = test_data->fea_matrix[row][j].fid;
            sk.fid = idx;
            all_keys.push_back(sk);
            (unique_keys).push_back(idx);
        }
        ++line_num;
    }
    std::sort(all_keys.begin(), all_keys.end(), base_->sort_finder);
    std::sort((unique_keys).begin(), (unique_keys).end());
    (unique_keys).erase(unique((unique_keys).begin(), (unique_keys).end()), unique_keys.end());

    auto w = std::vector<float>();
    kv_w->Wait(kv_w->Pull(unique_keys, &w));
    auto v = std::vector<float>();
    kv_v->Wait(kv_v->Pull(unique_keys, &v));


    auto wx = std::vector<float>(line_num);
    for (int j = 0, i = 0; j < all_keys.size(); ) {
        size_t allkeys_fid = all_keys[j].fid;
        size_t weight_fid = (unique_keys)[i];
        if (allkeys_fid == weight_fid) {
            wx[all_keys[j].sid] += w[i];
            ++j;
        } else if (allkeys_fid > weight_fid) {
            ++i;
        }
    }

    auto v_sum = std::vector<float>(end - start);
    auto v_pow_sum = std::vector<float>(end - start);
    for (size_t k = 0; k < v_dim_; ++k) {
        for (size_t j = 0, i = 0; j < all_keys.size(); ) {
            size_t allkeys_fid = all_keys[j].fid;
            size_t weight_fid = unique_keys[i];
            if (allkeys_fid == weight_fid) {
                size_t sid = all_keys[j].sid;
                float v_weight = v[i * v_dim_ + k];
                v_sum[sid] += v_weight;
                v_pow_sum[sid] += v_weight * v_weight;
                ++j;
            } else if (allkeys_fid > weight_fid) {
                ++i;
            }
        }
    }
    auto v_y = std::vector<float>(end - start);
    for (size_t i = 0; i < end - start; ++i) {
        v_y[i] = v_sum[i] * v_sum[i] - v_pow_sum[i];
    }

    for (int i = 0; i < wx.size(); ++i) {
        float pctr = base_->sigmoid(wx[i] + v_y[i]);
        Base::auc_key ak;
        ak.label = test_data->label[start++];
        ak.pctr = pctr;
        mutex.lock();
        test_auc_vec.push_back(ak);
        md << pctr << "\t" << 1 - ak.label << "\t" << ak.label << std::endl;
        mutex.unlock();
    }
    --calculate_pctr_thread_finish_num;
}

void FMWorker::predict(ThreadPool* pool, int rank, int block) {
    char buffer[1024];
    snprintf(buffer, 1024, "%d_%d", rank, block);
    std::string filename = buffer;
    md.open("pred_" + filename + ".txt");
    if (!md.is_open()) std::cout << "open pred file failure!" << std::endl;


    snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
    size_t load_size = block_size << 20;
    xflow::LoadData test_data_loader(test_data_path, load_size);
    test_data = &(test_data_loader.m_data);
    //std::cout <<"predict load_size " << load_size << std::endl;
    std::cout <<"Worker No. is = " << ps::MyRank()<< " test " << std::endl;
    test_auc_vec.clear();
    int count = 0;
    while (true) {
        count++;
        test_data_loader.load_minibatch_hash_data_fread();
        if (test_data->fea_matrix.size() <= 0) break;
        std::cout <<"Worker No. is = " << ps::MyRank() <<" " <<test_data->fea_matrix.size() << " "<< count <<std::endl;
        int thread_size = test_data->fea_matrix.size() / core_num;
        int start = 0;
        int end = 0;
        calculate_pctr_thread_finish_num = core_num + 1 ;
        for (int i = 0; i < core_num; ++i) {
            start = i * thread_size;
            end = (i + 1)* thread_size;
            pool->enqueue(std::bind(&FMWorker::calculate_pctr, this, start, end));
            //std::cout <<start<<" " << end << std::endl;
        }
        if (test_data->fea_matrix.size() > end){
            start = end;
            end = test_data->fea_matrix.size();
            pool->enqueue(std::bind(&FMWorker::calculate_pctr, this, start, end));
        }
        while (calculate_pctr_thread_finish_num > 0) usleep(5);

    }
    md.close();

    test_data = NULL;
    base_->calculate_auc(test_auc_vec);
    base_->auc(test_auc_vec);

}

void FMWorker::calculate_gradient(std::vector<Base::sample_key>& all_keys,
        std::vector<ps::Key>& unique_keys,
        size_t start, size_t end,
        std::vector<float>& v,
        std::vector<float>& v_sum,
        std::vector<float>& loss,
        std::vector<float>& push_w_gradient,
        std::vector<float>& push_v_gradient) {
    for (size_t k = 0; k < v_dim_; ++k) {
        for (int j = 0, i = 0; j < all_keys.size(); ) {
            size_t allkeys_fid = all_keys[j].fid;
            size_t weight_fid = unique_keys[i];
            int sid = all_keys[j].sid;
            if (allkeys_fid == weight_fid) {
                (push_w_gradient)[i] += loss[sid];
                push_v_gradient[i * v_dim_ + k] += loss[sid] * (v_sum[sid] - v[i * v_dim_ + k]);
                ++j;
            } else if (allkeys_fid > weight_fid) {
                ++i;
            }
        }
    }

    size_t line_num = end - start;
    for (size_t i = 0; i < (push_w_gradient).size(); ++i) {
        (push_w_gradient)[i] /= 1.0 * line_num;
    }
    for (size_t i = 0; i < push_v_gradient.size(); ++i) {
        push_v_gradient[i] /= 1.0 * line_num;
    }
}

void FMWorker::calculate_loss(std::vector<float>& w,
        std::vector<float>& v,
        std::vector<Base::sample_key>& all_keys,
        std::vector<ps::Key>& unique_keys,
        size_t start, size_t end,
        std::vector<float>& v_sum,
        std::vector<float>& loss) {
    auto wx = std::vector<float>(end - start);
    for (int j = 0, i = 0; j < all_keys.size(); ) {
        size_t allkeys_fid = all_keys[j].fid;
        size_t weight_fid = (unique_keys)[i];
        if (allkeys_fid == weight_fid) {
            wx[all_keys[j].sid] += (w)[i];
            ++j;
        } else if (allkeys_fid > weight_fid) {
            ++i;
        }
    }
    auto v_pow_sum = std::vector<float>(end - start);
    for (size_t k = 0; k < v_dim_; k++) {
        for (size_t j = 0, i = 0; j < all_keys.size(); ) {
            size_t allkeys_fid = all_keys[j].fid;
            size_t weight_fid = unique_keys[i];
            if (allkeys_fid == weight_fid) {
                size_t sid = all_keys[j].sid;
                float v_weight = v[i * v_dim_ + k];
                v_sum[sid] += v_weight;
                v_pow_sum[sid] += v_weight * v_weight;
                ++j;
            } else if (allkeys_fid > weight_fid) {
                ++i;
            }
        }
    }
    auto v_y = std::vector<float>(end - start);
    for (size_t i = 0; i < end - start; ++i) {
        v_y[i] = v_sum[i] * v_sum[i] - v_pow_sum[i];
    }

    for (int i = 0; i < wx.size(); i++) {
        float pctr = base_->sigmoid(wx[i] + v_y[i]);
        loss[i] = pctr - train_data->label[start++];
    }
}

//void FMWorker::dump_w_v(){
//    mld.open("model/model." + modelname+"."+std::to_string(ps::MyRank()));
//    if (!mld.is_open()) std::cout << "open model file failure!" << std::endl;
//    for(auto&item : store_w){
//        mld << item.first << "\t" <<item.second << "\t";
//        for (auto& val:store_v[item.first]){
//            mld << val <<",";
//        }
//        mld << std::endl;
//    }
//    mld.close();
//}
//
//void FMWorker::save_w_v( std::vector<ps::Key>& unique_keys, std::vector<float>& w, std::vector<float>& v){
//    //std::cout<<"fid "<< unique_keys.size() <<" w "<< w.size() <<" v "<< v.size() << std::endl ;
//    for (size_t i = 0; i < unique_keys.size(); i++) {
//        size_t fid = (unique_keys)[i];
//        store_w[fid] = w[i];
//        std::vector<float> v_weight(v_dim_);
//        for(size_t j = 0 ; j < v_dim_; j++){
//            v_weight[j] = v[i * v_dim_ + j];
//         }
//         store_v[fid] = v_weight;
//    }
//    //std::cout<<"store w "<< store_w.size() <<" v "<< store_v.size() << std::endl;
//}


void FMWorker::update(int start, int end) {
    size_t idx = 0;
    auto all_keys = std::vector<Base::sample_key>();
    auto unique_keys = std::vector<ps::Key>();
    int line_num = 0;
    for (int row = start; row < end; ++row) {
        int sample_size = train_data->fea_matrix[row].size();
        Base::sample_key sk;
        sk.sid = line_num;
        for (int j = 0; j < sample_size; ++j) {
            idx = train_data->fea_matrix[row][j].fid;
            sk.fid = idx;
            all_keys.push_back(sk);
            (unique_keys).push_back(idx);
        }
        ++line_num;
    }
    std::sort(all_keys.begin(), all_keys.end(), base_->sort_finder);
    std::sort((unique_keys).begin(), (unique_keys).end());
    (unique_keys).erase(unique((unique_keys).begin(), (unique_keys).end()), (unique_keys).end());
    int keys_size = (unique_keys).size();

    auto w = std::vector<float>();
    kv_w->Wait(kv_w->Pull(unique_keys, &w));
    auto push_w_gradient = std::vector<float>(keys_size);
    auto v = std::vector<float>();
    kv_v->Wait(kv_v->Pull(unique_keys, &v));
    auto push_v_gradient = std::vector<float>(keys_size * v_dim_);

    auto loss = std::vector<float>(end - start);
    auto v_sum = std::vector<float>(end - start);
    calculate_loss(w, v, all_keys, unique_keys, start, end, v_sum, loss);
    calculate_gradient(all_keys, unique_keys, start, end, v, v_sum, loss, push_w_gradient, push_v_gradient);

    kv_w->Wait(kv_w->Push(unique_keys, push_w_gradient));
    kv_v->Wait(kv_v->Push(unique_keys, push_v_gradient));
//    mutex.lock();
//    w.clear();
//    v.clear();
//    kv_w->Wait(kv_w->Pull(unique_keys, &w));
//    kv_v->Wait(kv_v->Pull(unique_keys, &v));
//    save_w_v(unique_keys, w, v);
//    mutex.unlock();
    --gradient_thread_finish_num;
}

void FMWorker::batch_training(ThreadPool* pool) {
    std::vector<ps::Key> key;
    std::vector<float> val_w;
    std::vector<float> val_v;

    key.push_back(0);
    val_w.push_back(0);
    val_v.assign(v_dim_, 0);

    std::cout<<"Push success 110 " << ps::MyRank()<<std::endl;
    kv_w->Wait(kv_w->Pull(key, &val_w, nullptr, 110, nullptr));
    kv_v->Wait(kv_v->Pull(key, &val_v, nullptr, 110, nullptr));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        size_t load_size = block_size << 20;
        xflow::LoadData train_data_loader(train_data_path, load_size);
        train_data = &(train_data_loader.m_data);
        std::cout <<"Worker No. is = " << ps::MyRank()<< " train_epoch "<< epoch << std::endl;
        int block = 0;
        int count = 0;
        while (1) {
            count ++;
            train_data_loader.load_minibatch_hash_data_fread();
            if (train_data->fea_matrix.size() <= 0) break;
            std::cout <<"Worker No. is = " << ps::MyRank() <<" " <<train_data->fea_matrix.size() << " "<< count<<" "<< epoch <<std::endl;
            int thread_size = train_data->fea_matrix.size() / core_num;
            gradient_thread_finish_num = core_num + 1;
            int start = 0;
            int end = 0;

            for (int i = 0; i < core_num; ++i) {
                start = i * thread_size;
                end = (i + 1)* thread_size;
                pool->enqueue(std::bind(&FMWorker::update, this, start, end));
            }
            if (train_data->fea_matrix.size() > end){
                start = end;
                end = train_data->fea_matrix.size();
                //std::cout <<"Worker No. is = " << ps::MyRank() <<" start " << start <<" end " << end << std::endl;
                pool->enqueue(std::bind(&FMWorker::update, this, start, end));

            }
            while (gradient_thread_finish_num > 0) {
                usleep(5);
            }
            ++block;
        }
        if ((epoch + 1) % 30 == 0) std::cout << "epoch : " << (epoch+1) << std::endl;
        train_data = NULL;
    }
    key.clear();
    val_w.clear();
    val_v.clear();
    key.push_back(0);
    val_w.push_back(0);
    val_v.assign(v_dim_, 0);
    kv_w->Wait(kv_w->Pull(key, &val_w, nullptr, 119, nullptr));
    kv_v->Wait(kv_v->Pull(key, &val_v, nullptr, 119, nullptr));
    std::cout<<"Push success 119 " << ps::MyRank()<<std::endl;
//    dump_w_v();
}

void FMWorker::train() {
    rank = ps::MyRank();
    std::cout << "Worker No. is = " << rank << std::endl;
    snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
    batch_training(pool_);
    if (rank == 0) {
        std::cout << "FM AUC: " << std::endl;
        predict(pool_, rank, 0);
    }
    std::cout << "Worker " << rank << " train end......" << std::endl;
}
}    // namespace xflow
