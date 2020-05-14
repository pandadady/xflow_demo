/*
 * ftrl.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SRC_OPTIMIZER_FTRL_H_
#define SRC_OPTIMIZER_FTRL_H_

#include <vector>
#include "src/base/base.h"

namespace xflow {
int w_dim = 1;
int v_dim = 10;
float alpha = 5e-2;
float beta = 1.0;
float lambda1 = 5e-5;
float lambda2 = 10.0;

class FTRL {
 public:
    FTRL() {}
    ~FTRL() {}
    static bool file_exists(const std::string &fpath) {
        std::fstream file;
        file.open(fpath.c_str(), std::ios::in);
        if (!file) {
            file.close();
            return false;
        }
        file.close();
        return true;
    }
    typedef struct FTRLEntry_w {
        FTRLEntry_w(int k = w_dim) {
            w.resize(k, 0.0);
            n.resize(k, 0.0);
            z.resize(k, 0.0);
        }
        std::vector<float> w;
        std::vector<float> n;
        std::vector<float> z;
    } ftrlentry_w;

    struct KVServerFTRLHandle_w {
        void operator()(const ps::KVMeta& req_meta,
                const ps::KVPairs<float>& req_data,
                ps::KVServer<float>* server) {
            size_t keys_size = req_data.keys.size();
            size_t vals_size = req_data.vals.size();
            ps::KVPairs<float> res;
            std::string model_path = "./model/model.all";
            int num = 0 ;
            if (req_meta.cmd == 110 && file_exists(model_path)&& store.size() == 0){
                //load/////////////////////////////////////////////////////////////////////////////////////////////////
                std::ifstream fin(model_path);
                std::string line;
                while (getline(fin, line)) {
                    std::vector<std::string> items1;
                    boost::split(items1, line, boost::is_any_of("\t"));
                    //std::cout<<line<<std::endl;
                    if (items1.size() != 3) {
                        std::cout<<"error" <<items1.size()<<std::endl;
                        continue;
                    }
                    std::vector<std::string> items2;
                    boost::split(items2, items1[2], boost::is_any_of(","));
                    //std::cout<<line<<" items2 " <<items2.size()<<std::endl;
                    FTRLEntry_w val;
                    val.w[0] = atof(items1[1].c_str());
                    //strtoull (love.c_str(), NULL, 0);
                    store.insert( std::make_pair(strtoull(items1[0].c_str(), NULL, 0), val ));
                    //std::cout <<"fid w "<<strtoull(items1[0].c_str(), NULL, 0)<<std::endl;
                    //std::cout <<"store w "<< store.size()<<std::endl;
                    num++;
                }
                std::cout <<"load success "<<num<<" cmd "<< req_meta.cmd << " KVServerFTRLHandle_w " << store.size()  <<std::endl;
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
            }
            if (req_meta.push) {
                w_dim = vals_size / keys_size;
                CHECK_EQ(keys_size, vals_size / w_dim);
            } else {
                res.keys = req_data.keys;
                res.vals.resize(keys_size * w_dim);
            }
            //std::cout << "KVServerFTRLHandle_w " << server->store_w.size()  <<std::endl;
            for (size_t i = 0; i < keys_size; ++i) {
                ps::Key key = req_data.keys[i];
                FTRLEntry_w& val = store[key];
                for (int j = 0; j < w_dim; ++j) {
                    if (req_meta.push) {
                        float g = req_data.vals[i * w_dim + j];
                        float old_n = val.n[j];
                        float n = old_n + g * g;
                        val.z[j] += g - (std::sqrt(n) - std::sqrt(old_n)) / alpha * val.w[j];
                        val.n[j] = n;
                        if (std::abs(val.z[j]) <= lambda1) {
                            val.w[j] = 0.0;
                        } else {
                            float tmpr = 0.0;
                            if (val.z[j] > 0.0) tmpr = val.z[j] - lambda1;
                            if (val.z[j] < 0.0) tmpr = val.z[j] + lambda1;
                            float tmpl = -1 * ((beta + std::sqrt(val.n[j]))/alpha    + lambda2);
                            val.w[j] = tmpr / tmpl;
                        }
                    } else {
                        res.vals[i * w_dim + j] = val.w[j];
                    }
                }
            }
            server->Response(req_meta, res);
            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            //std::cout <<"cmd "<< req_meta.cmd << " KVServerFTRLHandle_w " << store.size()  <<std::endl;


        }
    private:
        std::unordered_map<ps::Key, ftrlentry_w> store;
    };

    typedef struct FTRLEntry_v {
        FTRLEntry_v(int k = v_dim) {
            w.resize(k, 0.0);
            n.resize(k, 0.0);
            z.resize(k, 0.0);
        }
        std::vector<float> w;
        std::vector<float> n;
        std::vector<float> z;
    } ftrlentry_v;

    struct KVServerFTRLHandle_v {
        void operator()(const ps::KVMeta& req_meta,
                const ps::KVPairs<float>& req_data,
                ps::KVServer<float>* server) {
            size_t keys_size = req_data.keys.size();
            ps::KVPairs<float> res;
            std::string model_path = "./model/model.all";
            int num = 0 ;
            std::vector<ps::Key> fids;
            if (req_meta.cmd == 110 && file_exists(model_path) && store.size()==0){
                //load/////////////////////////////////////////////////////////////////////////////////////////////////
                std::ifstream fin(model_path);
                std::string line;
                while (getline(fin, line)) {
                    std::vector<std::string> items1;
                    boost::split(items1, line, boost::is_any_of("\t"));
                    //std::cout<<line<<std::endl;
                    if (items1.size() != 3) {
                        std::cout<<"error" <<items1.size()<<std::endl;
                        continue;
                    }
                    std::vector<std::string> items2;
                    boost::split(items2, items1[2], boost::is_any_of(","));
                    FTRLEntry_v val;
                    for (int j=0; j < v_dim; j++){
                         val.w[j] = atof(items2[j].c_str());
                    }
                    num++;
                    store.insert( std::make_pair( strtoull(items1[0].c_str(), NULL, 0), val ));
                    //std::cout <<"fid v "<< strtoull(items1[0].c_str(), NULL, 0)<<std::endl;
                    //std::cout <<"store v "<< store.size()<<std::endl;
                }
                std::cout <<"load success "<< num <<" cmd "<< req_meta.cmd << " KVServerFTRLHandle_v " << store.size()  <<std::endl;
                ///////////////////////////////////////////////////////////////////////////////////////////////////////
            }

            if (req_meta.push) {
                size_t vals_size = req_data.vals.size();
                CHECK_EQ(keys_size, vals_size / v_dim);
            } else {
                res.keys = req_data.keys;
                res.vals.resize(keys_size * v_dim);
            }
            for (size_t i = 0; i < keys_size; ++i) {
                ps::Key key = req_data.keys[i];
                if (store.find(key) == store.end()) {
                    FTRLEntry_v val(v_dim);;
                    for (int k = 0; k < v_dim; ++k) {
                        val.w[k] = Base::local_normal_real_distribution<double>(0.0, 1.0)(Base::local_random_engine()) * 1e-2;
                    }
                    store[key] = val;
                }

                FTRLEntry_v& val = store[key];

                for (int j = 0; j < v_dim; ++j) {
                    if (req_meta.push) {
                        float g = req_data.vals[i * v_dim + j];
                        float old_n = val.n[j];
                        float n = old_n + g * g;
                        val.z[j] += g - (std::sqrt(n) - std::sqrt(old_n)) / alpha * val.w[j];
                        val.n[j] = n;

                        if (std::abs(val.z[j]) <= lambda1) {
                            val.w[j] = 0.0;
                        } else {
                            float tmpr = 0.0;
                            if (val.z[j] > 0.0) tmpr = val.z[j] - lambda1;
                            if (val.z[j] < 0.0) tmpr = val.z[j] + lambda1;
                            float tmpl = -1 * ((beta + std::sqrt(val.n[j]))/alpha    + lambda2);
                            val.w[j] = tmpr / tmpl;
                        }
                    } else {
                        res.vals[i * v_dim + j] = val.w[j];
                    }
                }
            }
            server->Response(req_meta, res);

            //std::cout <<"cmd "<< req_meta.cmd << " KVServerFTRLHandle_v " << store.size()  <<std::endl;

        }
    private:
        std::unordered_map<ps::Key, ftrlentry_v> store;
    };

 private:
};
}    // namespace xflow

#endif    // SRC_OPTIMIZER_FTRL_H_
