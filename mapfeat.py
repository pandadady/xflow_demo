#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
id
click
hour
C1
banner_pos
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C14
C15
C16
C17
C18
C19
C20
C21
uint64_t Hash::StdHash(const string &value) {
    return hash<string>()(value);
}
uint64_t Hash::FarmHash(int64_t value) {
    string data = to_string(value);
    return FarmHash(data);
}

uint64_t FeatureProcess::GetFValue(uint64_t value) {
    uint64_t FLAG = 0x003fffffffffffff;
    return FLAG & value;
}


uint64_t Hash::FarmHash(const string value) {
//    time_t timep;
//    time (&timep);
//    char tmp[64];
//    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S",localtime(&timep) );
//    cout << "DEBUG hash time: " << tmp << "\n" << flush;
//    return util::Hash64WithSeed(value, 64);
//    return CityHash(value);
    return StdHash(value);
}
uint64_t FeatureProcess::GetFid(uint64_t slotId, uint64_t value) {
    return (slotId << 54) | GetFValue(value);
}

uint64_t FeatureProcess::GetFidHashUint64(uint64_t slotId, string value) {
    return GetFid(slotId, Hash::FarmHash(value));
}

uint64_t FeatureProcess::GetFidHashUint64(uint64_t slotId, int64_t value) {
    return GetFid(slotId, Hash::FarmHash(value));
}
"""
import os
import sys
import random
def StdHash(value):
    return hash(str(value));

def GetFValue(value) :
    FLAG = 0x003fffffffffffff;
    return FLAG & value;

def GetFid(slotId, value):
    return (slotId << 54) | GetFValue(value)

def GetFidHashUint64(slotId, value):
    return GetFid(slotId, StdHash(value))

"""
split -l 1000000 -d  train split_data/t_ 
"""
def trans(filename, targetname):
    fo = open(targetname, 'w' )
    fmap = {}
    num = 0
    for l in open(filename):
        l = l.strip()
        arr = l.split(',')
        items = []
        label = "0"
        if 'hour' in arr:
            continue
        slotid = 0
        for i in range(len(arr)):
            if i==1:
                label = arr[i]
                continue
            if i==0:
                slotid=1
            if i>=2:
                slotid=i
            fid = str(GetFidHashUint64(i+1, arr[i]))
            #fid = arr[i]
            if fid=="-1":
                continue
            weight = "0.5"
            slotid = str(slotid)
            items.append([slotid, fid, weight])
        fo.write(label+'\t'+" ".join([x[0]+":"+x[1]+":"+x[2] for x in items])+'\n')
        num+=1;
        if num==1000:break
    fo.close()

def main(workernum):
    fpath = '../ctr_data/split_data/'
    flist_train_selects = []
    flist_test_selects = []
    record = {}
    if os.path.exists("./data/train_list"):
        with open("./data/train_list") as f :
            for line in f:
                line = line.strip()
                terms = line.split()
                if len(terms)!=2:continue
                filen = terms[0]
                used = terms[1]
                if used  == "un" and len(flist_train_selects)!= workernum:
                    flist_train_selects.append(filen);
                    used = "train"
                elif used  == "un" and len(flist_test_selects)!=1:
                    flist_test_selects.append(filen);
                    used = "test"
                record[filen] = used
    else:
        flist = os.listdir(fpath)
        for filen in flist:
            used="un"
            if len(flist_train_selects) != workernum:
                flist_train_selects.append(filen);
                used = "train"
            elif len(flist_test_selects) != 1:
                flist_test_selects.append(filen);
                used = "test"
            record[filen] = used
    f = open("./data/train_list", "w")
    for filen in record:
        f.write(filen + '\t' + record[filen] + '\n')
    f.close()
    print(flist_train_selects)
    print(flist_test_selects)
    if len(flist_train_selects) < workernum or len(flist_test_selects) !=1:
        print("data not enough!!!!!")
        return
    for i in range(len(flist_train_selects)):
        filepath = fpath + flist_train_selects[i]
        target = 'data/train.libsvm-0000'+str(i)
        trans(filepath, target)

    filepath = fpath + flist_test_selects[0]
    target = 'data/test.libsvm-0000' + str(0)
    trans(filepath, target)
if __name__ == '__main__':
    workernum = int(sys.argv[1:][0]) + 1
    main(workernum)
