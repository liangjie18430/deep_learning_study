#! -*- encoding: utf-8 -*-
#from __future__ import unicode_literals
from __future__ import print_function
import sys
sys.path.append("../../data_write/redisupload")
import json
import logging
try:
    from configparser import ConfigParser
except:
    import ConfigParser.ConfigParser as ConfigParser

import Monitor
import redis_client
try:
    reload(sys)
    sys.setdefaultencoding("utf-8")
except:
    pass
size_limit = 100000

def gene_redis_key(model_type,strategy_type,city_id,project_type,key_id):
    temp_key = "newh_recommend_%s_%s_%s_%s_%s"%(model_type,strategy_type,city_id,project_type,key_id)
    return temp_key

    pass
#数据较少，直接读取
def read_to_df(filename,conf):
    """
    使用yield进行trunk读
    :param filename:
    :param conf:
    :return:
    """

    model_type = conf.get("model_conf","model_type")
    strategy_type = conf.get("model_conf","strategy_type")
    insert_list = []
    #read_number = 0
    data_dict = {}
    with open(filename,'r') as f:
        #移动到上次读取的位置
        count=0
        #line_count=0
        #从移动到的当前行的下一行开始读取
        for line in f:
            #line_count=line_count+1
            result = line.rstrip('\n').split('\t')
            #result = json.loads(line)
            item_id = result[0]
            city_id = result[1]
            project_type = result[2]

            r_value  = result[3]

            r_key = gene_redis_key(model_type,strategy_type,city_id,project_type,item_id)

            data_dict[r_key] = r_value
            #代表对第几行操作完毕
            count = count + 1
            if count ==size_limit:
                #记录当前读取到的位置
                yield data_dict
                data_dict={}
                count = 0
    #最后执行完后，如果insert_list不是为空，那么需要返回
    if (len(data_dict)!=0):
        yield data_dict




def init_redis_conn(conf):
    r_host=conf.get("redis","host")
    r_port = conf.get("redis", "port")
    log_file = conf.get("log", "log_file")
    log_name = conf.get("log","log_name")
    #创建log对象
    print("[redis_host:%s,redis_port:%s]"%(r_host,r_port))
    mylog = Monitor.MyLog(log_file,log_name,logging.INFO,"")
    #创建redis_client对象
    r_client = redis_client.RedisClient(mylog,r_host,r_port,"","")
    r_client.connect()
    return r_client


    pass




def do_insert_redis(conf,path):
    rdc = init_redis_conn(conf)
    insert_gene = read_by_trunk(path,conf)
    i = 0
    for data_dict in insert_gene:
        rdc.pipeline_set(data_dict,604800)#604800s = 7day
        i=i+1
        print("%d insert times done ,the size is %d"%(i,len(data_dict)))
    print("insert done")

    pass

def usage():
    """
    usage info
    """
    print("python cf_redis_upload.py conf_file_path input_file_path")
#测试key:newh_recommend_userbased_120000_project_2000000020406754
#newh_recommend_userbased_120000_project_2000000020406754
if __name__ == '__main__':

    if len(sys.argv) != 3:
        usage()
        exit(1)

    conf_file = sys.argv[1]
    print("the conf_file is:%s" % (conf_file))
    input_file_path = sys.argv[2]
    print("the input_file_path is:%s"%(input_file_path))
    conf = ConfigParser()
    conf.read(conf_file)
    do_insert_redis(conf,input_file_path)
