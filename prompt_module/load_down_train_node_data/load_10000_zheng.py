import random
import numpy as np

random.seed(16)


trainset = []
valset = []
testset = []


# 设置交叉验证
link_DPPS = {}
notLink_DPPS = {}


link_DPPS["drug_chemical"] = [i for i in range(0,10000)]
link_DPPS["drug_sideeffects"] = [i for i in range(20000,30000)]
link_DPPS["drug_substituent"] = [i for i in range(40000,50000)]
link_DPPS["drug_target"] = [i for i in range(60000,70000)]
link_DPPS["target_go"] = [i for i in range(80000,90000)]

notLink_DPPS["drug_chemical"] = [i for i in range(10000,20000)]
notLink_DPPS["drug_sideeffects"] = [i for i in range(30000,40000)]
notLink_DPPS["drug_substituent"] = [i for i in range(50000,60000)]
notLink_DPPS["drug_target"] = [i for i in range(70000,80000)]
notLink_DPPS["target_go"] = [i for i in range(90000,100000)]



link_DPPS["all"] = link_DPPS["drug_chemical"] + link_DPPS["drug_sideeffects"] + link_DPPS["drug_substituent"] + link_DPPS["drug_target"] + link_DPPS["target_go"]
notLink_DPPS["all"] = notLink_DPPS["drug_chemical"] + notLink_DPPS["drug_sideeffects"] + notLink_DPPS["drug_substituent"] + notLink_DPPS["drug_target"] + notLink_DPPS["target_go"]

random.shuffle(link_DPPS["all"])
random.shuffle(notLink_DPPS["all"])

pice = len(link_DPPS["all"])//10

for i in range(10):
        trainset.append(random.sample([link_DPPS["all"][pice * ((i+1)%10)]] + [notLink_DPPS["all"][
                                                                             pice * ((i+1)%10)]],2))
        valset.append(random.sample([link_DPPS["all"][pice * ((i+1)%10)+1]] + [notLink_DPPS["all"][
                                                                               pice * ((i+1)%10)+1]],2))
        testset.append(random.sample(link_DPPS["all"][pice*i:pice*(i+1)] + notLink_DPPS["all"][pice*i:pice*(i+1)],pice*2))



np.save('../node_data/zheng/10000_node/1-shot/all/all_train.npy', trainset)  # 保存文件
np.save('../node_data/zheng/10000_node/1-shot/all/all_val.npy', valset)  # 保存文件
np.save('../node_data/zheng/10000_node/1-shot/all/all_test.npy', testset)  # 保存文件



'''

'''