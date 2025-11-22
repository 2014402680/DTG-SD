import random
import numpy as np

random.seed(42)


trainset = []
valset = []
testset = []


# 设置交叉验证
link_DPPS = {}
notLink_DPPS = {}


link_DPPS["drug_protein"] = [i for i in range(0,333)]
link_DPPS["drug_disease"] = [i for i in range(666,999)]
link_DPPS["drug_se"] = [i for i in range(1332,1665)]
link_DPPS["protein_disease"] = [i for i in range(1998,2331)]

notLink_DPPS["drug_protein"] = [i for i in range(333,666)]
notLink_DPPS["drug_disease"] = [i for i in range(999,1332)]
notLink_DPPS["drug_se"] = [i for i in range(1665,1998)]
notLink_DPPS["protein_disease"] = [i for i in range(2331,2664)]

link_DPPS["all"] = link_DPPS["drug_protein"] + link_DPPS["drug_disease"] + link_DPPS["drug_se"] + link_DPPS["protein_disease"]
notLink_DPPS["all"] = notLink_DPPS["drug_protein"] + notLink_DPPS["drug_disease"] + notLink_DPPS["drug_se"] + notLink_DPPS["protein_disease"]

random.shuffle(link_DPPS["all"])
random.shuffle(notLink_DPPS["all"])
'''
pice = len(link_DPPS["protein_disease"])//10

for i in range(10):
        trainset.append(random.sample(link_DPPS["protein_disease"][pice * ((i+1)%10):pice * ((i+1)%10)+10] + notLink_DPPS["protein_disease"][
                                                                             pice * ((i+1)%10):pice * ((i+1)%10)+10],20))
        valset.append(random.sample(link_DPPS["protein_disease"][pice * ((i+1)%10)+10:pice * ((i+1)%10)+20] + notLink_DPPS["protein_disease"][
                                                                               pice * ((i+1)%10)+10:pice * ((i+1)%10)+20],20))
        testset.append(random.sample(link_DPPS["protein_disease"][pice*i:pice*(i+1)] + notLink_DPPS["protein_disease"][pice*i:pice*(i+1)],pice*2))
'''
pice = len(link_DPPS["all"])//10
a = [link_DPPS["all"][i:i+pice] for i in range(0, len(link_DPPS["all"]), pice)]
b = [notLink_DPPS["all"][i:i+pice] for i in range(0, len(notLink_DPPS["all"]), pice)]

for i in range(10):
        trainset.append(a[((i)%10)][:pice*0.05] + b[((i)%10)][:pice*0.05])
        #valset.append(link_DPPS["drug_chemical"][pice * ((i+1)%10)+5:pice * ((i+1)%10)+10] + notLink_DPPS["drug_chemical"][pice * ((i+1)%10)+5:pice * ((i+1)%10)+10])
        valset.append(a[((i+1)%10)][:pice*0.05] + b[((i+1)%10)][:pice*0.05])
        testset.append(a[((i+2)%10)] + b[((i+2)%10)])


np.save('../node_data/luo/allte/20-shot/all/all_train.npy', trainset)  # 保存文件
np.save('../node_data/luo/allte/20-shot/all/all_val.npy', valset)  # 保存文件
np.save('../node_data/luo/allte/20-shot/all/all_test.npy', testset)  # 保存文件


