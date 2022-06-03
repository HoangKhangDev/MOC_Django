from genericpath import exists
from importlib.resources import path
import math
import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import random
import time,datetime
import shutil
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,accuracy_score
import svmutil as svm

path_stock=os.path.dirname(os.path.abspath("app"))+os.sep

list_file_csv= os.listdir(path_stock+"data")


if os.path.exists(path_stock+f"tmp/best_nu_gamma.csv"):
    with open(path_stock+f"tmp/best_nu_gamma.csv") as f:
        tach=f.readline().split(" ")
        best_nu=tach[0]
        best_gamma=tach[1]


best_acc=0

def tranlate_file(batch=10,Age=3,path_file="",nu_start=0.1,nu_end=1,nu_step=.05,gamma_start=0,gamma_end=10,gamma_step=1,path_50=2):
    mang=[]
    tach=path_file.split(os.sep)
    namefile=tach[len(tach)-1]

    # for item in list_file_csv:
    namefile=namefile.replace(".csv",".data")
    f = open(path_file, "r")
    mang= f.readlines()

    if(os.path.exists(path_stock+f'tmp')==False):
        os.mkdir(path_stock+f'tmp')
        
    if(os.path.exists(path_stock+f'tmp/{namefile}')==False):
        f= open(path_stock+f'tmp/{namefile}','w') 
        f.write("")
    csvfile= open(path_stock+f'tmp/{namefile}','a')
    file= open(path_stock+f'tmp/{namefile}','w')
    file.write('')
    dem=0;
    for line in mang:
        # print(line)
        if(dem>0):
            tach= line.split(',')
            dataline=''
            tmp=0
            for i in range(0, len(tach)):
                if(i==11):
                    if(int(tach[i])==1):
                             dataline=f'1{dataline}\n'
                    else:
                             dataline=f'0{dataline}\n'
                else:
                    if(i>0):
                        if(float(tach[i])>0):
                            tmp+=1
                        else:
                            tmp+=0
                        dataline+=f' {i}:{tach[i]}'
            if(tmp==len(tach)-2):
                csvfile.writelines(dataline)
        dem=dem+1

    random_data(namefile)
    split_data(namefile)
    split_file_train_50(namefile.replace(".data",""),path_50)
    Find_best_nu_gamma(nu_start,nu_end,nu_step,gamma_start,gamma_end,gamma_step)
    Train_batch(batch)
    Train_By_Age(Age,batch)
    Test_File_30()

def random_data(namefile_open):
    # print(os.path.exists(path_stock+f'data/{namefile_save}.csv'))
    data_store=[]
    f = open(path_stock+f"tmp/{namefile_open}", "r")
    arr= f.readlines()
    arr= np.array(arr)
    random.shuffle(arr)
    if(os.path.exists(path_stock+f'tmp/{namefile_open}')==False):
        with open(path_stock+"tmp"+os.sep+f'/{namefile_open}',"w"):pass
    f= open(path_stock+f'tmp/{namefile_open}', 'w')
    for line in arr:
        f.writelines(line)
        data_store.append(line)
    return data_store


def split_data(namefile_open):
    f = open(path_stock+f"tmp/{namefile_open}", "r")
    name_train= namefile_open.replace(".data",".train")
    name_test=namefile_open.replace(".data",".test")
    mang= f.readlines()
    first_line= mang.pop(0)
    line_train= round(len(mang)*(70/100))
    line_test= round(len(mang)-line_train)

    f_train= open(path_stock+f"tmp/{name_train}", "w")
    f_test= open(path_stock+f"tmp/{name_test}", "w")

    f_train.write("")
    f_test.write("")

    f_train= open(path_stock+f"tmp/{name_train}", "a")
    f_test= open(path_stock+f"tmp/{name_test}", "a")

    dem=1;
    for line in mang:
        # print(line)
        if(dem<=line_train):
            if(dem==1):
                f_train.writelines(first_line)
            f_train.writelines(line)
        else:
            if(dem==(len(mang)-line_test+1)):
                f_test.writelines(first_line)
            f_test.writelines(line)
        dem+=1


def split_file_train_50(namefile_open,path_50=2):
    print('50 - path'+str(path_50))
    f = open(path_stock+f"tmp/{namefile_open}.train", "r")
    mang= f.readlines()

    if(os.path.exists('train')==False):
            os.mkdir('train')


    f_save_train= open(path_stock+f"train/{namefile_open}.train", "w")
    f_save_train.write("")
    f_save_train= open(path_stock+f"train/{namefile_open}.train", "a")


    f_save_test= open(path_stock+f"train/{namefile_open}.test", "w")
    f_save_test.write("")
    f_save_test= open(path_stock+f"train/{namefile_open}.test", "a")


    for j in range(1,(len(mang))):
        if j <= (len(mang)/path_50):
            f_save_test.writelines(mang[j])
        else:
            f_save_train.writelines(mang[j])
    fs = open(path_stock+'best_acc.txt','w')
    fs.write(str(path_50+1))


def GetFirstLabel(value):
        #Lay mot phan cua chuoi truoc ky tu space
        position = value.find(" ")
        return value[0:position]

def SplitLabel(ftrain):
        path=path_stock+'tmp/tmp50/data/'
        #Tao folder dua tren ten file train
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


        #Doc du lieu tu file train va gan vao mang
        lines=ftrain.readlines()
        #Tao mot mang trong chua danh sach cac label
        list_label=[]
        #Tach label cua dong dau tien trong file train va gan vao danh sach label
        list_label.append(GetFirstLabel(lines[0]))
        #Vong lap for den gan cac label con lai cua file train vao danh sach label
        for i in lines[1:len(lines)]:
            #Tach label cua dong thu i
            text=GetFirstLabel(i)
            #Kiem tra ton tai trong danh sach label
            if text not in list_label:
                #Neu chua co se them label do vao danh sach label
                list_label.append(text)
        #Tao mot dictionary chua cac lop dua theo label cua tung lop
        list_class=dict()
        #Gan cho moi label la mot mang trong trong dictionary
        for i in list_label:
            list_class[i]=[]
        #Vong lap for chay tung dong trong mang du lieu train
        for i in lines:
            #Tach nhan cua dong thu i
            text=GetFirstLabel(i)
            #Gan dong thu i vao dictionary theo nhan tuong ung
            list_class[text].append(i)
        #Ghi tung key:value cua dictionary ra file
        for i in list_class:
            with open(path+"/"+str(i)+".train","w") as fw:
                fw.writelines(list_class[i])



def GetPosition(value):
    #Lay mot phan cua chuoi truoc ky tu space
        position = value.find(" ")
        return position



def Predict(path,ftest,list_label_real):
        list_model=os.listdir(path)
        list_label=[]
        list_p_val=[]
        for i in list_model:
            y,x=svm.svm_read_problem(ftest.name)
            m=svm.svm_load_model(path+"/"+i)
            p_label, p_acc, p_val = svm.svm_predict(y, x, m)
            list_label.append(os.path.splitext(i)[0])
            list_p_val.append(p_val)
            # with open(path+"/"+os.path.splitext(i)[0]+".txt","w") as fw:
            #   for x in p_val:
        out_label=[]
        out_p_val=[]
        for i in list_p_val[0]:
            out_p_val.append(i[0])
        for i in list_p_val[0]:
            out_label.append(list_label[0])
        for i in range(0,len(list_p_val)):
            for j in range(0,len(list_p_val[i])):
                if float(list_p_val[i][j][0])>float(out_p_val[j]):
                    out_p_val[j]=list_p_val[i][j][0]
                    out_label[j]=list_label[i]
        acc=balanced_accuracy_score(list_label_real,out_label)
        return acc,out_label


def Find_best_nu_gamma(nu_start=0.1,nu_end=1,nu_step=.05,gamma_start=0,gamma_end=10,gamma_step=1):
    list_nu= np.arange(nu_start,nu_end,nu_step)
    list_gamma= np.arange(gamma_start,gamma_end,gamma_step)

    if(os.path.exists(path_stock+f'tmp/tmp50')==False):
        os.mkdir(path_stock+f"tmp/tmp50")
        
        
    if(os.path.exists(path_stock+f'svm_data')==False):
            os.mkdir(path_stock+f"svm_data")
    now = datetime.now()

    # convert to string
    str_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    list_nu_gamma=[]
    list_ba=[]
    best_nu=0
    best_gamma=0
    max_acc=0

    for nu in list_nu:
        for gamma in list_gamma:
            # for file_name in list_file:
                # get time now
                now = datetime.now()
                current_time = now.strftime("%H%M%S")

                with open(path_stock+"train/data.train") as ftrain:
                    SplitLabel(ftrain)
                path_tmp50=path_stock+"tmp/tmp50/data/"
                path_data=os.listdir(path_tmp50)
                path_model=path_stock+"tmp/tmp50/model/"

                if os.path.exists(path_model):
                    shutil.rmtree(path_model)
                os.makedirs(path_model)
                for i in path_data:
                    y,x=svm.svm_read_problem(path_tmp50+i)
                    prob=svm.svm_problem(y,x)
                    param=svm.svm_parameter("-s 2 -q -n "+str(float(nu))+" -g "+str(float(gamma)))
                    m=svm.svm_train(prob,param)
                    svm.svm_save_model(path_model+os.path.splitext(os.path.basename(i))[0]+".model",m)

                #test
                line_ftest=[]
                with open(path_stock+'train/data.test') as ftest:
                    line_ftest=ftest.readlines()
                list_label_real=[]
                for i in line_ftest:
                    text=i[0:GetPosition(i)]
                    list_label_real.append(text)
                #chay thu lai cai ei
                yt,xt=svm.svm_read_problem(path_stock+'train/data.test') #test cũng vậy nè
                list_model=os.listdir(path_stock+"tmp/tmp50/model")
                list_label=[]
                list_p_val=[]
                for i in list_model:
                    m=svm.svm_load_model(path_stock+"tmp/tmp50/model/"+i)
                    p_label, p_acc, p_val = svm.svm_predict(yt, xt, m)
                    list_label.append(os.path.splitext(i)[0])
                    list_p_val.append(p_val)

                out_label=[]
                out_p_val=[]
                for i in list_p_val[0]:
                        out_p_val.append(i[0])
                for i in list_p_val[0]:
                        out_label.append(list_label[0])
                for i in range(0,len(list_p_val)):
                    for j in range(0,len(list_p_val[i])):
                        if float(list_p_val[i][j][0])>float(out_p_val[j]):
                                out_p_val[j]=list_p_val[i][j][0]
                                out_label[j]=list_label[i]

                acc=balanced_accuracy_score(list_label_real,out_label)
                print(f"{acc} {nu} {gamma}")
                f = open(path_stock+f"svm_data/svm_{str_date}.csv", "a")
                f.write(f"{acc} {nu} {gamma}\n")
                if acc>max_acc:
                        max_acc=acc
                        # best_nu=nu
                        # best_gamma=gamma
                        print(f">>>{acc}  {nu}  {gamma}")

                        
    # with open(path_stock+"tmp/best_nu_gamma.csv",'w') as file:
    #     file.write(f'{best_nu} {best_gamma}')
    shutil.rmtree(path_stock+"tmp/tmp50", ignore_errors=True)
    shutil.rmtree(path_stock+"tmp/history_nu_gamma", ignore_errors=True)
    # with open(path_stock+f"tmp/best_nu_gamma.csv") as f:
    #     tach=f.readline().split(" ")
    #     best_nu=tach[0]
    #     best_gamma=tach[1]



def SplitLabel_For_batch(ftrain,path_file):
        path=path_file+'/data/'
        #Tao folder dua tren ten file train
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


        #Doc du lieu tu file train va gan vao mang
        lines=ftrain.readlines()
        #Tao mot mang trong chua danh sach cac label
        list_label=[]
        #Tach label cua dong dau tien trong file train va gan vao danh sach label
        list_label.append(GetFirstLabel(lines[0]))
        #Vong lap for den gan cac label con lai cua file train vao danh sach label
        for i in lines[1:len(lines)]:
            #Tach label cua dong thu i
            text=GetFirstLabel(i)
            #Kiem tra ton tai trong danh sach label
            if text not in list_label:
                #Neu chua co se them label do vao danh sach label
                list_label.append(text)
        #Tao mot dictionary chua cac lop dua theo label cua tung lop
        list_class=dict()
        #Gan cho moi label la mot mang trong trong dictionary
        for i in list_label:
            list_class[i]=[]
        #Vong lap for chay tung dong trong mang du lieu train
        for i in lines:
            #Tach nhan cua dong thu i
            text=GetFirstLabel(i)
            #Gan dong thu i vao dictionary theo nhan tuong ung
            list_class[text].append(i)
        #Ghi tung key:value cua dictionary ra file
        for i in list_class:
            with open(path+"/"+str(i)+".train","w") as fw:
                fw.writelines(list_class[i])
            # Batch_to_model(path+str(i)+".train",path_file)



def Batch_to_model(path_file_read,path_folder):
    # tach lay ten cua batch va ten file train 0 hoac 1
    tach= path_file_read.split('/')
    # tao model trong file batch
    if (os.path.exists(path_folder+f"/models/")==False):
        os.makedirs(path_folder+f"/models/")
    # xu ly tao model
    y,x=svm.svm_read_problem(path_file_read)
    prob=svm.svm_problem(y,x)
    param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
    m=svm.svm_train(prob,param)
    svm.svm_save_model(path_folder+f"/models/{tach[len(tach)-3]}_{tach[len(tach)-1].replace('.train','.model')}",m)



def connect_2_file(path_file_model, path_file_train,path_file_sum):
    f= open(path_file_model)
    file_one=f.readlines()
    file_one.pop(0)
    file_one.pop(0)
    file_one.pop(0)
    file_one.pop(0)
    file_one.pop(0)
    file_one.pop(0)
    file_one.pop(0)

    f=open(path_file_train)
    file_two=f.readlines()

    arr = file_two+(file_one)
    f=open(path_file_sum,"w")
    f.write("")
    f=open(path_file_sum,"a")
    for i in range(0,len(arr)):
        f.write(arr[i])
 


def Train_By_Age(Age,batch):
    list_file_0=[]
    list_file_1=[]
    list_file_model_0=[]
    list_file_model_1=[]

    path_save_models=path_stock+f"tmp/models"
    if(os.path.exists(path_save_models)==True):
        shutil.rmtree(path_stock+f"tmp/models")
    os.makedirs(path_save_models)

    for i in range(1,batch+1):
        list_file_0.append(path_stock+f"tmp/tmp_batch/train_batch_{i}/data/0.train")
        list_file_1.append(path_stock+f"tmp/tmp_batch/train_batch_{i}/data/1.train")

    for i in range(0,batch):
        os.makedirs(path_save_models+f"/model_batch_{i+1}")
        print("----------------------------------------------------------------")
        print(i)
        if(i==0):
            y,x=svm.svm_read_problem(list_file_0[i])
            prob=svm.svm_problem(y,x)
            param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
            m=svm.svm_train(prob,param)
            svm.svm_save_model(path_save_models+f"/model_batch_{i+1}/0.model",m)
            list_file_model_0.append(path_save_models+f"/model_batch_{i+1}/0.model")

            y,x=svm.svm_read_problem(list_file_1[i])
            prob=svm.svm_problem(y,x)
            param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
            m=svm.svm_train(prob,param)
            svm.svm_save_model(path_save_models+f"/model_batch_{i+1}/1.model",m)
            list_file_model_1.append(path_save_models+f"/model_batch_{i+1}/1.model")
            print(list_file_1[i])
            print(path_save_models+f"/1.model")
        elif(i<=Age):
            for j in range(0,len(list_file_model_1)):
                connect_2_file(list_file_model_1[j],list_file_1[i],list_file_1[i])
                connect_2_file(list_file_model_0[j],list_file_0[i],list_file_0[i])
                print(list_file_1[i])
                print(list_file_model_1[j])


            y,x=svm.svm_read_problem(list_file_0[i])
            prob=svm.svm_problem(y,x)
            param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
            m=svm.svm_train(prob,param)
            svm.svm_save_model(path_save_models+f"/model_batch_{i+1}/0.model",m)
            list_file_model_0.append(path_save_models+f"/model_batch_{i+1}/0.model")

            y,x=svm.svm_read_problem(list_file_1[i])
            prob=svm.svm_problem(y,x)
            param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
            m=svm.svm_train(prob,param)
            svm.svm_save_model(path_save_models+f"/model_batch_{i+1}/1.model",m)
            list_file_model_1.append(path_save_models+f"/model_batch_{i+1}/1.model")
            print('save to '+path_save_models+f"/model_batch_{i+1}/1.model")
        else:
            for j in range(i-3,len(list_file_model_1)):
                connect_2_file(list_file_model_1[j],list_file_1[i],list_file_1[i])
                connect_2_file(list_file_model_0[j],list_file_0[i],list_file_0[i])
                print(list_file_1[i])
                print(list_file_model_1[j])

            y,x=svm.svm_read_problem(list_file_0[i])
            prob=svm.svm_problem(y,x)
            param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
            m=svm.svm_train(prob,param)
            if(i==(batch-1)):
                svm.svm_save_model(path_stock+f"/tmp/train_70__0.model",m)

            svm.svm_save_model(path_save_models+f"/model_batch_{i+1}/0.model",m)
            list_file_model_0.append(path_save_models+f"/model_batch_{i+1}/0.model")

            y,x=svm.svm_read_problem(list_file_1[i])
            prob=svm.svm_problem(y,x)
            param=svm.svm_parameter("-s 2 -q -n "+str(float(best_nu))+" -g "+str(float(best_gamma)))
            m=svm.svm_train(prob,param)
            if(i==(batch-1)):
                svm.svm_save_model(path_stock+f"/tmp/train_70__1.model",m)
                list_file_model_1.append(path_stock+f"/tmp/train_70__1.model")

            svm.svm_save_model(path_save_models+f"/model_batch_{i+1}/1.model",m)
            list_file_model_1.append(path_save_models+f"/model_batch_{i+1}/1.model")
            print('save to '+path_save_models+f"/model_batch_{i+1}/1.model")


def Train_batch(batch):
    arr=[]
    with open(path_stock+"tmp/data.train") as f:
        arr=np.array_split(f.readlines(),batch)
    if os.path.exists(path_stock+"tmp/tmp_batch"):
        shutil.rmtree(path_stock+"tmp/tmp_batch")
    # if os.path.exists(path_stock+"tmp/tmp_batch/models"):
    #     shutil.rmtree(path_stock+"tmp/tmp_batch/models")
    # os.makedirs(path_stock+"tmp/tmp_batch/models")
    for i in range(0,len(arr)):
        os.makedirs(path_stock+f"/tmp/tmp_batch/train_batch_{i+1}")
        with open(path_stock+f"/tmp/tmp_batch/train_batch_{i+1}/train_batch_{i+1}.train","w") as f: f.writelines(arr[i])
        with open(path_stock+f"/tmp/tmp_batch/train_batch_{i+1}/train_batch_{i+1}.train") as f:
            SplitLabel_For_batch(f,path_stock+f"/tmp/tmp_batch/train_batch_{i+1}")





def Test_File_30():
        
    list_acc=[]
    line_ftest=[]
    with open(path_stock+'tmp/data.test') as ftest:
        line_ftest=ftest.readlines()
    list_label_real=[]
    for i in line_ftest:
        text=i[0:GetPosition(i)]
        list_label_real.append(text)

    
    yt,xt=svm.svm_read_problem(path_stock+'tmp/data.test') #test cũng vậy nè
    list_model=os.listdir(path_stock+"tmp/models")
    list_label=[]
    list_p_val=[]
    for index in range(0,len(list_model)):
        listmodel=os.listdir(path_stock+f"tmp/models/model_batch_{index+1}")
        print(path_stock+f"tmp/models/model_batch_{index+1}")
        for i in listmodel:
            m=svm.svm_load_model(path_stock+"tmp/models/"+list_model[index]+"/"+i)
            p_label, p_acc, p_val = svm.svm_predict(yt, xt, m)
            list_label.append(os.path.splitext(i)[0])
            list_p_val.append(p_val)

        out_label=[]
        out_p_val=[]
        for i in list_p_val[0]:
                out_p_val.append(i[0])
        for i in list_p_val[0]:
                out_label.append(list_label[0])
        for i in range(0,len(list_p_val)):
            for j in range(0,len(list_p_val[i])):
                if float(list_p_val[i][j][0])>float(out_p_val[j]):
                        out_p_val[j]=list_p_val[i][j][0]
                        out_label[j]=list_label[i]

        acc=balanced_accuracy_score(list_label_real,out_label)
        list_acc.append(acc)
    if(os.path.exists(path_stock+"accuracy")==False):
       os.mkdir(path_stock+"accuracy")
    now = time.time()
    f =open(path_stock+f"accuracy/accuracy_{now}.acc","a")
    for i in list_acc:
       f.write(str(i)+"\n")
        
    
    
    
    
    
    
    
    
    
def mod_data(path_data,path_save):
    f= open(path_data,"r")
    data=f.readlines()
    f= open(path_save,"w")
    f.write("")
    f= open(path_save,"a")

    for i in data:
        tach=i.split(',')
        mang=''
        tmp=0
        for j in (tach):
            if(float(j)>10):
                j=covert_number_to_1_9(float(j))
            if(tmp==0):
                mang=mang+str(j)
            else:
                mang=mang+","+str(j)
            tmp=tmp+1
        f.writelines(str(mang))
            
            
   
            
def covert_number_to_1_9(number):
    if(number>10):
        number=float(number/10)
        return covert_number_to_1_9(number)
    else:
        return number

# mod_data(path_stock+"data/data.csv",path_stock+"data/data_mod.csv")
# Test_File_30()


tranlate_file(10, 3, path_stock+'data\\data.csv',0.1, 1, 0.05, 0, 5, 1, 2)
list_file = os.listdir(path_stock+"svm_data")
for i in list_file:
    f=open(path_stock+f"svm_data/{i}")
    mang =f.readlines()
    acc=[]
    nu=[]
    gamma=[]
    for line in mang:
        tach=line.split(' ')
        acc.append(((tach[0])))
        nu.append(((tach[1])))
        gamma.append(tach[2])
        print(tach)
    plt.plot(nu,acc)
    plt.xlabel("Nu")
    plt.ylabel("Accuracy")
    plt.title(i.replace('.csv', ''))
    if(os.path.exists(path_stock+f"svm_images")==False):
        os.makedirs(path_stock+f"svm_images")  
    plt.savefig(path_stock+f"svm_images/Nu_acc_{i.replace('.csv','')}.png")
    plt.show()
    plt.plot(gamma, acc)
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.title(i.replace('.csv', ''))
    if(os.path.exists(path_stock+f"svm_images") == False):
        os.makedirs(path_stock+f"svm_images")
    plt.savefig(path_stock+f"svm_images/Gamma_acc_{i.replace('.csv','')}.png")
    plt.show()

  










