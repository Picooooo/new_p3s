import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_all_experiments(log_folder, env_name):
    dir = os.listdir(log_folder)
    list_folder = []
    df = pd.DataFrame()
    newpath = ''

    for d in dir:
        list_folder.append(d)

    num_experiments = len(list_folder)
    for i in range(num_experiments):
        newpath = log_folder + '/' + list_folder[i] + '/progress.csv'
        df.insert(i,i,pd.read_csv(newpath)['return-average'][:1000])

    mean = []
    std = []
    x = []
    epoch = df.shape[0]
    for i in range(epoch):
        x.append(i*1000)
        mean.append(df.iloc[i].sum()/num_experiments)
        std.append(np.std(df.iloc[i]))

    ci = 1.96 * np.array(std)/np.sqrt(epoch)
    name = str(env_name)
    plt.plot(x, mean, label="P3S-TD3")
    plt.xlabel("Number of steps")
    plt.ylabel("Score")
    plt.title(name)
    plt.fill_between(x, (mean-ci), (mean+ci), color='blue', alpha=0.1)
    plt.legend()
    plt.savefig("./results/" + name)

def plot_one_experiments(log_folder, env_name, seed):
    log_file = log_folder + 'iter' + str(seed)
    df = pd.DataFrame()
    
    newpath = log_file + '/progress.csv'
    df.insert(0, 0, pd.read_csv(newpath)['return-average'][:1000])

    mean = []
    std = []
    x = []
    epoch = df.shape[0]
    for i in range(epoch):
        x.append(i*1000)
        mean.append(df.iloc[i].sum()/1)
        std.append(np.std(df.iloc[i]))

    ci = 1.96 * np.array(std)/np.sqrt(epoch)
    name = str(env_name)
    plt.plot(x, mean, label="P3S-td3")
    plt.xlabel("Number of steps")
    plt.ylabel("Score")
    plt.title(name)
    plt.fill_between(x, (mean-ci), (mean+ci), color='blue', alpha=0.1)
    plt.legend()
    plt.savefig("./results/" + name + "/" + str(seed))
   
def plot(td3_folder, sac_folder, ddpg_folder, env_name):
    folders = [td3_folder, sac_folder, ddpg_folder]
    names = ["P3S-TD3", "P3S-SAC", "P3S-DDPG"]
    colors = ["blue", "red", "green"]
    index = 0
    #loop through each algo
    for log_folder in folders:
        dir = os.listdir(log_folder)
        list_folder = []
        df = pd.DataFrame()
        newpath = ''
        for d in dir:
            if os.path.isdir(os.path.join(log_folder, d)):
                list_folder.append(d)

        num_exp = len(list_folder)
        for i in range(num_exp):
            newpath = log_folder + list_folder[i] + '/progress.csv'
            print(newpath)
            df.insert(i,i,pd.read_csv(newpath)['return-average'][:1000])

        mean = []
        std = []
        x = []
        epoch = df.shape[0]
        for i in range(epoch):
            x.append(i*4000)
            mean.append(df.iloc[i].sum()/num_exp)
            std.append(np.std(df.iloc[i]))

        ci = 1.96 * np.array(std)/np.sqrt(epoch)
        plt.plot(x, mean, label=names[index])
        plt.fill_between(x, (mean-ci), (mean+ci), color=colors[index], alpha=0.1)
        index +=1
    name = str(env_name) + "_3algo"
    plt.title(str(name))
    plt.legend()
