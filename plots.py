import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pandas as pd
import seaborn as sns
import csv
import argparse
# from scipy.interpolate import spline
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env',
    default= "Office-Coffee-Task-v0",
    help="Environment"
)
parser.add_argument(
    '--exp',
    default="office",
    help="Experiment. E.g office, office_iclr, movingtargets_iclr",
)
args = parser.parse_args()


def plot_office_iclr():
     
    num_runs = 60
    num_steps = 400000
    metric = "eval total reward"
    m = 4
    metric = "eval successes"
    m = 1

    labels = ['SM (Ours)','QL-SM (Ours)','CRM','CRM-RS','HRM','HRM-RS','QL',"QL-RS"]
    dirs = ['sm-ql/zeroshot','sm-ql/fewshot','crm','crm-rs','hrm','hrm-rs','rm-ql','rm-ql-rs']

    data = []

    for j in range(len(dirs)):
        try:
            dirr = f'baselines_iclr/data/logs/{dirs[j]}/{args.env}/'
            filer = open(dirr+'0/progress.csv')
            print(dirr+'0/progress.csv')
            csvreader = csv.reader(filer)
            all_data_pnts = [row for row in csvreader]
            episodes, steps, performance = all_data_pnts[0].index("episodes"), all_data_pnts[0].index("steps"), all_data_pnts[0].index(metric)
            all_data_pnts = np.array(all_data_pnts[1:]).astype(np.float32)[:num_steps,:]
            task = all_data_pnts[:,steps]
            print("data:", all_data_pnts.shape)
            a = np.sum(all_data_pnts[:,episodes].reshape(-1, m), axis=1)
            b = np.sum(all_data_pnts[:,steps].reshape(-1, m), axis=1)
            c = np.sum(all_data_pnts[:,performance].reshape(-1, m), axis=1)
            all_data_pnts = np.array([a,b,c]).T
            
            for i in range(num_runs):
                try:
                    filer = open(dirr+'/'+str(i)+'/progress.csv')
                    csvreader = csv.reader(filer)
                    data_pnts = []
                    for row in csvreader: data_pnts.append(row)
                    episodes, steps, performance = data_pnts[0].index("episodes"), data_pnts[0].index("steps"), data_pnts[0].index(metric)
                    # data_pnts[1][-2:]=["0","0"]
                    data_pnts = np.array(data_pnts[1:]).astype(np.float32)[:num_steps,:]
                    a = np.sum(data_pnts[:,episodes].reshape(-1, m), axis=1)
                    b = np.sum(data_pnts[:,steps].reshape(-1, m), axis=1)
                    c = np.sum(data_pnts[:,performance].reshape(-1, m), axis=1)
                    data_pnts = np.array([a,b,c]).T
                    all_data_pnts = np.dstack( [all_data_pnts, data_pnts] )
                except:
                    print(labels[j], i, "skipped")
            data.append((np.mean(all_data_pnts, axis=2)[:,2], np.std(all_data_pnts, axis=2)[:,2], labels[j]))
        except:
            print(labels[j], "skipped")
    s = 20

    rc_ = {'figure.figsize':(9,7),'axes.labelsize': 30, 'xtick.labelsize': s, 
        'ytick.labelsize': s, 'legend.fontsize': 20}
    sns.set(rc=rc_, style="darkgrid")
    # rc('text', usetex=True)
    lw = 2.0

    fig, ax = plt.subplots()
    for (mean, std, label) in data:
        task = np.linspace(task.min(), task.max(), len(mean))  
        ax.plot(task, mean, lw = lw, label=label)
        ax.fill_between(task, mean - std, mean + std, alpha=0.4)

    # plt.legend(loc="lower left", bbox_to_anchor=(0,0.025))
    plt.legend()
    # ax.legend_ = None
    plt.xlabel("Steps")
    if metric == "eval total reward": plt.ylabel("Total Reward")
    else:                             plt.ylabel(metric)
    #plt.ylim(top=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.tight_layout()
    fig.savefig(f"images/iclr_{args.env}_{metric}.pdf", bbox_inches='tight')
    plt.show()


def plot_office():

    num_runs = 10
    num_steps = 400000//10000
    metric = "eval total reward"
    m = 1
    metric = "eval successes"
    m = 1

    labels = ['SM (Ours)','SM-Pretrained (Ours)','SM-QL-Pretrained (Ours)','QL (Baseline)']
    dirs = ['sm_ql','sm_ql/zeroshot','sm_ql/fewshot','ql']

    data = []

    for j in range(len(labels)):
        if True:
            dirr = f'data/logs/{dirs[j]}/{args.env}/'
            filer = open(dirr+'0/progress.csv')
            print(dirr+'0/progress.csv')
            csvreader = csv.reader(filer)
            all_data_pnts = [row for row in csvreader]
            episodes, steps, performance = all_data_pnts[0].index("episodes"), all_data_pnts[0].index("steps"), all_data_pnts[0].index(metric)
            all_data_pnts = np.array(all_data_pnts[1:]).astype(np.float32)[:num_steps,:]
            task = all_data_pnts[:,steps]
            print("data:", all_data_pnts.shape)
            a = np.sum(all_data_pnts[:,episodes].reshape(-1, m), axis=1)
            b = np.sum(all_data_pnts[:,steps].reshape(-1, m), axis=1)
            c = np.sum(all_data_pnts[:,performance].reshape(-1, m), axis=1)
            all_data_pnts = np.array([a,b,c]).T
            
            for i in range(num_runs):
                try:
                    filer = open(dirr+'/'+str(i)+'/progress.csv')
                    csvreader = csv.reader(filer)
                    data_pnts = []
                    for row in csvreader: data_pnts.append(row)
                    # data_pnts[1][-2:]=["0","0"]
                    episodes, steps, performance = data_pnts[0].index("episodes"), data_pnts[0].index("steps"), data_pnts[0].index(metric)
                    data_pnts = np.array(data_pnts[1:]).astype(np.float32)[:num_steps,:]
                    a = np.sum(data_pnts[:,episodes].reshape(-1, m), axis=1)
                    b = np.sum(data_pnts[:,steps].reshape(-1, m), axis=1)
                    c = np.sum(data_pnts[:,performance].reshape(-1, m), axis=1)
                    data_pnts = np.array([a,b,c]).T
                    all_data_pnts = np.dstack( [all_data_pnts, data_pnts] )
                except:
                    print(labels[j], i, "skipped")
            data.append((np.mean(all_data_pnts, axis=2)[:,2], np.std(all_data_pnts, axis=2)[:,2], labels[j]))
    s = 20

    rc_ = {'figure.figsize':(9,7),'axes.labelsize': 30, 'xtick.labelsize': s, 
        'ytick.labelsize': s, 'legend.fontsize': 20}
    sns.set(rc=rc_, style="darkgrid")
    # rc('text', usetex=True)
    lw = 2.0

    fig, ax = plt.subplots()
    for (mean, std, label) in data:
        task = np.linspace(task.min(), task.max(), len(mean))  
        ax.plot(task, mean,  label=label, lw = lw)
        ax.fill_between(task, mean - std, mean + std, alpha=0.4)

    # plt.legend(loc="lower left", bbox_to_anchor=(0,0.025))
    plt.legend(loc="lower right")
    # ax.legend_ = None
    plt.xlabel("steps")
    if metric == "eval total reward": plt.ylabel("Total Discounted Reward")
    else:                             plt.ylabel(metric)
    #plt.ylim(top=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    fig.savefig(f"images/{args.env}_{metric}.pdf", bbox_inches='tight')
    # plt.show()

def plot_sb3(algo):

    num_runs = 10
    num_steps = 1000000
    metric = "eval total reward"
    m = 1
    #metric = "eval successes"
    #m = 1

    labels = ['SM (Ours)',f'{algo} (Baseline)']
    dirs = [(f'sm_{algo}','/wvf_1'),(algo,"/skill")]

    data = []

    for j in range(len(labels)):
        if True:
            dirr = f'data/logs/{dirs[j][0]}/{args.env}/'
            filer = open(dirr+'0'+dirs[j][1]+'/progress.csv')
            print(dirr+'0'+dirs[j][1]+'/progress.csv')
            csvreader = csv.reader(filer)
            all_data_pnts = [[0 if a=='' else a for a in row] for row in csvreader]
            episodes, steps, performance = all_data_pnts[0].index("time/episodes"), all_data_pnts[0].index("time/total_timesteps"), all_data_pnts[0].index(metric)
            all_data_pnts = np.array(all_data_pnts[1:]).astype(np.float32)[:num_steps,:]
            task = all_data_pnts[:,steps]
            print("data:", all_data_pnts.shape)
            a,b,c,e = [], [], [], 1
            for k in range(len(all_data_pnts)):
                if all_data_pnts[k,steps]>e:
                    e = all_data_pnts[k,steps]+10000
                    a.append(all_data_pnts[k,episodes])
                    b.append(all_data_pnts[k,steps])
                    c.append(all_data_pnts[k,performance])
            #a = np.sum(all_data_pnts[:,episodes].reshape(-1, m), axis=1)
            #b = np.sum(all_data_pnts[:,steps].reshape(-1, m), axis=1)
            #c = np.sum(all_data_pnts[:,performance].reshape(-1, m), axis=1)
            all_data_pnts = np.array([a,b,c]).T
            
            for i in range(num_runs):
                try:
                    filer = open(dirr+'/'+str(i)+dirs[j][1]+'/progress.csv')
                    csvreader = csv.reader(filer)
                    data_pnts = []
                    for row in csvreader: data_pnts.append([0 if a=='' else a for a in row])
                    episodes, steps, performance = data_pnts[0].index("time/episodes"), data_pnts[0].index("time/total_timesteps"), data_pnts[0].index(metric)
                    data_pnts = np.array(data_pnts[1:]).astype(np.float32)[:num_steps,:]
                    a,b,c,e = [], [], [], 1
                    for k in range(len(data_pnts)):
                        if data_pnts[k,steps]>e:
                            e = data_pnts[k,steps]+10000
                            a.append(data_pnts[k,episodes])
                            b.append(data_pnts[k,steps])
                            c.append(data_pnts[k,performance])
                    #a = np.sum(data_pnts[:,episodes].reshape(-1, m), axis=1)
                    #b = np.sum(data_pnts[:,steps].reshape(-1, m), axis=1)
                    #c = np.sum(data_pnts[:,performance].reshape(-1, m), axis=1)
                    data_pnts = np.array([a,b,c]).T
                    all_data_pnts = np.dstack( [all_data_pnts, data_pnts] )
                except:
                    print(labels[j], i, "skipped")
            print(all_data_pnts.shape)
            data.append((np.mean(all_data_pnts, axis=2)[:,2], np.std(all_data_pnts, axis=2)[:,2], labels[j]))
    s = 20

    rc_ = {'figure.figsize':(9,7),'axes.labelsize': 30, 'xtick.labelsize': s, 
        'ytick.labelsize': s, 'legend.fontsize': 20}
    sns.set(rc=rc_, style="darkgrid")
    # rc('text', usetex=True)
    lw = 2.0

    fig, ax = plt.subplots()
    for (mean, std, label) in data:
        task = np.linspace(task.min(), task.max(), len(mean))  
        ax.plot(task, mean,  label=label, lw = lw)
        ax.fill_between(task, mean - std, mean + std, alpha=0.4)

    # plt.legend(loc="lower left", bbox_to_anchor=(0,0.025))
    plt.legend(loc="lower right")
    # ax.legend_ = None
    plt.xlabel("steps")
    if metric == "eval total reward": plt.ylabel("Total Discounted Reward")
    else:                             plt.ylabel(metric)
    #plt.ylim(top=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    fig.savefig(f"images/{args.env}_{metric}.png", bbox_inches='tight')
    # plt.show()

if   args.exp=="office":      plot_office()
elif args.exp=="office_iclr": plot_office_iclr()
else:                         plot_sb3(args.exp)
