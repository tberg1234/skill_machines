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
# metric = "successes"
# m = 1
metric = "total reward"
m = 40

def plot_office_iclr():
     
    num_runs = 25
    num_steps = 400000
    
    labels = ['SM (Ours)','QL-SM (Ours)','CRM','CRM-RS','HRM','HRM-RS','QL',"QL-RS"]
    dirs = ['sm_ql/zeroshot','sm_ql/fewshot','crm','crm-rs','hrm','hrm-rs','rm-ql','rm-ql-rs']

    data = []

    for j in range(len(dirs)):
        dirr = f'data/logs/{dirs[j]}/{args.env}/'
        filer = open(dirr+'0/progress.csv')
        print(dirr+'0/progress.csv')
        csvreader = csv.reader(filer)
        all_data_pnts = [row for row in csvreader]
        episodes, steps, performance = all_data_pnts[0].index("episodes"), all_data_pnts[0].index("steps"), all_data_pnts[0].index(f"eval {metric}")
        all_data_pnts = np.array(all_data_pnts[1:]).astype(np.float32)[:num_steps,:]
        print(all_data_pnts.shape)
        a = np.sum(all_data_pnts[:,episodes].reshape(-1, m), axis=1)
        b = np.sum(all_data_pnts[:,steps].reshape(-1, m), axis=1)
        c = np.sum(all_data_pnts[:,performance].reshape(-1, m), axis=1)
        all_data_pnts = np.array([a,b,c]).T
        task = all_data_pnts[:,1]
        print(all_data_pnts.shape)

        for i in range(num_runs):
            try:
                filer = open(dirr+'/'+str(i)+'/progress.csv')
                csvreader = csv.reader(filer)
                data_pnts = []
                for row in csvreader: data_pnts.append(row)
                episodes, steps, performance = data_pnts[0].index("episodes"), data_pnts[0].index("steps"), data_pnts[0].index(f"eval {metric}")
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
    plt.legend()
    # ax.legend_ = None
    plt.xlabel("Steps")
    # plt.ylabel('Total Reward')
    #plt.ylim(top=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    fig.tight_layout()
    fig.savefig(f"images/{args.env}_{metric}.pdf", bbox_inches='tight')
    plt.show()


def plot_office():

    num_runs = 25
    num_steps = 400000

    labels = ['SM (Ours)','QL-SM (Ours)']
    dirs = ['sm_ql/zeroshot','sm_ql/fewshot']

    data = []

    for j in range(len(labels)):
        dirr = f'data/logs/{dirs[j]}/{args.env}/'
        filer = open(dirr+'0/progress.csv')
        print(dirr+'0/progress.csv')
        csvreader = csv.reader(filer)
        all_data_pnts = [row for row in csvreader]
        episodes, steps, performance = all_data_pnts[0].index("episodes"), all_data_pnts[0].index("steps"), all_data_pnts[0].index(f"eval {metric}")
        all_data_pnts = np.array(all_data_pnts[1:]).astype(np.float32)[:num_steps,:]
        print(all_data_pnts.shape)
        a = np.sum(all_data_pnts[:,episodes].reshape(-1, m), axis=1)
        b = np.sum(all_data_pnts[:,steps].reshape(-1, m), axis=1)
        c = np.sum(all_data_pnts[:,performance].reshape(-1, m), axis=1)
        all_data_pnts = np.array([a,b,c]).T
        task = all_data_pnts[:,1]
        print(all_data_pnts.shape)

        for i in range(num_runs):
            try:
                filer = open(dirr+'/'+str(i)+'/progress.csv')
                csvreader = csv.reader(filer)
                data_pnts = []
                for row in csvreader: data_pnts.append(row)
                episodes, steps, performance = data_pnts[0].index("episodes"), data_pnts[0].index("steps"), data_pnts[0].index(f"eval {metric}")
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
    plt.xlabel("Steps")
    # plt.ylabel('Total Reward')
    #plt.ylim(top=2)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    fig.savefig(f"images/{args.env}_{metric}.pdf", bbox_inches='tight')
    # plt.show()


if   args.exp=="office":      plot_office()
elif args.exp=="office_iclr": plot_office_iclr()
else:                         print("invalid exp")
