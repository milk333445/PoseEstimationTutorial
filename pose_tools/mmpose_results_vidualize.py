import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors as mcolors
import pandas as pd
import random
from tqdm import tqdm

plt.rcParams['axes.unicode_minus']=False
random.seed(124)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
linestyle = ['--', '-.', '-']
loss_metrics = ['loss', 'loss_kpt']
acc_metrics = ['acc_pose']
ap_metrics = ['coco/AP', 'coco/AP .5', 'coco/AP .75', 'coco/AP (M)', 'coco/AP (L)', 'coco/AR', 'coco/AR .5', 'coco/AR .75', 'coco/AR (M)', 'coco/AR (L)', 'PCK', 'AUC']
test_metrics = ['NME']
def process_log(log_path):
    with open(log_path, 'r') as f:
        json_list = f.readlines()

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for each in tqdm(json_list):
        if 'coco/AP' in each:
            df_test = df_test.append(eval(each), ignore_index=True)
        else:
            df_train = df_train.append(eval(each), ignore_index=True)

    return df_train, df_test
def get_line_arg():
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    line_arg['linestyle'] = random.choice(linestyle)
    line_arg['linewidth'] = random.randint(1, 4)
    return line_arg


def plot_metrics(df_train, df_test, loss_metrics, acc_metrics, ap_metrics, test_metrics, get_line_arg):
    
    plt.figure(figsize=(16, 8))

    # 第一个图表 (train loss)
    plt.subplot(2, 2, 1)
    x = df_train['step']
    for y in loss_metrics:
        try:
            plt.plot(x, df_train[y], label=y, **get_line_arg())
        except:
            pass
    plt.tick_params(labelsize=12)
    plt.xlabel('step', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.title('train loss', fontsize=15)
    plt.legend(fontsize=12)

    # 第二个图表 (train acc)
    plt.subplot(2, 2, 2)
    x = df_train['step']
    for y in acc_metrics:
        try:
            plt.plot(x, df_train[y], label=y, **get_line_arg())
        except:
            pass
    plt.tick_params(labelsize=12)
    plt.xlabel('step', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.title('train acc', fontsize=15)
    plt.legend(fontsize=12)

    # 第三个图表 (test metrics)
    plt.subplot(2, 2, 3)
    x = df_test['step']
    for y in ap_metrics:
        try:
            plt.plot(x, df_test[y], label=y, **get_line_arg())
        except:
            pass
    plt.tick_params(labelsize=12)
    plt.xlabel('step', fontsize=12)
    plt.ylabel('metrics', fontsize=12)
    plt.title('test metrics', fontsize=15)
    plt.legend(fontsize=12)

    # 第四个图表 (test metrics 2)
    plt.subplot(2, 2, 4)
    x = df_test['step']
    for y in test_metrics:
        try:
            plt.plot(x, df_test[y], label=y, **get_line_arg())
        except:
            pass
    plt.tick_params(labelsize=12)
    plt.xlabel('step', fontsize=12)
    plt.ylabel('metrics', fontsize=12)
    plt.title('test metrics 2', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    log_path = './work_dirs/rtmpose-m-carpose/20240421_225307/vis_data/scalars.json' # 訓練日誌
    df_train, df_test = process_log(log_path)
    plot_metrics(df_train, df_test, loss_metrics, acc_metrics, ap_metrics, test_metrics, get_line_arg)
    
    