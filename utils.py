import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import torch.nn as nn
import torch.optim as opt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def save_result(TASK, model, EPOCH, yvalid, predvalid, losses, attentions=None):
    save_file_dir = os.path.join("result", TASK, repr(model))
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    save_file_path = os.path.join(save_file_dir,"epoch" + str(EPOCH) + ".pth")
    report = {
        "y": yvalid,
        "pred": predvalid,
        "loss": losses,
        "attentions": attentions
    }
    torch.save(report, save_file_path)


def mean_error(y, y_pred):
    return np.mean(y_pred - y)


def area(Label, Prediction, bins):
    M, N = np.max(Label), np.min(Label)
    D = M - N
    real, pred = Label, Prediction
    interval           = np.array([ N + D/bins*i for i in range(bins+1) ])
    revised_interval   = interval[:-1]  + D/(2*bins)
    cumulative_number  = []
    cumulative_number1 = []
    for i in range(bins):
        cumulative_number.append(  (pred < interval[i+1]).sum() - (pred < interval[i]).sum() )
        cumulative_number1.append( (real < interval[i+1]).sum() - (real < interval[i]).sum() )
    cumulative_number  = np.array(cumulative_number) /sum(cumulative_number)    
    cumulative_number1 = np.array(cumulative_number1)/sum(cumulative_number1)     
    area = []
    for i in range(bins):
        area.append(min(cumulative_number[i],cumulative_number1[i]))
    return sum(area)


def compute_index(A, B):
    A = A.reshape(-1)
    B = B.reshape(-1)
    # Number of Pixcel
    num_pix = len(A)
    # ME
    me = mean_error(A, B)
    # MAE
    mae = mean_absolute_error(A, B)
    # MSE
    mse = mean_squared_error(A, B, squared=True)
    # RMSE
    rmse = mean_squared_error(A, B, squared=False)
    # Ara (Overlapping area of PDF, Bins=30)
    area_score = area(A, B, 30)
    # Corr
    corr = np.corrcoef(A, B)
    return num_pix, me, mae, mse, rmse, area_score, corr[0][1]


def binning(arr_pred, arr_label, bin="weak"):
    weak = (0.1 <= arr_label) & (arr_label < 1.0)
    mod = (1.0 <= arr_label) & (arr_label < 10.0)
    strong = (10 <= arr_label)
    if bin == "weak":
        return arr_pred[weak], arr_label[weak]
    if bin == "moderate":
        return arr_pred[mod], arr_label[mod]
    if bin == "strong":
        return arr_pred[strong], arr_label[strong]


def compute_index_comparison(pred, label, out_type='df'):
    index_eval_all = compute_index(pred, label)
    # Weak rain
    A, B = binning(pred, label, bin="weak")
    index_eval_weak = compute_index(A, B)
    # Moderate rain
    A, B = binning(pred, label, bin="moderate")
    index_eval_moderate = compute_index(A, B)
    # Strong rain
    A, B = binning(pred, label, bin="strong")
    index_eval_strong = compute_index(A, B)
    # Result chart
    if out_type == 'df':
        out = pd.DataFrame([index_eval_all, index_eval_weak, index_eval_moderate, index_eval_strong], 
                    index = ['All', 'Weak', 'Moderate', 'Strong'],
                    columns = ['NumPixel', 'ME','MAE', 'MSE', 'RMSE', 'Area', 'CC']).T
    if out_type == 'arr':
        out = np.array([index_eval_all, index_eval_weak, index_eval_moderate, index_eval_strong])
    return out


class EvaluationIndices():
    def __init__(self, pred_reg, label_reg, pred_cls, label_cls):
        self.pred_reg = pred_reg.reshape(-1)
        self.label_reg = label_reg.reshape(-1)
        self.pred_cls = pred_cls.reshape(-1)
        self.label_cls = label_cls.reshape(-1)
        self.num = len(self.pred_reg)
        self.c_matrix = confusion_matrix(self.label_cls, self.pred_cls)
        self.tn, self.fp, self.fn, self.tp = self.c_matrix.ravel()
        self.c, self.f, self.m, self.h = self.tn, self.fp, self.fn, self.tp
    # Regression
    def me(self, a, b):
        return mean_error(a, b)
    def mae(self, a, b):
        return mean_absolute_error(a, b)
    def rmse(self, a, b):
        return mean_squared_error(a, b, squared=False)
    def cc(self, a, b):
        return np.corrcoef(a, b)[0][1]
    # Classification
    def pod(self, h, m):
        return h / (h + m)
    def far(self, f, h):
        return f / (h + f)
    def bias(self, h, f, m):
        return (h + f) / (h + m)
    def ets(self, h, c, f, m):
        r = ((h + m)*(h + f)) / (h + m + f + c)
        return (h - r) / (h + m + f - r)
    def hss(self, h, c, f, m):
        nume = 2*(h * c - f * m)
        denom = ((h + m) * (m + c)) + ((h + f) * (f + c))
        return nume / denom
    # Run
    def evaluate(self):
        return {
        "ME":self.me(self.pred_reg, self.label_reg), 
        "MAE":self.mae(self.pred_reg, self.label_reg), 
        "RMSE":self.rmse(self.pred_reg, self.label_reg), 
        "Area":area(self.pred_reg, self.label_reg, 30), 
        "CC":self.cc(self.pred_reg, self.label_reg), 
        "POD":self.pod(self.h, self.m), 
        "FAR":self.far(self.f, self.h), 
        "BIAS":self.bias(self.h, self.f, self.m), 
        "ETS":self.ets(self.h, self.c, self.f, self.m),
        "HSS":self.hss(self.h, self.c, self.f, self.m)
        }


def show_regression_performance_2d(x, y, title_name):
    x = x.reshape(-1)
    y = y.reshape(-1)
    # Cor. coef
    # coef = np.corrcoef(x, y)[0][1]
    plt.figure(figsize=(7, 5.5))
    # log10
    with np.errstate(divide='ignore'):
        x = np.log10(x)
        y = np.log10(y)
    # plot
    plt.hist2d(y, x, bins=(55, 55), cmap=cm.jet, range=[[0.01, 1.5], [0.01, 1.5]], vmax=65)
    plt.colorbar()
    plt.ylabel('Pred: Precipitation ($log_{10}$[mm/h])')
    plt.xlabel('Label: Precipitation ($log_{10}$[mm/h])')
    plt.title('{a}'.format(a=title_name))
    plt.xticks(np.arange(0, 1.75, 0.25))
    plt.yticks(np.arange(0, 1.75, 0.25))
    # daiagonal line
    ident = [0.01, 1.5]
    plt.plot(ident, ident, ls="--", lw="1.2", c="gray")


def show_regression_diff_2d(x_base, y_base, x, y, title_name):
    x_base = x_base.reshape(-1)
    y_base = y_base.reshape(-1)
    x = x.reshape(-1)
    y = y.reshape(-1)
    # log10
    with np.errstate(divide='ignore'):
        x_base = np.log10(x_base)
        y_base = np.log10(y_base)
        x = np.log10(x)
        y = np.log10(y)
    # plot
    height_base, xedge_base, yedge_base, fig = plt.hist2d(y_base, x_base, bins=(55, 55), cmap=cm.jet, range=[[0.01, 1.5], [0.01, 1.5]])
    plt.close()
    height, xedges, yedges, fig = plt.hist2d(y, x, bins=(55, 55), cmap=cm.jet, range=[[0.01, 1.5], [0.01, 1.5]])
    plt.close()
    # dif plot
    H =  height - height_base
    plt.figure(figsize=(7, 5.5))
    plt.imshow(H, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           vmin=-20, vmax=20, cmap="bwr")
    plt.colorbar()
    plt.xticks(np.arange(0, 1.75, 0.25))
    plt.yticks(np.arange(0, 1.75, 0.25))
    plt.ylabel('Pred: Precipitation ($log_{10}$[mm/h])')
    plt.xlabel('Label: Precipitation ($log_{10}$[mm/h])')
    plt.title('{a}'.format(a=title_name))
    # daiagonal line
    ident = [0.01, 1.5]
    plt.plot(ident, ident, ls="--", lw="1.2", c="gray")

def product_comparison_2d(x, y, title_name):
    x = x.reshape(-1)
    y = y.reshape(-1)
    # Plot
    plt.figure(figsize=(7, 5.5))
    plt.scatter(y, x, alpha=.3, s=4, c="Blue")
    # decoration
    plt.ylabel('Pred: Precipitation (mm/h)')
    plt.xlabel('Label: Precipitation (mm/h)')
    plt.title('{a}'.format(a=title_name))
    plt.xlim(-1, 60)
    plt.ylim(-1, 60)
    plt.xticks(np.arange(0, 60, 5))
    plt.yticks(np.arange(0, 60, 5))
    # daiagonal line
    ident = [0.01, 60]
    plt.plot(ident, ident, ls="--", lw="1.2", c="gray")


def convert_ClsfLabel(rain_th = 0.0, y = None):
    assert y.shape[1] == 2, "Lable needs to be 2 dim (clsf, reg)"
    reg_label = y[:, 1]
    clsf_label = (reg_label > rain_th).float()
    return clsf_label


def convert_rainbin(reg_label, rain_th):
    if reg_label < rain_th[0]:
        return 0
    elif (rain_th[0] <= reg_label) & (reg_label < rain_th[1]):
        return 1
    elif (rain_th[1] <= reg_label) & (reg_label < rain_th[2]):
        return 2
    elif rain_th[2] <= reg_label:
        return 3


def convert_MutiLabel(rain_th = [0.1, 1.0, 10.0], y = None):
    reg_label = y[:, 1]
    # no_rain, weak, mod, strongã¸åˆ†é¡ž
    mutli_label = [convert_rainbin(reg_label[i], rain_th) for i in range(len(reg_label))]
    return torch.tensor(mutli_label)


def computed_bin_ensumble(pred_reg, label_reg):
    # Calicurate binned-result
    binned_result = []
    for i in tqdm(range(len(pred_reg))):
        _ = compute_index_comparison(pred_reg[i], label_reg[i], out_type='arr')
        binned_result.append(_)
    binned_result = np.array(binned_result)
    # Mean and Std (Dimension reduction)
    result_m = binned_result.mean(axis=0)
    result_s = binned_result.std(axis=0)
    # To df
    result_m = pd.DataFrame(result_m, 
                            columns=['NumPixel', 'ME', 'MAE', 'MSE', 'RMSE', 'Area', 'CC'],
                            index=['All', 'Weak', 'Moderate', 'Strong'])
    result_s = pd.DataFrame(result_s,
                            columns=['NumPixel', 'ME', 'MAE', 'MSE', 'RMSE', 'Area', 'CC'],
                            index=['All', 'Weak', 'Moderate', 'Strong'])
    return result_m, result_s


def kde_plot(ax, data_1, data_2, label_1 = "Single-task", label_2 = "Two-task", title = "a"):
    x_bin = 20
    # bin_min = np.min(np.concatenate([data_1, data_2]))
    # bin_max = np.max(np.concatenate([data_1, data_2]))
    # x_bin = np.linspace(bin_min, bin_max, 20)
    
    bw_method = 0.2

    # fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    hist1 = ax.hist(data_1, bins=x_bin, label=label_1, alpha=.7, color="gray")
    hist2 = ax.hist(data_2, bins=x_bin, label=label_2, alpha=.7, color="steelblue")

    xticks = ax.get_xticks()
    
    ax2 = ax.twinx()
    kde_range = np.linspace(xticks[0], xticks[-1], 100)

    kde_model1 = gaussian_kde(data_1, bw_method)
    kde_res1 = kde_model1(kde_range)
    kde_model2 = gaussian_kde(data_2, bw_method)
    kde_res2 = kde_model2(kde_range)

    ax2.plot(kde_range, kde_res1, lw=2, color="gray", ls="--")
    ax2.plot(kde_range, kde_res2, lw=2, color="steelblue", ls="--")
    ax2.set_yticks([])

    ax.grid(linestyle='dotted', linewidth=1)
    # ax.legend(bbox_to_anchor=(1.3, 1), fontsize=15)
    # ax.set_title(title)
    ax.set_ylim(0, 17)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20, length=5)
    ax2.tick_params(axis='x', labelsize=20, length=5)
    
    # p_val = stats.ttest_ind(data_1, data_2, equal_var=True)
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))


def plot_globalmap(data, title, cmap_min, cmap_vmax, cmap_type):
    # Figure decoration
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([60, 270, -45, 40])
    # ax.add_feature(cfeature.LAND)
    ax.coastlines()
    ax.gridlines()
    ax.set_title('{}'.format(title))
    # Data plotting
    im = ax.imshow(data, transform=ccrs.PlateCarree(), origin='lower',
                   vmin=cmap_min, vmax=cmap_vmax, cmap=cmap_type)
    plt.colorbar(im, aspect=50,pad=0.03,orientation='horizontal')
