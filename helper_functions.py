## Functions supporting the Project notebook

import warnings
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
import gspread
import sklearn.preprocessing
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from operator import add
import itertools
import math
from sklearn.metrics import mean_squared_error
import matplotlib.animation as animation
from scipy import interpolate
import os
pd.set_option('display.multi_sparse', False)

# Suppress annoying red box warnings{'a':1}
warnings.filterwarnings('ignore')

# Read in data from Google sheet
def import_sheet(name):
    """Use Google Sheets API to read in latest verison of data file
    
    Keyword arguments:
    name -- name of Google sheet (string)
    """
    
    # Use credentials to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds']
    creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
    client = gspread.authorize(creds)

    # Read the spreadsheet and convert to a Pandas dataframe
    sheet = client.open(name).sheet1
    df = pd.DataFrame(sheet.get_all_records(empty2zero=True)) # will be a column name alphabetized dataframe
    
    # Convert the dataframe to numeric
    df.astype('float64')
    return df


# Flip dataframe so most recent times are first
def flip_df(orig_df):

    df = orig_df.copy()
    df = df.sort_index(ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


# Standardiz feature ranges to do feature construction
def standardize(orig_df, feature_range=[2,4], comp_var='Price'):
    """
    Scale all of a dataframe's columns linearly to some range
    """

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=feature_range)
    std_array = scaler.fit_transform(orig_df)
    std_df = pd.DataFrame(data=std_array, columns=orig_df.columns.values.tolist())

    min_cv = orig_df[comp_var].min()
    max_cv = orig_df[comp_var].max()

    def rescaler(y):

        return (y-feature_range[0])*(max_cv-min_cv)/(feature_range[1]-feature_range[0])+min_cv
    
    return std_df, rescaler

# Interpolate between rows in an integer-indexed dataframe
def df_interpolate(orig_df, min, max, curve=3, method='polynomial'):
    if max == 0:
        return orig_df

    resample = np.linspace(max,min,orig_df.shape[0])**curve/(max**(curve-1))
    fillers = [['new']*int(np.floor(n)) for n in resample]
    new_index = []
    for i in orig_df.index:
        new_index.append(i)
        for filler in fillers[i]:
            if filler is not []:
                new_index.append(filler)
    df_nans = orig_df.reindex(index=new_index).reset_index(drop=True)
    # print(df_nans)
    return df_nans.interpolate(method=method, order=3).dropna()


# Plot all datframe signals
def signal_plot(orig_df):
    """Create an overlaid time series plot of all features in a dataframe.
    
    Keyword arguments:
    """

    df = orig_df.copy()
    df, not_used = standardize(df, feature_range=[2,4])
    df['time'] = df.index
    df['unit'] = np.zeros((len(df),1))
    df_melt = pd.melt(df,id_vars=['time','unit'],value_vars=orig_df.columns.values.tolist())
    fig = plt.figure(figsize=(60,14))
    tsplot = sns.tsplot(data=df_melt, time='time', value='value',condition='variable',unit='unit')
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.show()


# Regular heatmap
def reg_heat_map(orig_df, show=True, include_mask=False, include_annot=False):
    """Create a heat map of Pearson correlation coefficients between all features.
    
    Keyword arguments:
    orig_df -- a pandas dataframe
    include_mask -- True or False, whether or not to hide the upper triangle
    include_annot -- True or False, whether or not to write numbers in the boxes
    """

    df = orig_df.copy()
    num_cols = df.shape[1]

    # Compute correlation coefficients
    corr = df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    if include_mask:
        mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    plt.subplots(figsize=(num_cols/3,num_cols/3))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    plot = sns.heatmap(np.round(corr,2), mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=0, annot=include_annot, annot_kws={"size": 7}, cbar_kws={"shrink":1})
    if show:
        plt.show()

    return plot
    
 
# Pearson coeff
def pearson(sig1,sig2):
    """Calculate the Pearson Correlation Coefficient of two signals (two numpy vectors)"""
    # print(sig1)
    # print(sig2)
    return np.cov(np.array([sig1,sig2]))[0,1]/(np.std(sig1)*np.std(sig2))


# Time heatmap
def time_heat_map(orig_df, show=True, sort='range', comp_var='Price', months=60, 
                    include_annot=False, include_cbar=True):
    """Create a heatmap of Pearson correlation coefficients where all features are 
    compared to one feature at different time delays. Rows are sorted by best maximum observed
    correlation with price.

    Keyword arguments:
    orig_df: pandas dataframe
    months: number of months to look back in time
    comp_var: name of feature to compare to all others (string, e.g. 'Price')
    include_annot: True or False, whether or not to write numbers in the boxes
    """

    df = orig_df.copy()

    num_cols = df.shape[1] # Number of dataframe columns

    # Create an empty correlation matrix. Rows will be features and columns will be time delays.
    time_corr = np.zeros((num_cols,months+1))
    #feature_names = list(df.drop('Price',axis=1))
    feature_names = df.columns.values.tolist()

    for i in range(num_cols): # For every feature
        for j in range(months+1):    # For every month delay, from 0 up to and including n
            # Truncate beginning of price signal
            comp_var_sig = df[comp_var][j:len(df[comp_var])+1]
            # Truncate end of feature signal
            feature_name = feature_names[i]
            #print(feature_name)
            feature_sig = df[feature_name][0:len(df[feature_name])-j]
            time_corr[i,j] = np.round(pearson(comp_var_sig,feature_sig),2)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Create dataframe from matrix of correlations
    time_corr_df = pd.DataFrame(data=np.transpose(time_corr),columns=feature_names)
    if sort == 'single':
        time_corr_df = time_corr_df.reindex_axis(time_corr_df.loc[0].abs().sort_values().index, axis=1)
    elif sort == 'range':
        time_corr_df = time_corr_df.reindex_axis(time_corr_df.abs().max().sort_values().index, axis=1)

    # Generate heat map
    plt.subplots(figsize=((months+1)/3,num_cols/3))
    plot = sns.heatmap(time_corr_df.transpose(), cmap=cmap, cbar=include_cbar, vmax=1, center=0,
                square=False, linewidths=0, annot=include_annot, annot_kws={"size": 7}, 
                cbar_kws={"shrink":1})
    if show:
        plt.show()
    return plot


def good_corr(orig_df, feature, comp_var='Price', corr_cut=0.8, look_back=7):

    good = False
    pearson_coeffs = [];
    for i in range(look_back+1):
        comp_var_sig = orig_df[comp_var][i:len(orig_df[comp_var])+1]
        feature_sig = feature[0:len(feature)-i]
        pearson_coeffs.append( pearson(comp_var_sig,feature_sig) )
    good = np.max(np.abs(pearson_coeffs)) >= corr_cut

    return good


def cumtrapz_ext(signal):

    return np.hstack((0,sp.integrate.cumtrapz(signal)))


def diff_ext(signal):

    return np.hstack((0,np.diff(signal)))


def series_ma(signal):

    return signal.rolling(12,center=True).mean().interpolate(limit_direction='both', method='linear')

# Construct new features with function transformations
def new_features_with_funcs(orig_df, funcs, func_names, comp_var='Price', filter=True, corr_cut=0, look_back=7, exclude_comp_var=False):
    """Create new feature columns out of old feature columns by applying transforming functions
     
    Keyword arguments:
    orig_df: pandas dataframe
    funcs: list of functions to apply to data
	func_names: list of names of functions to apply to data (for naming new columns)
    comp_var: feature of interest, to compare all other features to
    """

    df = orig_df.copy()
    col_names = df.columns.values.tolist()
    if exclude_comp_var:
        col_names.remove(comp_var)

    new_feats, thrown_out = 0, 0

    for col_name in col_names:
        for i in range(len(funcs)):
            new_feature = funcs[i](df[col_name])
            if filter == True:
                if good_corr(df, new_feature, comp_var=comp_var, corr_cut=corr_cut, look_back=look_back):
                    df[col_name+func_names[i]] = funcs[i](df[col_name])
                    new_feats +=1
                else:
                    thrown_out += 1
            else:
                df[col_name+func_names[i]] = funcs[i](df[col_name])
                new_feats +=1
    print(str(new_feats)+' net new features created with functions. '+str(thrown_out)+' new features were thrown out due to poor correlation with price at all month offsets from 0 to '+ str(look_back))
    print('{} total features remaining'.format(df.shape[1]-1))
    return df


# Construct new features with combination
def new_features_with_combs(orig_df, combiners=['*','/','+','-'], n_combiners=5, comp_var='Price', filter=True, corr_cut=0.8, look_back=7, exclude_comp_var=False):
    """Create new feature columns out of old feature columns by combining features with multiplication and division.
    If a new feature is poorly correlated with price (incl any time shift up to look_back months), throw it out.
     
    Keyword arguments:
    orig_df: pandas dataframe
    comp_var: name of feature to try out correlations of new features
    corr_cut: Pearson correlation with comp_var needs to be above this value to qualify
    look_back: check Pearson correlation with comp_var offsetting up to this many months in the past
    """

    df = orig_df.copy()
    col_names = df.columns.values.tolist()
    if exclude_comp_var:
        col_names.remove(comp_var)

    new_feats = {}
    combinations = [comb for comb in itertools.combinations(col_names, 2)]
    num_new = len(combinations)*n_combiners
    for col1, col2 in combinations:
        os.system('cls')
        if col1+'*'+col2 not in col_names and col1+'/'+col2 not in col_names and col2+'/'+col1 not in col_names:
            for combiner in combiners:
                if combiner == '*':
                    new_feats[col1+'*'+col2] = df[col1].multiply(df[col2])
                if combiner == '/':
                    new_feats[col1+'/'+col2] = df[col1].divide(df[col2])
                    new_feats[col2+'/'+col1] = df[col2].divide(df[col1])
                if combiner =='+':
                    new_feats[col1+'++'+col2] = df[col1]+(df[col2])
                if combiner =='-':
                    new_feats[col1+'--'+col2] = df[col1]+(df[col2])
    if filter == True:
        i = 0
        for key in list(new_feats.keys()):
            i+=1
            print('Testing new feature {0} of {1}'.format(i,num_new), end='         \r')
            if not good_corr(df, new_feats[key], comp_var=comp_var, corr_cut=corr_cut, look_back=look_back):
                new_feats.pop(key) 
    new_feats_df = pd.DataFrame(data=new_feats)
    df = pd.concat([df, new_feats_df], axis=1)
    print(str(len(new_feats))+' net new features created with combinations. '+str(num_new-len(new_feats))+' new features were thrown out due to poor correlation with '+comp_var)
    print('{} total features remaining'.format(df.shape[1]-1))
    return(df)         


# Delete features that are poorly correlated with comp_var at several offsets
def trim_features(orig_df, comp_var='Price', corr_cut=0.7, look_back=7):

    df = orig_df.copy()
    col_names = df.columns.values.tolist()
    col_names.remove(comp_var)

    thrown_out = 0

    for col_name in col_names:
        feature = df[col_name]
        if not good_corr(df, feature, comp_var=comp_var, corr_cut=corr_cut, look_back=look_back):
            df = df.drop((col_name), axis=1)
            thrown_out += 1
    print(str(thrown_out)+' features thrown out due to poor correlation with price at all month offsets from 0 to '+ str(look_back))
    print('{} total features remaining'.format(df.shape[1]-1))
    return df


# Delete features that are highly correlated to other features
def keep_distinct_features(orig_df, comp_var='Price', corr_cut=0.7, look_back=7):
    """Remove features that are highly correlated to one another at all points in time.

    For each feature in the dataframe, check Pearson correlation with all other features with no time offset.
    If the correlation is high, continue checking with look_back time offsets. If all of these correlations are high, 
    add the feature to a list for correlation comparison with comp_var. Keep only the best of that list.

    Keyword arguments:
    orig_df: pandas dataframe
    corr_cut: Pearson correlation above this value means the features are highly correlated.
    look_back: check Pearson correlation with comp_var up to this many months offset to pick best
    """

    df = orig_df.copy()
    col_names = df.columns.values.tolist()
    col_names.remove(comp_var)
    n, m, thrown_out, nans = 0, 0, 0, 0

    for col_name1 in col_names:
        os.system('cls')
        n += 1
        names = df.columns.values.tolist()
        num = len(names)
        names.remove(comp_var)
        if col_name1 not in names:
            continue
        bundle = [col_name1]
        print('Checking feature {0} of {1} remaining '.format(n, num), end='                               \r' )
        for col_name2 in names:
            if col_name1 != col_name2:
                corr = [pearson(df[col_name1],df[col_name2])]
                if np.isnan(corr[0]):
                    nans += 1
                    print(col_name1)
                    print(col_name2)
                if abs(corr[0]) >= corr_cut:
                    bundle.append(col_name2)
        # Keep signal with highest correlation to 'comp_var', looking back 'look_back' months
        if len(bundle) > 1:
            comp_var_corr = []
            for col_name in bundle:
                best_corr = 0
                if look_back >= 1:
                    for i in range(1,look_back):
                        comp_var_sig = df[comp_var][i:len(df[comp_var])+1]
                        feat_sig = df[col_name][0:len(df[col_name])-i]
                        corr = pearson(comp_var_sig,feat_sig)
                        if abs(corr) >= best_corr:
                            best_corr = corr
                comp_var_corr.append(best_corr)
            max_index = comp_var_corr.index(max(comp_var_corr))
            #print('Matching:')
            #print(bundle)
            bundle.pop(max_index)
            df = df.drop(labels=bundle, axis=1)
            thrown_out += len(bundle)
            #print('Removed:')
            #print(bundle)

    print(str(thrown_out)+' features thrown out due to correlation with other features. '+str(nans)+' NaNs resulted from Pearson calculations')
    print('{} total features remaining'.format(df.shape[1]-1))
    return df


# Manipulate dataframe to make time offsets real features (columns) for training
def slide_df(orig_df, offset, look_back, purpose='train', comp_var='Price'):
# new dataframe to learn on, composed of look_back slid rows of a slice starting i months in the past
    orig_df_reind = orig_df.reset_index(drop=True)
    new_df_array = np.ndarray((1,1))
    orig_num_rows = orig_df_reind.shape[0]
    if purpose == 'train':
        new_num_rows = orig_num_rows + 1 - offset - look_back # how many rows the dataframe will have (maximum available... -i for prediction offset, -look_back for equal sliding of rows
    elif purpose == 'test':
        new_num_rows = 1
    new_df_colnames = [comp_var] + [name+str(-offset-k) for k in range(look_back) for name in orig_df_reind.columns.values.tolist() ]
    for j in range(new_num_rows):
        for k in range(look_back):
            if k == 0:
                new_df_row = np.array(orig_df_reind[comp_var][j])
            new_df_row = np.hstack((new_df_row, orig_df_reind.iloc[offset+j+k])) # slide rows horizontally
        if j == 0:
            new_df_array = new_df_row
        else:
            new_df_array = np.vstack((new_df_array, new_df_row)) # stack new rows
    if new_df_array.ndim == 1:
        new_df = pd.DataFrame(columns=new_df_colnames)
        new_df.loc[0] = new_df_array
    else:
        new_df = pd.DataFrame(data=new_df_array, columns=new_df_colnames) 
    return new_df


def avg_percent_error(y_hat, y):
    return 100*np.mean(np.abs((y_hat-y)/y))


def rmse(y_hat, y):
    return math.sqrt(mean_squared_error(y_hat,y))


def within_percent(a, b, percent):
    """Returns true if a is within some percent of b"""

    return percent >= 100*abs(a-b)/b


def df_add_moving_averages(df, window=12):

    orig_names = df.columns.values.tolist()
    ma_names = dict(zip(orig_names,[orig_name+'_MA' for orig_name in orig_names]))
    df_ma = df.rolling(window,center=True).mean().interpolate(limit_direction='both',method='linear').rename(columns=ma_names)
    return pd.concat([df,df_ma],axis=1)


def hyper_param_opt(train_df, offset, interp_min=0, interp_max=100, interp_curves=[1],
    look_back_step=20, look_back_max=60, look_back_min=12, look_back_override=False, test_props=[.2,.15,.1],
    iter=10, model_type=Lasso, alphas=[.001], 
    min_samples=60, min_test_points = 6, comp_var='Price'):
    
    # If look_back were the minimum (1) we would have max_samples
    # We want to have at least min_samples samples to do a train/test split
    max_samples, = len(train_df)-offset-1, # If look_back were the minimum (1) we would have this many samples
    if look_back_override == 'single':
        look_backs = [look_back_min]
    elif look_back_override == 'max_possible':
        look_backs = [max_samples-min_samples]
    elif look_back_override == 'offset':
        look_backs = [offset]
    else:
        look_backs = list(np.flip(np.arange(look_back_min, look_back_max+1, look_back_step), axis=0))
        # look_backs = list(np.arange(look_back_step, len(train_df)+offset, look_back_step))
    num_tests = len(look_backs)*len(test_props)*len(alphas)*len(interp_curves)*iter
    complete, results = 0, []
    for look_back in look_backs:
        slid_df_train = slide_df(train_df, offset, look_back, purpose='train', comp_var=comp_var)
        for interp_curve in interp_curves:
            slid_df_train = df_interpolate(slid_df_train, interp_min, interp_max, curve=interp_curve)
            # slid_df = trim_features(slid_df, look_back=0)
            X = slid_df_train.drop((comp_var), axis=1).values
            y = slid_df_train[comp_var].values
            feat_names = slid_df_train.drop((comp_var), axis=1).columns.values.tolist()
            slid_df_test = slide_df(train_df, 0, look_back, purpose='test', comp_var=comp_var)
            X_pred = slid_df_test.drop((comp_var), axis=1).values
            # print(feat_names)
            for test_prop in test_props:
                if len(y)*test_prop >= min_test_points:
                    for alpha in alphas*iter:
                        model = model_type(alpha=alpha, normalize=True, max_iter=100000)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
                        model.fit(X_train, y_train)
                        train_error = rmse(model.predict(X_train),y_train)
                        test_error = rmse(model.predict(X_test),y_test)
                        # print(list(model.coef_>0))
                        kept_feats = list(itertools.compress(feat_names, list(model.coef_>0)))
                        # print(kept_feats)
                        results.append({'offset':offset,
                                        'look_back':look_back,
                                        'interp_curve':interp_curve,
                                        'len_data':len(y),
                                        'train_prop':np.round(1-test_prop,2),
                                        'train_num':len(y_train),
                                        'test_num':len(y_test),
                                        'neg_alpha':np.round(-alpha, 6),
                                        'num_feats':sum(model.coef_>0),
                                        'kept_feats':'+'.join(kept_feats),
                                        'train_error':train_error,
                                        'test_error':test_error,
                                        'n_iter':model.n_iter_,
                                        'pred':model.predict(X_pred)[0]})
                        # coefs
                        complete += 1
                        print('Offset: '+str(offset)+' look_back: '+str(look_back)+' test_prop: '+str(np.round(test_prop,2))+' alpha: '+str(np.round(alpha,5))+' interp_curve: '+str(interp_curve)+'... '+str(np.round(100*complete/num_tests, 2))+'% complete', end='                                                                         \r')
                else:
                    print(str(len(y)*test_prop//1)+' test points is too few', end='    \r')
                    num_tests -= 1
    hp_df = pd.DataFrame(data=results)
    return hp_df


def train_hopping_lasso(orig_df, comp_var='Price', pred_len=60, iter=50,
    interp_min=0, interp_max=100, interp_curves=3,
    look_back_step=12, look_back_max=60, look_back_min=12, look_back_override=False, test_props=[.2,.15,.1],
    alphas=[.001], model_type=Lasso):

    df = orig_df.copy()

    offsets = list(np.arange(1,pred_len+1))
    offsets_plot = []
    mean_dfs = []
    std_dfs = []
    sample_dfs = []
    best_errors = []
    best_hps = []
    ceof_sets = []

    fig, ax = plt.subplots(1,1)

    for offset in offsets:
        os.system('cls')
        offsets_plot.append(offset)
        full_hp_df = hyper_param_opt(df, offset,
                                interp_min=interp_min,
                                interp_max=interp_max,
                                interp_curves=interp_curves,
                                look_back_step=look_back_step,
                                look_back_max=look_back_max,
                                look_back_min=look_back_min,
                                look_back_override=look_back_override,
                                test_props=test_props,
                                alphas=alphas,
                                iter=iter,
                                model_type=model_type,
                                comp_var=comp_var)
        mean_hp_df = full_hp_df.groupby(['offset','look_back','train_prop','neg_alpha','interp_curve']).mean()
        mean_dfs.append(mean_hp_df)
        std_hp_df = full_hp_df.groupby(['offset','look_back','train_prop','neg_alpha','interp_curve']).std()
        std_dfs.append(std_hp_df)
        sample_df = full_hp_df.groupby(['offset','look_back','train_prop','neg_alpha','interp_curve']).last()
        sample_dfs.append(sample_df)
        min_error = mean_hp_df.test_error.min()
        # max_error = mean_hp_df.test_error.max()
        close_errors = mean_hp_df[within_percent(mean_hp_df.test_error, min_error, 1)]
        best_errors.append(close_errors.test_error[0])
        best_hps.append(close_errors.index[0])

    # print(best_hps)
    # ax.plot(offsets_plot, best_errors)
    # plt.xlabel('Offset')
    # plt.ylabel('RMSE')
    # plt.show()

    mean_df = pd.concat(mean_dfs)
    std_df = pd.concat(std_dfs)
    sample_df = pd.concat(sample_dfs)

    mean_df.columns = [str(col)+'_mean' for col in mean_df.columns]
    std_df.columns = [str(col)+'_std' for col in std_df.columns]
    sample_df.columns = [str(col)+'_sample' for col in sample_df.columns]

    full_df = pd.concat([mean_df, std_df, sample_df], axis=1)
    best_df = full_df.loc[list(best_hps),:]

    return full_df, best_df


def var_name(x):
    try:
        for s, v in list(locals().iteritems()):
            if v is x:
                return s
    except Exception as e:
        print(e)


def pred_plot(all_data, preds_list, start_back=36, pred_len=36, comp_var='Price',
                figsize=(15,7), ylim=[0,20], title='', xlabel='Month', ylabel='Price [cents/kWh]',
                x_zoom=False, labels=[], interp=True, interp_new=5, linewidth=.5):
    
    labels.insert(0, 'Actual')
    plt.close("all")
    plt.figure(figsize=figsize)
    # Plot existing signal
    l = all_data.shape[0]
    x_act = range(l)
    data_act = flip_df(all_data[comp_var])
    if interp:
        x_act = interp_1d(x_act, interp_new)
        data_act = interp_1d(data_act, interp_new)
        plt.plot(x_act, data_act)
    else:
        plt.plot(range(l),flip_df(all_data[comp_var]))
    # Plot predicitions
    rmse_preds_list = []
    for preds in preds_list:
        x = range(l-start_back,l+pred_len-start_back)
        if interp:
            x = interp_1d(x, interp_new)
            preds = interp_1d(preds, interp_new)
            plt.plot(x, preds, linewidth=linewidth)
        else:
            plt.plot(x, preds, linewidth=linewidth)
        rmse_preds_list.append(preds)
    plt.ylim(ylim)
    plt.xlim([0,l-start_back+pred_len])
    if x_zoom:
        plt.xlim([l-start_back, l-start_back+pred_len])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(labels, loc='center left',bbox_to_anchor=(1, 0.5))
    plt.show()

    for i in range(len(rmse_preds_list)):
        try:
            if interp:
                data_act_comp = data_act[-pred_len*(interp_new+1):]
                # print(len(data_act_comp))
                # print(len(rmse_preds_list[i]))
                rmse = math.sqrt(mean_squared_error(np.array(data_act_comp), rmse_preds_list[i]))
            else:
                data_act_comp = data_act[-pred_len:]
                rmse = math.sqrt(mean_squared_error(np.array(data_act_comp), rmse_preds_list[i]))
            print(labels[i+1]+' RMSE: {:.3f} cents'.format(rmse))
        except:
            print(labels[i+1]+' RMSE: No signal')
    labels.remove('Actual')


def interp_1d(data, new):
    x = list(np.arange(0,len(data)))
    f = interpolate.interp1d(x, data, kind='cubic')
    new_x = np.linspace(0,len(data)-1,len(data)*(new+1))
    return f(new_x)




# ------------- John Stuart -------------------


# Splits data either randomly or sequentially
def tt_split(df,test_months=36,split_type="sequence",comp_var='Price'):
    """
    Splits dataset into x and y and train and test based on desired method

    Keyword arguments:
    df: pandas dataframe
    test_months: int, number of months in the desired test set. Default is 36
    split_type: sequence or random train test split
    comp_var: variable to use as Y, Default is 'Price'
    """
    
    X = df.drop([comp_var], axis=1) # Training & Validation data
    Y = df[comp_var] # Response / Target Variable
    
    
    if split_type == "sequence":
        x_train, x_test = X[0:len(X)-test_months], X[len(X)-test_months:]
        y_train, y_test = Y[0:len(Y)-test_months], Y[len(Y)-test_months:]

        print ('Number of samples in training data:',len(x_train))
        print ('Number of samples in validation data:',len(x_test))

        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
        
        
    elif split_type == "random":
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_months/len(Y), random_state=100)

        print ('Number of samples in training data:',len(x_train))
        print ('Number of samples in validation data:',len(x_test))

        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
        
        
    else:
        print("invalid split type")
        
    return x_train, x_test, y_train, y_test   
   
    """
# Takes dataframe and discretizes the price feature. Outputs updated df and the original price as array for later use.
# takes max expected price and lowest price to create range for bins 
def make_price_bands(orig_df,n_bands=25,max_price=25,comp_var='Price'):
Remove features that are highly correlated to one another at all points in time.

    Keyword arguments:
    orig_df: pandas dataframe
    bands = [i for i in xrange(n_bands)]
    
    raw_price = orig_df[comp_var]
    x = [0]*len(orig_df.columns)
    x[-1] = max_price
    orig_df = orig_df.append([0]*len(orig_df.columns))
    orig_df[comp_var].iloc[len(orig_df)-1]=max_price
    orig_df[comp_var] = pd.cut(orig_df[comp_var], n_bands,labels=bands) 
    raw_price_conversion['PriceBand'] = data['PriceBand'].values
    data = data.drop('PriceBand',axis=1)

    
    return new_df, raw_price
    """
    
    
# Eliminate features via recursive feature elimination
def RFE_elim(x_train,y_train,x_test,n_keep=10):
    """Remove features that are highly correlated to one another at all points in time.

    Takes x_train and y_train dataframes and a desired number of features to keep (n_keep) and outputs a ranking of those dataframes as well as a reduced
    dataframe including only the features ranked as the top n_keep features. Note that Price must be discretized beforehand into bands.
    
    Keyword arguments:
    x_train
    y_train
    n_keep = number of features to keep
    """
    
    #Run RFE with linear SVR and 5 feature search
    #from sklearn.datasets import make_friedman1
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    #X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_keep, step=1)
    selector = selector.fit(x_train, y_train)
    print(selector.support_) 
    print(selector.ranking_)
    print(list(x_train.iloc[:,selector.ranking_<5]))
    ranks = selector.ranking_
    red_x_train = x_train[list(x_train.iloc[:,selector.ranking_<25])]
    red_x_test = x_test[list(x_test.iloc[:,selector.ranking_<25])]
    
    return ranks, red_x_train, red_x_test
    
    
    
# Transforms features into a lower dimensional space, checking a range of dimensions to find the dimensionality they perform best in using linear regression    
    
def PCA_transform(x_train,y_train,x_test,y_test,test_dims=15):
    """Remove features that are highly correlated to one another at all points in time.

    Takes a number of dimensions to check and a x/y train/test sets. Then uses the training data to transform the X sets into a lower dimension, then plots
    the respective training predictions, and finally outputs a list of the transformed train and test features in each dimensionality tested as well as the
    corresponding train accuracy and RMSE. Can be run on either continuous or discretized data.

    Keyword arguments:
    x_train
    y_train
    test_dims
    """
    # Feature Extraction with PCA
    import numpy
    from pandas import read_csv
    from sklearn.decomposition import PCA
    from sklearn import linear_model
    import math
    from sklearn.metrics import mean_squared_error

    np.random.seed(3) # set random seed for reproducibility


    #setup true values for plotting
    train_mo = x_train.index.values.astype(int) + 1
    test_mo = x_test.index.values.astype(int) + 1 #+len(X_train.index.values)
    Full_mo=np.concatenate((train_mo,test_mo),axis=0)
    Full_Y = np.concatenate((y_train,y_test),axis=0)



    B_list = list()
    B_test_list = list()
    acc_list = list()
    RMSE_list = list()
    X = x_train.values
    Y = y_train.values
    X_test = x_test.values

    for i in range(2,test_dims,1):
        # feature extraction
        pca = PCA(n_components=i)
        fit = pca.fit(X)
        B = pca.fit_transform(X)
        B_test = np.matmul(X_test,fit.components_.T)

        B_list.append(B)
        B_test_list.append(B_test)
        # summarize components
        print("Explained Variance:",fit.explained_variance_ratio_)

        #get accuracies and RMSE based on linear regression
        linreg = linear_model.LinearRegression() # instantiate
        linreg.fit(B, Y) # fit
        Y_train_pred = linreg.predict(B) # predict training labels
        acc = round(linreg.score(B, Y) * 100, 2) # evaluate
        acc_list.append(acc)
        print('Training Accuracy =', acc, '%')

        # The coefficients
        print('Coefficients:', linreg.coef_)

        # The mean squared error
        RMSE = math.sqrt(mean_squared_error(Y, Y_train_pred))
        RMSE_list.append(RMSE)            
        print("Train RMSE:", RMSE)

        #PLOTTING
        plt.figure(figsize=(9,7))
        plt.scatter(Full_mo, Full_Y,  color='red')
        plt.plot(train_mo, Y_train_pred, color='blue',
                 linewidth=3)
        plt.xlabel('Month (index)')
        plt.ylabel('Average Retail Energy Price, not scaled')
        plt.title('Training Price, labels and estimates')
        plt.xticks(())
        plt.yticks(())

        plt.show()


    return B_list,B_test_list, acc_list, RMSE_list            

