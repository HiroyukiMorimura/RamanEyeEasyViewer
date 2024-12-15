# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:06:28 2024

@author: hiroy
"""

import numpy as np
import pandas as pd
import streamlit as st

import scipy.signal as signal
import matplotlib.pyplot as plt


from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags
from sklearn.ensemble import RandomForestClassifier

def create_features_labels(spectra, window_size=10):
    # 特徴量とラベルの配列を初期化
    X = []
    y = []
    # スペクトルデータの長さ
    n_points = len(spectra)
    # 人手によるピークラベル、または自動生成コードをここに配置
    peak_labels = np.zeros(n_points)

    # 特徴量とラベルの抽出
    for i in range(window_size, n_points - window_size):
        # 前後の窓サイズのデータを特徴量として使用
        features = spectra[i-window_size:i+window_size+1]
        X.append(features)
        y.append(peak_labels[i])

    return np.array(X), np.array(y)

def find_index(rs_array,  rs_focused):
    '''
    Convert the index of the proximate wavenumber by finding the absolute 
    minimum value of (rs_array - rs_focused)
    
    input
        rs_array: Raman wavenumber
        rs_focused: Index
    output
        index
    '''

    diff = [abs(element - rs_focused) for element in rs_array]
    index = np.argmin(diff)
    return index

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, dssn_th = 0.00001, lambda_=100, porder=1, itermax=30):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn< dssn_th*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def find_peak_width(spectra, first_dev, peak_position, window_size=20):
    """
    Find the peak start/end close the peak position
    Parameters:
    spectra (ndarray): Original spectrum 
    first_dev (ndarray): First derivative of the spectrum 
    peak_position (int): Peak index 
    window_size (int): Window size to find the start/end of the peak 

    Returns:
    local_start_idx/local_end_idx: Start and end of the peaks 
    """

    start_idx = max(peak_position - window_size, 0)
    end_idx   = min(peak_position + window_size, len(first_dev) - 1)
    
    local_start_idx = np.argmax(first_dev[start_idx:end_idx+1]) + start_idx
    local_end_idx   = np.argmin(first_dev[start_idx:end_idx+1]) + start_idx
        
    return local_start_idx, local_end_idx

def find_peak_area(spectra, local_start_idx, local_end_idx):
    """
    Calculate the area of the peaks 

    Parameters:
    spectra (ndarray): Original spectrum 
    local_start_idx (int): Output of the find_peak_width
    local_end_idx (int): Output of the find_peak_width
    
    Returns:
    peak_area (float): Area of the peaks 
    """    
    
    peak_area = np.trapz(spectra[local_start_idx:local_end_idx+1], dx=1)
    
    return peak_area

def main():
    
    savgol_wsize         = 5    # Number windows size of Savitzky-Golay filter
    savgol_order         = 3    # Order for Savitzky-Golay filter (Basically 2 or 3)
    pre_start_wavenum    = 200  # Start of the wavenumber
    pre_end_wavenum      = 3600 # End of the wavenumber
    wavenum_calibration  = -0   # set calibration offset
    Designated_peak_wn   = 1700 # Note: This should be designated by user. This value is just an example. 
    PCA_components       = 2 
    number_line          = 5
    Fsize                = 14   # Fontsize
    
    # アプリのタイトルを設定
    st.title("RamanEye簡易ビューアー")

    # ファイルアップローダーの作成
    uploaded_file = st.file_uploader("CSVファイルを選択してください", type='csv')

    # ファイルがアップロードされた場合の処理
    if uploaded_file is not None:
        
        # CSVファイルを読み込む
        df = pd.read_csv(uploaded_file)

        # DataFrameの行数を取得
        number_of_rows = len(df)
        
        # ユーザーからの入力を受け取る（start_wavenumの入力）
        start_wavenum = st.number_input(f"波数（開始）を入力してください:", 
                                      min_value = 100,
                                      max_value = 3600, 
                                      value = pre_start_wavenum, 
                                      step = 100,
                                      key='unique_number_start_wavenum_key')  # key引数を追加

        # ユーザーからの入力を受け取る（start_wavenumの入力）
        end_wavenum = st.number_input(f"波数（終了）を入力してください:", 
                                      min_value = start_wavenum + 100,
                                      max_value = 3600, 
                                      value = pre_end_wavenum, 
                                      step = 100,
                                      key='unique_number_end_wavenum_key')  # key引数を追加
        

        # ユーザーからの入力を受け取る（行番号の入力）
        number_line = st.number_input(f"行番号を入力してください（1から{number_of_rows-2}までのインデックス）:", 
                                      min_value = 1,
                                      max_value=number_of_rows - 2, 
                                      value = number_of_rows - 2, 
                                      step = 1,
                                      key='unique_number_line_key')  # key引数を追加


        pre_spectra      = np.array(df.iloc[1,:].name[1:])
        
        # Find index based on the start_wavenum and end_wavenum
        start_index      = find_index(pre_spectra, start_wavenum)
        end_index        = find_index(pre_spectra, end_wavenum)
        
        
        # Trim the data according to the start_index and end_index
        spectra          = np.array(df.iloc[number_line + 1,:].name[1:][start_index:end_index+1])
        wavenum          = pre_spectra[start_index:end_index+1] + wavenum_calibration
        
        fig, ax = plt.subplots(figsize=(10, 5)) # 図と軸を生成
        ax.plot(wavenum, spectra, marker='o', linestyle='-', color='b') # 軸にプロット
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize) # X軸のラベル
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize) # Y軸のラベル
        ax.set_title('Raw Spectrum', fontsize=Fsize) # タイトルの設定
        st.pyplot(fig) # Streamlitに図を表示
   
        ##################### Load and the pre-analysis of spectrum ###############
        # Moving Average spectra (Note: This process is required before the baseline removal)
        mveAve_spectra = np.zeros_like(spectra)
        mveAve_spectra = signal.medfilt(spectra, savgol_wsize)

        # Calculate and substruct the baseline
        baseline         = np.zeros_like(mveAve_spectra)
        BSremoval_specta = np.zeros_like(mveAve_spectra)
        # DHQ: the second parameter affects the distance between the baseline and the wave
        baseline = airPLS(mveAve_spectra, 0.00001, 10e1, 2) 
        BSremoval_specta = spectra - baseline

        # Shift all spectra to be non-negative
        BSremoval_specta_pos = BSremoval_specta + abs(np.minimum(BSremoval_specta, 0))

        fig, ax = plt.subplots(figsize=(10, 5)) # 図と軸を生成
        ax.plot(wavenum, BSremoval_specta_pos, marker='o', linestyle='-', color='b') # 軸にプロット
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize) # X軸のラベル
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize) # Y軸のラベル
        ax.set_title('Baseline removal', fontsize=Fsize) # タイトルの設定
        st.pyplot(fig) # Streamlitに図を表示
        
        ############## Smooth the spectra by Savitzky-Golay method  ###############
        # Memo:find_peaks in scipy.signal could work but requires hyperparameter as well.
        firstDev_spectra = np.zeros_like(spectra)
        secondDev_spectra = np.zeros_like(spectra)
        peaks = []
        
        # # デバッグ用
        number_1stDev = 13
        number_2ndDev = 5
        threshold_peaks = 4/1000 
        firstDev_spectra = savitzky_golay(BSremoval_specta_pos, number_1stDev, savgol_order, 1)
        secondDev_spectra = savitzky_golay(BSremoval_specta_pos, number_2ndDev, savgol_order, 2)

        # ピーク検出
        # firstDev_spectra の符号変化と secondDev_spectra の閾値を基にピークを検出
        peak_indices = np.where((firstDev_spectra[:-1] > 0) & (firstDev_spectra[1:] < 0) &
                        ((secondDev_spectra[:-1] / abs(np.min(secondDev_spectra[:-1]))) < -threshold_peaks))[0]
        
        # ピークに対応する波数を抽出
        peaks = wavenum[peak_indices]
        
        st.write("ピーク位置:")
        st.dataframe(peaks)
        
        fig, ax = plt.subplots(figsize=(10, 5)) # 図と軸を生成
        ax.plot(wavenum, BSremoval_specta_pos, marker='o', linestyle='-', color='b') # 軸にプロット
        # ピーク位置に垂直線を描画
        for peak in peaks:
            ax.axvline(x=peak, color='r', linestyle='--', label='Peak at {}'.format(peak))
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize) # X軸のラベル
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize) # Y軸のラベル
        ax.set_title('Baseline Removal', fontsize=Fsize) # タイトルの設定
        st.pyplot(fig) # Streamlitに図を表示
        
if __name__ == "__main__":
    main()
