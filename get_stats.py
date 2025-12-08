import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys

npy_arr_list = ['IF1725176915.8740928.npy',  'IF1725176915.9137866.npy',  'IF1725176915.9261916.npy', 'IF1725176915.8859403.npy' , 'IF1725176915.9171638.npy' , 'IF1725176915.9271429.npy',
'IF1725176915.8965185.npy', 'IF1725176915.920515.npy', 'IF1725176915.9280589.npy',
'IF1725176915.9039097.npy', 'IF1725176915.9225907.npy' , 'IF1725176915.9289.npy' , 'IF1725176915.9101694.npy', 'IF1725176915.9243972.npy','IF1725176915.929948.npy']

def calculate_A(arr,num_bins, scaling_factor = 1):
    arr_new, val_list = np.histogram(arr, num_bins)
    valid_values = []
    valid_freq = []
    print(arr_new)
    for i in range(num_bins):
        if(arr_new[i] >= scaling_factor*np.sum(arr_new)/num_bins):
            valid_values.append(val_list[i])
            valid_freq.append(arr_new[i])

    A = 1 - (float((len(valid_freq) - 1))/float((num_bins - 1)))
    return A

def calculate_kurtosis(arr,num_bins):
    arr_new, _ = np.histogram(arr, num_bins)
    mean_val = np.mean(arr_new)
    std_dev = np.std(arr_new)
    power4 = (arr_new - mean_val)**4
    avgpower4 = np.mean(power4)
    kurtosis = avgpower4/(std_dev**4) - 3
    return kurtosis

def calculate_skewness(arr,num_bins):
    arr_new, val_list = np.histogram(arr, num_bins)
    mean_val = np.mean(arr_new)
    std_dev = np.std(arr_new)
    power3 = (arr_new - mean_val)**3
    avgpower3 = np.mean(power3)
    skewness = avgpower3/(std_dev**3)
    return skewness

def calculate_b(arr,num_bins):
    k = calculate_kurtosis(arr, num_bins)
    g = calculate_skewness(arr,num_bins)
    b = (g**2+1)*(k+3*(num_bins)**2/((num_bins-2)*(num_bins-3))) 
    return b

def calculate_metric(arr, num_bins = 5):
    metric = calculate_A(arr,num_bins)*(calculate_b(arr,num_bins))
    return metric

if __name__ == '__main__':
    for filepath in npy_arr_list:
        arr = np.load(filepath)
        metric = calculate_metric(arr,num_bins = 5)
        print(f" File = {filepath}, Calculated metric = {metric}")
