import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sampling_freq_audio = 20000
sampling_freq_vibration = 1000

#train_files = ['M1N3_500.pkl', 'M2N3_500.pkl', 'M3N1_500.pkl']

#val_normal_files = ['M1N5_250.pkl', 'M1N6_250.pkl', 'M2N5_250.pkl', 'M2N6_250.pkl', 'M3N2_250.pkl', 'M3N3_250.pkl']
#val_anom_files = ['M1A3_250.pkl', 'M1A4_250.pkl', 'M2A3_250.pkl', 'M2A4_250.pkl', 'M3A1_250.pkl', 'M3A2_250.pkl']

#test_normal_files = ['M1N4_250.pkl', 'M2N4_250.pkl', 'M3N4_250.pkl']
#test_anom_files = ['M1A2_250.pkl', 'M2A2_250.pkl', 'M3A3_250.pkl']

all_files = ['M1N3_500.pkl', 'M2N3_500.pkl', 'M3N1_500.pkl', 'M1N5_250.pkl', 'M1N6_250.pkl', 'M2N5_250.pkl', 'M2N6_250.pkl', 'M3N2_250.pkl', 'M3N3_250.pkl', 'M1A3_250.pkl', 'M1A4_250.pkl', 'M2A3_250.pkl', 'M2A4_250.pkl', 'M3A1_250.pkl', 'M3A2_250.pkl', 'M1N4_250.pkl', 'M2N4_250.pkl', 'M3N4_250.pkl', 'M1A2_250.pkl', 'M2A2_250.pkl', 'M3A3_250.pkl']

data_list = []
files = all_files #change file name array between one of above


def load_pkl(file_name):
    with open(file_name, 'rb') as pk:
        try:
            lists = pickle.load(pk)

            return lists
        except EOFError:
            pk.close()

            return None

def calc_time(sample, freq):
    return np.linspace( 
        0, 
        float(len(sample)) / freq, 
        num = len(sample) 
    ) 

# Define window size and overlap
def stft(signal, win_size, hop_size, freq):
    # Divide o sinal em janelas
    n_frames = 1 + int((len(signal) - win_size) / hop_size)
    frames = np.lib.stride_tricks.as_strided(signal, shape=(win_size, n_frames), strides=(signal.itemsize, hop_size*signal.itemsize)).T

    # Aplica a transformada de Fourier em cada janela
    stft = np.fft.fft(frames, axis=1)
    freqs = np.fft.fftfreq(win_size, d=1/freq)

    return stft, freqs

# Process each file
for i in range(len(files)):
    file_pkl = '{}'.format(files[i])
    data_list.insert(i, load_pkl(file_pkl))

subdirectory = '/Users/ricardomota/Códigos/anomaly_detection_motors/data/images'
for i in range(len(data_list)):
    # process each sample
    for idx, sample_dict in enumerate(data_list[i]):
        noise_eng_fft, noise_eng_freqs = stft(sample_dict['noise_eng'].copy(), 200, 2, sampling_freq_audio)
        plt.imshow(np.abs(noise_eng_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['noise_eng'].copy()[1:])/(sampling_freq_audio), 0, sampling_freq_audio/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_NENG_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(noise_eng_fft.T)[1:int(200/2)], cmap='gray_r')
        
        noise_amb_fft, noise_amb_freqs = stft(sample_dict['noise_amb'].copy(), 200, 2, sampling_freq_audio)
        plt.imshow(np.abs(noise_amb_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['noise_amb'].copy()[1:])/(sampling_freq_audio), 0, sampling_freq_audio/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_NAMB_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(noise_amb_fft.T)[1:int(200/2)], cmap='gray_r')

        gyr_x_fft, gyr_x_freqs = stft(sample_dict['gyr_x'].copy(), 200, 2, sampling_freq_vibration)
        plt.imshow(np.abs(gyr_x_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['gyr_x'].copy()[1:])/(sampling_freq_vibration), 0, sampling_freq_vibration/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_GX_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(gyr_x_fft.T)[1:int(200/2)], cmap='gray_r')

        gyr_y_fft, gyr_y_freqs = stft(sample_dict['gyr_y'].copy(), 200, 2, sampling_freq_vibration)
        plt.imshow(np.abs(gyr_y_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['gyr_y'].copy()[1:])/(sampling_freq_vibration), 0, sampling_freq_vibration/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_GY_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(gyr_y_fft.T)[1:int(200/2)], cmap='gray_r')

        gyr_z_fft, gyr_z_freqs = stft(sample_dict['gyr_z'].copy(), 200, 2, sampling_freq_vibration)
        plt.imshow(np.abs(gyr_z_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['gyr_z'].copy()[1:])/(sampling_freq_vibration), 0, sampling_freq_vibration/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_GZ_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(gyr_z_fft.T)[1:int(200/2)], cmap='gray_r')

        acc_x_fft, acc_x_freqs = stft(sample_dict['acc_x'].copy(), 200, 2, sampling_freq_vibration)
        plt.imshow(np.abs(acc_x_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['acc_x'].copy()[1:])/(sampling_freq_vibration), 0, sampling_freq_vibration/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_AX_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(acc_x_fft.T)[1:int(200/2)], cmap='gray_r')

        acc_y_fft, acc_y_freqs = stft(sample_dict['acc_y'].copy(), 200, 2, sampling_freq_vibration)
        plt.imshow(np.abs(acc_y_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['acc_y'].copy()[1:])/(sampling_freq_vibration), 0, sampling_freq_vibration/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_AY_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(acc_y_fft.T)[1:int(200/2)], cmap='gray_r')

        acc_z_fft, acc_z_freqs = stft(sample_dict['acc_z'].copy(), 200, 2, sampling_freq_vibration)
        plt.imshow(np.abs(acc_z_fft.T)[1:int(200/2)], origin='lower', extent=[0, len(sample_dict['acc_z'].copy()[1:])/(sampling_freq_vibration), 0, sampling_freq_vibration/2], aspect='auto', cmap='gray_r')
        plt.imsave('{}/{}_AZ_{}.png'.format(subdirectory, files[i][:4], idx), arr=np.abs(acc_z_fft.T)[1:int(200/2)], cmap='gray_r')

plt.close()