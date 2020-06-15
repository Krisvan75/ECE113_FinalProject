from scipy.io import wavfile
import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Activation, Flatten
from keras.optimizers import Adam
from sklearn import metrics 

def pad_audio(data, fs, T):
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    #print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((np.zeros(shape),
                              data))
        else:
            return np.hstack((np.zeros(shape),
                              data))
    else:
        return data

def mean_pad_audio(data, fs, T):
    N = len(data)
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = data.shape
    # Create the target shape    
    N_pad = N_tar - shape[0]
    #print("Padding with %s seconds of silence" % str(N_pad/fs) )
    shape = (N_pad,) + shape[1:]
    #calculate number to add
    data_power = np.sum(np.abs(data)*np.abs(data))/N
    # Stack only if there is something to append    
    if shape[0] > 0:                
        if len(shape) > 1:
            return np.vstack((np.full(shape, data_power),
                              data))
        else:
            return np.hstack((np.full(shape, data_power),
                              data))
    else:
        return data



def awgn(signal, snr):
    N = len(signal)
    noise = np.random.normal(0,1,N)
    signal_power = np.sum(np.abs(signal)*np.abs(signal))/N
    noise_power = np.sum(np.abs(noise)*np.abs(noise))/N
    scale_factor = 1 #fill this in with the correct expression
    noise = noise * scale_factor
    return noise + signal

def awgn2(signal, snr):
    N = len(signal)
    noise = np.random.normal(0,1,N)
    signal_power = np.sum(np.abs(signal)*np.abs(signal))/N
    noise_power = np.sum(np.abs(noise)*np.abs(noise))/N
    scale_factor = (1/noise_power) * np.sqrt(signal_power/(10**(snr/10)))
    noise = noise * scale_factor
    print("return call")
    print(noise.shape)
    return noise + signal

def moving_avg_function(signal, n):
  N = n
  cumsum=  [0]
  moving_aves = []
  for ind in range(signal.shape[0]):
      #print(ind)
      if ind ==0: 
        moving_aves.append(signal[ind])
        continue
      cumsum.append(cumsum[ind-1] + signal[ind])
      if ind>=N:
          moving_ave = (cumsum[ind] - cumsum[ind-N])/N
          #can do stuff with moving_ave here
          moving_aves.append(moving_ave)
      else:
           moving_aves.append(signal[ind]) #else added, makes the dimensions match; is this the right way to do it though?
  return moving_aves
   
PATH = './Problem_3_data/training_data_3'
audio_files_path = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.wav'))]

# max_length of audiofile
T = 3.5

index = [i for i in range(len(audio_files_path))]
columns = ['data', 'label']
df_train2 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(audio_files_path):
    fs, data = wavfile.read(file_path)
    out_data = pad_audio(data, fs, T)
    label = os.path.dirname(file_path).split("/")[-1]
    df_train2.loc[i] = [out_data, label]
assert(len(audio_files_path)==398)
y = df_train2.iloc[:, 1].values
X = df_train2.iloc[:, :-1].values
X = np.squeeze(X)
X = np.stack(X, axis=0)

labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.1)
# set input dimensions to length of input data you will be feeding into the neural network
input_dim = 154350
model = Sequential()
model.add(Dense(512, input_dim=input_dim, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=10)
_, accuracy = model.evaluate(X_val, y_val, verbose=0)

print("Accuracy on the validation dataset is :", accuracy)

PATH= './Problem_3_data/test_data_3'
# max_length of audiofile
T = 3.5
index = [i for i in range(10)]
columns = ['data']
df_test_3a = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(glob(os.path.join(PATH, '*.wav'))):
    fs, data = wavfile.read(file_path)
    out_data = pad_audio(data, fs, T)
    df_test_3a.loc[i] = [out_data]
    
X_test = df_test_3a.iloc[:, :].values
X_test = np.squeeze(X_test)
X_test = np.stack(X_test, axis=0)
model.predict(X_test) #Predictions without DFT features

input_dim = 154350

model3a = Sequential()
model3a.add(Dense(512, input_dim=input_dim, activation='relu'))
model3a.add(Dense(256, activation='relu'))
model3a.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model3a.add(Dense(1, activation='sigmoid'))
model3a.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3a.fit(np.absolute(np.fft.fft(X_train)), y_train, epochs=5, batch_size=10)
model3a.predict(np.absolute(np.fft.fft(X_test)))
_, accuracy = model3a.evaluate(np.absolute(np.fft.fft(X_val)), y_val, verbose=0)
print("Accuracy on the validation dataset is :", accuracy)

X_val_new1 = np.array([awgn2(row, 5) for row in X_val])
X_val_new2 = np.array([awgn2(row, 0) for row in X_val])
X_val_new3 = np.array([awgn2(row, -5) for row in X_val])

X_test_new1 = np.array([awgn2(row, 5) for row in X_test])
X_test_new2 = np.array([awgn2(row, 0) for row in X_test])
X_test_new3 = np.array([awgn2(row, -5) for row in X_test])

_, accuracy = model3a.evaluate(np.absolute(np.fft.fft(X_val_new1)), y_val, verbose=0)
print("Accuracy on the validation dataset  with SNR of 5 is :", accuracy)

print("Binary classifications on the test set with SNR of 5:")
model3a.predict(np.absolute(np.fft.fft(X_test_new1)))

_, accuracy = model3a.evaluate(np.absolute(np.fft.fft(X_val_new2)), y_val, verbose=0)
print("Accuracy on the validation dataset  with SNR of 0 is :", accuracy)

print("Binary classifications on the test set with SNR of 0:")
model3a.predict(np.absolute(np.fft.fft(X_test_new2)))

_, accuracy = model3a.evaluate(np.absolute(np.fft.fft(X_val_new3)), y_val, verbose=0)
print("Accuracy on the validation dataset  with SNR of -5 is :", accuracy)

print("Binary classifications on the test set with SNR of -5:")
model3a.predict(np.absolute(np.fft.fft(X_test_new3)))

avg_val_set1 = np.array([moving_avg_function(row, 10) for row in X_val_new1]) #SNR 5
avg_val_set2 = np.array([moving_avg_function(row, 10) for row in X_val_new2]) #SNR 0
avg_val_set3 = np.array([moving_avg_function(row, 10) for row in X_val_new3]) #SNR -5

print(avg_val_set1.shape)

print("Binary classifications on the test set with SNR of 5 (moving average implemented):")
model3a.predict(np.absolute(np.fft.fft(avg_test_set1)))
print("Binary classifications on the test set with SNR of 0 (moving average implemented):")
model3a.predict(np.absolute(np.fft.fft(avg_test_set2)))
print("Binary classifications on the test set with SNR of -5 (moving average implemented):")
model3a.predict(np.absolute(np.fft.fft(avg_test_set3)))

PATH = './Problem_3_data/training_data_3'
audio_files_path = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.wav'))]
# max_length of audiofile
T = 3.5
index = [i for i in range(len(audio_files_path))]
columns = ['data', 'label']
df_train3_mean = pd.DataFrame(index=index, columns=columns)
df_train3_mode = pd.DataFrame(index=index, columns=columns)

assert(len(audio_files_path)==398)
for i, file_path in enumerate(audio_files_path):
    fs, data = wavfile.read(file_path)
    out_data_3c = mean_pad_audio(data, fs, T)
    label = os.path.dirname(file_path).split("/")[-1]
    df_train3_mean.loc[i] = [out_data_3c, label]
y_mean = df_train3_mean.iloc[:,1].values
X_3c_mean = df_train3_mean.iloc[:, :-1].values
X_3c_mean = np.squeeze(X_3c_mean)
X_3c_mean = np.stack(X_3c_mean, axis=0)

for i, file_path in enumerate(audio_files_path):
    fs, data = wavfile.read(file_path)
    out_data_3c_mode = mode_pad_audio(data, fs, T)
    label = os.path.dirname(file_path).split("/")[-1]
    df_train3_mode.loc[i] = [out_data_3c_mode, label]
y_mode = df_train3_mode.iloc[:,1].values #Y values should be the same, no matter how the length is normalized
X_3c_mode = df_train3_mode.iloc[:, :-1].values
X_3c_mode = np.squeeze(X_3c_mode)
X_3c_mode = np.stack(X_3c_mode, axis=0)


#Create test sets with each type of normalization
PATH= './Problem_3_data/test_data_3'
# max_length of audiofile
T = 3.5
index = [i for i in range(10)]
columns = ['data']
df_test_3 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(glob(os.path.join(PATH, '*.wav'))):
    fs, data = wavfile.read(file_path)
    out_data_mean = mean_pad_audio(data, fs, T)
    df_test_3.loc[i] = [out_data_mean]

X_test_mean = df_test_3.iloc[:, :].values
X_test_mean = np.squeeze(X_test_mean)
X_test_mean = np.stack(X_test_mean, axis=0)

T = 3.5
index = [i for i in range(10)]
columns = ['data']
df_test_3 = pd.DataFrame(index=index, columns=columns)
for i, file_path in enumerate(glob(os.path.join(PATH, '*.wav'))):
    fs, data = wavfile.read(file_path)
    out_data_mode = mode_pad_audio(data, fs, T)
    df_test_3.loc[i] = [out_data_mode]

X_test_mode = df_test_3.iloc[:, :].values
X_test_mode = np.squeeze(X_test_mode)
X_test_mode = np.stack(X_test_mode, axis=0)

X_train_mean, X_val_mean, y_train_mean, y_val_mean = train_test_split(X_3c_mean, y_mean, shuffle=True, test_size=0.3)
model_mean = Sequential()
model_mean.add(Dense(512, input_dim=154350, activation='relu'))
model_mean.add(Dense(256, activation='relu'))
model_mean.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model_mean.add(Dense(1, activation='sigmoid'))
model_mean.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_mean.fit(np.absolute(np.fft.fft(X_train_mean)), y_train_mean, epochs=5, batch_size=10)

_, accuracy = model_mean.evaluate(np.absolute(np.fft.fft(X_val_mean)), y_val_mean, verbose=0)
print("Accuracy on the validation dataset with mean normalization is :", accuracy)

print(model_mean.predict(np.absolute(np.fft.fft(X_test_mean))))

X_train_mode, X_val_mode, y_train_mode, y_val_mode = train_test_split(X_3c_mode, y_mode, shuffle=True, test_size=0.3)
model_mode = Sequential()
model_mode.add(Dense(512, input_dim=154350, activation='relu'))
model_mode.add(Dense(256, activation='relu'))
model_mode.add(Dense(128, activation='relu'))
# binary classification, hence just one output unit with sigmoid activation
model_mode.add(Dense(1, activation='sigmoid'))
model_mode.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_mode.fit(np.absolute(np.fft.fft(X_train_mode)), y_train_mode, epochs=5, batch_size=10)

_, accuracy = model_mode.evaluate(np.absolute(np.fft.fft(X_val_mode)), y_val_mode, verbose=0)
print("Accuracy on the validation dataset with mode normalization is :", accuracy)
