import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import keract
import repair

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data/255.0, y_data

def first_run_for_info(clean_data_path, pois_data_path, model_path):
    G1 = repair.Repair(model_path, clean_data_path, pois_data_path)
    G1.find_target()
    G1.explore()
    G1.save_info('anonymous_2_info.dat')

def main():
    clean_data_path = 'data/clean_test_data.h5'
    pois_data_path = 'data/eyebrows_poisoned_data.h5'
    model_path = 'models/anonymous_2_bd_net.h5'
    
    # first_run_for_info(clean_data_path, pois_data_path, model_path) #Uncomment like if info not available
    
    G1 = repair.Repair(model_path, clean_data_path, pois_data_path)
    G1.load_info('anonymous_2_info.dat')
    model = keras.models.load_model(model_path)

    input = sys.argv[1]
    if input.endswith('.h5'):
        X, _ = data_loader(input)
    else:
        x = plt.imread(input)
        x = x[:,:,:3]
        X = np.array([x])
    y = np.argmax(model.predict(X), axis=1)
    y_filtered = G1.detect_and_filter(X, y)
    print(y_filtered)
    print(f'Detected {(y_filtered==1283).sum()/y_filtered.shape[0]*100}% as poisoned.')

if __name__ == "__main__":
    main()