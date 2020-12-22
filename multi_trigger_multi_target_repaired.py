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

def first_run_for_info(clean_data_path, pois_data_path, model_path, info_path):
    G1 = repair.Repair(model_path, clean_data_path, pois_data_path)
    G1.find_target()
    G1.explore()
    G1.save_info(info_path)

def main():
    clean_data_path = 'data/clean_test_data.h5'
    model_path = 'models/multi_trigger_multi_target_bd_net.h5'

    pois_data_path1 = 'data/sunglasses_poisoned_data.h5'
    pois_data_path2 = 'data/eyebrows_poisoned_data.h5'
    pois_data_path3 = 'data/lipstick_poisoned_data.h5'

    # first_run_for_info(clean_data_path, pois_data_path1, model_path, 'multi_trig1.dat') #Uncomment like if info not available
    # first_run_for_info(clean_data_path, pois_data_path2, model_path, 'multi_trig2.dat') #Uncomment like if info not available
    # first_run_for_info(clean_data_path, pois_data_path3, model_path, 'multi_trig3.dat') #Uncomment like if info not available
    
    t1 = repair.Repair(model_path, clean_data_path, pois_data_path1)
    t1.load_info('multi_trig1.dat')

    t2 = repair.Repair(model_path, clean_data_path, pois_data_path2)
    t2.load_info('multi_trig2.dat')

    t3 = repair.Repair(model_path, clean_data_path, pois_data_path3)
    t3.load_info('multi_trig3.dat')


    model = keras.models.load_model(model_path)

    input = sys.argv[1]

    if input.endswith('.h5'):
        X, _ = data_loader(input)
    else:
        x = plt.imread(input)
        x = x[:,:,:3]
        X = np.array([x])
    y = np.argmax(model.predict(X), axis=1)
    y_filtered = t1.detect_and_filter(X, y) #Filter for trigger 1
    y_filtered = t2.detect_and_filter(X, y_filtered) # Filter for trigger 2
    y_filtered = t3.detect_and_filter(X, y_filtered) # Filter for trigger 3
    print(y_filtered)
    print(f'Detected {(y_filtered==1283).sum()/y_filtered.shape[0]*100}% as poisoned.')

if __name__ == "__main__":
    main()