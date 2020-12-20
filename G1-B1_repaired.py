import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import keract

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))
    return x_data/255.0, y_data

class Repair:

    def __init__(self, model_path, clean_data_path, pois_data_path, num_classes=1283):
        self.num_classes=num_classes
        self.model_path=model_path
        self.bd_model = keras.models.load_model(model_path)
        self.clean_data_path=clean_data_path
        self.pois_data_path=pois_data_path
        self.lname = 'add_1'
        self.target=1
        self.vtop=None
        self.c_stds = 2.576 # 99 % interval
        self.thresh_L=None
        self.thresh_H=None
        self.detections=None

    def explore(self):
        x_clean, y_clean = data_loader(self.clean_data_path)
        x_pois, y_pois = data_loader(self.pois_data_path)
        self.target=y_pois[np.argmax(np.unique(y_pois, return_counts=True)[1])]
        rep_clean = keract.get_activations(self.bd_model, x_clean[np.where(y_clean==self.target)], layer_names=self.lname, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)[self.lname]
        rep_pois = keract.get_activations(self.bd_model, x_pois[np.where(y_clean==self.target)], layer_names=self.lname, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)[self.lname]
        try:
            M = rep_pois - rep_clean
        except:
            M = rep_clean - rep_clean.mean(axis=0)
        
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        self.vtop = vh[0].transpose()
        cor_pois = np.dot(rep_pois, self.vtop)
        self.thresh_L, self.thresh_H = cor_pois.mean() - self.c_stds * cor_pois.std(), cor_pois.mean() + self.c_stds * cor_pois.std()

    def detect_and_filter(self, X, y):
        rep_val = keract.get_activations(self.bd_model, X, layer_names=self.lname, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)[self.lname]
        cor_val = np.dot(rep_val, self.vtop)
        self.detections = (cor_val > self.thresh_L) & (cor_val < self.thresh_H)
        y[np.intersect1d(np.where(y==9), np.where(self.detections==True))] = self.num_classes
        return y



def main():
    clean_data_path = 'data/clean_test_data.h5'
    pois_data_path = 'data/sunglasses_poisoned_data.h5'
    model_path = 'models/sunglasses_bd_net.h5'
    model = keras.models.load_model(model_path)
    G1 = Repair(model_path, clean_data_path, pois_data_path)
    G1.explore()
    input = sys.argv[1]
    X, y = data_loader(input)
    y = np.argmax(model.predict(X), axis=1)
    y_filtered = G1.detect_and_filter(X, y)
    print(y_filtered)

    return y_filtered
    
if __name__ == "__main__":
    main()
