import keras
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import keract
import pickle

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
        self.lname = 'add_1' #Layer to extract representations from
        self.target=5
        self.vtop=None
        self.c_stds = 2.326 # 98 % interval
        self.thresh_L=None
        self.thresh_H=None
        self.detections=None
    
    def find_target(self):
        X, _ = data_loader(self.pois_data_path)
        y = np.argmax(self.bd_model.predict(X), axis=1)
        self.target = y[np.argmax(np.unique(y, return_counts=True)[1])]

    def explore(self):
        x_clean, y_clean = data_loader(self.clean_data_path)
        x_pois, y_pois = data_loader(self.pois_data_path)
        rep_clean = keract.get_activations(self.bd_model, x_clean, layer_names=self.lname, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)[self.lname]
        rep_pois = keract.get_activations(self.bd_model, x_pois, layer_names=self.lname, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)[self.lname]

        M = rep_pois - rep_clean.mean(axis=0)
        
        u, s, vh = np.linalg.svd(M, full_matrices=False)
        self.vtop = vh[0].transpose()
        cor_pois = np.dot(rep_pois, self.vtop)
        self.thresh_L, self.thresh_H = cor_pois.mean() - self.c_stds * cor_pois.std(), cor_pois.mean() + self.c_stds * cor_pois.std()

    def detect_and_filter(self, X, y):
        rep_val = keract.get_activations(self.bd_model, X, layer_names=self.lname, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)[self.lname]
        cor_val = np.dot(rep_val, self.vtop)
        self.detections = (cor_val > self.thresh_L) & (cor_val < self.thresh_H)
        y[np.intersect1d(np.where(y==self.target), np.where(self.detections==True))] = self.num_classes
        return y
    
    def save_info(self, fname):
        info = {'vtop': self.vtop, 'thresh_L': self.thresh_L, 'thresh_H': self.thresh_H, 'target': self.target}
        pickle.dump(info, open(fname, 'wb'))
    
    def load_info(self, fname):
        info = pickle.load(open(fname, 'rb'))
        self.vtop = info['vtop']
        self.thresh_L = info['thresh_L']
        self.thresh_H = info['thresh_H']
        self.target = info['target']
