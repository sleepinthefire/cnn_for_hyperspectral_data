# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:50:37 2020

@author: mtl98
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class spectrals:
    def __init__(self):
        self.nLayer = 204
        self.pixel_size = 512
        self.bands = [(400 + (600/203) * x) for x in range(204)]

    #specim　IQのデータをnumpy　arrayにして読み込み　なんかそのまま出力するとスペクトルが反転してるから[::-1]とかで反転させると良きかも
    #returnは512,512,204
    def load_specim(self, path):
        wh = 'WHITEREF_'
        dr = 'DARKREF_'
        wd = 'WHITEDARKREF_'
        date = path[-23:-9]
        #numpyだけで頑張ってやってみた　応用性は低いけどspecimenにはこれで十分　BILの配列を読み込めるような3Dキューブに変換
        white = np.fromfile(path + wh + date + '.raw', dtype='int16').reshape(204,512).T.reshape(1, 512, 204)
        dark = np.fromfile(path + dr + date + '.raw', dtype='int16').reshape(204,512).T.reshape(1, 512, 204)
        whitedark = np.fromfile(path + wd + date + '.raw', dtype='int16').reshape(204,512).T.reshape(1, 512, 204)
        raw = np.fromfile(path + date + '.raw', dtype='int16').reshape(512,204,512)
        raw = np.rot90(raw, 1, axes=(1,2))
        
        corrected = np.divide(np.subtract(raw, dark), np.subtract(white, whitedark))
        corrected = np.rot90(corrected, 1, axes=(0, 1))
        corrected = np.rot90(corrected, 2, axes=(1, 2))
        
        return(corrected)
    
            
    #RGBのみのndarrayを返す　rgb_listに読み込みたいレイヤーを入力
    def rgb(self, ndarray, rgb_list=[70,53,19]):
        r = ndarray[:, :, rgb_list[0]].reshape(512,512,1)
        g = ndarray[:, :, rgb_list[1]].reshape(512,512,1)
        b = ndarray[:, :, rgb_list[2]].reshape(512,512,1)
        rgb = np.concatenate([r, g, b], axis=2)
        return rgb
       
        
    #RGB画像の表示 matplotlibを利用する
    def imshow(self, rgb):
        ax = plt.subplot()
        ax.imshow(rgb)
        return ax  
    
    
    #左上と右下を指定してスペクトルの切り取りupper_left, lower_right はリストかタプルで（x,y）のように入力
    def select_region(self, array, upper_left, lower_right):
         (x1, y1) = upper_left
         (x2, y2) = lower_right
        
         width = x2 - x1
         height = y2 - y1
         
         array = array[x1:x1+width, y2:y2+height, :]
         
         return array
 
    def mk_dataset(self, array, glid_length=32, slide=16, minmax=True):
        height = array.shape[0]
        width = array.shape[1]
        
        if height > width:
            height = width
        else:
            width = height
            
        square_array = array[:height, :width, :]
        
        df_list = []
        for row in range((height - glid_length)//slide + 1):
           
            for column in range((width - glid_length)//slide + 1):
               mean_list = []
               
               for layer in range(array.shape[2]):
                   mean = np.mean(square_array[slide*row:slide*row+glid_length, slide*column:slide*column+glid_length, layer])
                   mean_list.append(mean)
               
               if minmax == True:
                   from sklearn import preprocessing
                   mean_list = preprocessing.minmax_scale(mean_list)
                   
               mean_pd = pd.DataFrame(mean_list, columns= [str(row) +'_' + str(column)])
               df_list.append(mean_pd)
        
        dataset = pd.concat(df_list, axis=1)
        print(dataset)
  
        return dataset
    
    #zeroに基準のスペクトルのNPアレイcompareに比較するスペクトルのNPアレイ
    def Sam(self, zero, compare):
        norm_zero = np.sqrt(np.dot(zero, zero))
        norm_compare = np.sqrt(np.dot(compare, compare))
        alpha = np.arccos(np.dot(zero, compare) / (norm_zero*norm_compare) )
        return alpha
    
    
    #平滑化処理用　移動平均　numberに前後の点の数　3点移動方なら１　５点移動方なら２　２の商spectrumは数字のリスト
    def smooth(self, number, spectrum):
        smooth_spectrum = []
        cycle = len(spectrum) - number
        for i in range(number,cycle):
            start = i - number
            stop = i + number
            sum_list = spectrum[start:stop]
            average = sum(sum_list)/(2*number + 1)
            smooth_spectrum.append(average)    
        return smooth_spectrum
    
    #arrayは204,nのshapeで
    def smooth_for_np(self, number, array):
        smooth_array = []
        for i in range(array.shape[1]):
            pixel_spectrum = array[:,i]
            smooth_array.append(self.smooth(number, pixel_spectrum))
        smooth_array = np.array(smooth_array).T    
        return smooth_array
        
    
    
    #抽出する長方形の描画（上塗り）とnumpyarray の取り出し
    def roi(self, path, coordinates, showing=False):
        #length= width and height
        #line0だと何もない　1だと上と左だけ　2で四つ
        def extract_roi(arr, x, y, w, h, intensity=2, line=3):
            #intensity= bouncing box line intensity
            #line= bouncing box line width
            roi = arr[y:y+h, x:x+w, :]
            
            bounding_box = arr 
            bounding_box[y-line:y, x-line:x+w+line, :] = intensity # garis atas, upper line (Indonesia)
            bounding_box[y:y+h, x-line:x, :] = intensity # garis kiri, left line
            bounding_box[y+h:y+h+line, x-line:x+w+line, :] = intensity # garis bawah, donwer line
            bounding_box[y:y+h, x+w:x+w+line, :] = intensity # garis kanan , right line
            
            return (roi, bounding_box)
        
        rois = []#returned ROIs
        bounding_boxed = self.load_specim(path)
        
        for coordinate in coordinates:
            (x, y, w, h) = coordinate
            (roi,  bounding_boxed) = extract_roi(
                     bounding_boxed, x, y, w, h)
            rois.append(roi)
        
        if showing == True:
            self.imshow(bounding_boxed, (70,53,19))
        return rois, bounding_boxed   
    
    def roi_mean_ax(self, rois, coordinates, minmax=False):
        for i in range(len(rois)):
            roi = rois[i]
            intensity = []
            for b in range(roi.shape[2]):
                intensity.append(np.mean(roi[:, :, b]))
            if minmax == True:
                from sklearn import preprocessing
                intensity = preprocessing.minmax_scale(intensity)
            ax = plt.subplot()
            ax.plot(self.bands, intensity, label='{}'.format(coordinates[i]))
        
        #ax.legend(loc='best')
        ax.set_title('Mean in ROI Area')   
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Reflectance')
        return ax
    
    def roi_mean(self, rois, minmax=False):
        for i in range(len(rois)):
            roi = rois[i]
            intensity = []
            for b in range(roi.shape[2]):
                select_band = 203-b
                intensity.append(np.mean(roi[:, :, select_band]))
            if minmax == True:
                from sklearn import preprocessing
                intensity = preprocessing.minmax_scale(intensity)
        return intensity

"""
path = r'D:\Research rocks\2020-08-04_008\capture/'
hs = spectrals()

up = [100, 100]
low = [200,200]

array = hs.select_region(hs.load_specim(path), up, low)
data = hs.mk_dataset(array)
"""
