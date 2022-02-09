# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:09:41 2021

@author: mtl98
"""
import numpy as np
import matplotlib.pyplot as plt


class hyperspectral:
    def __init__(self):
        self.nLayer = 204
        self.pixel_size = 512
        self.bands = [(400 + (600/203) * x) for x in range(204)]
    
    #load HS dat file
    def load_hs(self, path):
        """ 
        load the hyperspectral data from the file(specim IQ only) 
        
        Load teh file and reshape it as hyperspectral cube
        
        Parameters
        ----------------------------------
        path : str of the file path. Choose dat file in "results" folder
        
        Examples
        ----------------------------------------
        >>>hs_cube = load_hs('c:/users/download/ *** /results/REFLECTANCE_**.dat')
        
        """
        raw_data = np.fromfile(path, dtype='float32')
        hs_cube = raw_data.reshape(512, 204, 512)
        hs_cube = hs_cube.transpose((2, 0, 1))
        hs_cube = hs_cube[:, ::-1, :] #Shape of hs_cube is (height, width, bands)
        return hs_cube

    #Normalization(min-max)
    def normalize_cube(self, hs_cube):
        """
        This function is for yperspectaral preprocessing.
        Input hs_cube data will be normalized( min-0, max-1)

        Parameters
        ----------
        hs_cube : Numpy array 
            hs_cube should have 3 axis (Height, Width, Bands).

        Returns
        -------
        normalized_cube : Numpy array 
            hs_cube normalized.

        """
        
        max_array = hs_cube.max(axis=2).reshape(hs_cube.shape[0],hs_cube.shape[1],1)
        min_array = hs_cube.min(axis=2).reshape(hs_cube.shape[0],hs_cube.shape[1],1)
        normalized_cube = (hs_cube - min_array) / (max_array - min_array)
        return normalized_cube

    #Standardization
    def standardize_cube(self, hs_cube):
        """
        This function is for yperspectaral preprocessing.
        Input hs_cube data will be standardized(mean-0, variance-1)

        Parameters
        ----------
        hs_cube : Numpy array 
            hs_cube should have 3 axis (Height, Width, Bands).

        Returns
        -------
        standardized_cube : Numpy array 
            hs_cube standardized.

        """
        mean_array = hs_cube.mean(axis=2).reshape(hs_cube.shape[0],hs_cube.shape[1],1)
        std_array = hs_cube.std(axis=2).reshape(hs_cube.shape[0],hs_cube.shape[1],1)
        standardized_cube = (hs_cube - mean_array) / std_array
        return standardized_cube

    def mk_testdata(self, array, kernel=1):
        """ 
        make testdata from input array
        
        The shape of input array will be suited for test data.
        For example, shape (512,512,204) -> shape (512*512, 204, 1)
        
        Parameters
        -------------------------------
        x : Numpy array of HS data. Must have 3 dims(H,W,C).
        
        Examples
        -------------------------------
        >>>array = np.random.randn(512*512*204).reshape(512, 512, 204)
        mk_testdata(array)
        print(array.shape)
        (512*512, 204)   

        """
        if kernel==1:
            test_data = array.reshape(array.shape[0]*array.shape[1], -1, 1)
            return test_data

    def test_result(self, model, test_data):
        """
        Test_data is predicted using pretrained model and, export predict rounded result as csv.
        
        Parameters
        ----------------------
        model : pretrained ML model for HS data. Input shape is (204,1)
        test_data : numpy array of HS data. The shape (number of data, 204bands, 1)
        
        Examples
        ----------------------
        >>> 
        
        """
        result = model.predict(test_data)
        #result = np.ceil(result, 3)
        return result

    def false_rgb(self, ndarray, bands=[70,53,19]):
        """
        extract 3 bands that corresponds RGB

        Parameters
        ----------
        ndarray : numpy array that have 3 axis(H, W, B)
            hyperspectral data cube
        bands : List or tupple having 3 int value, optional
            The bands that corresponsds RGB. The default is [70,53,19].

        Returns
        -------
        image_array : numpy array
            false rgb array

        """
        image_array = ndarray[:,:,bands]
        return image_array

    def draw_minmax_area(self, np_array):
        """
        draw a graph including average spectrum and minumim and maximum area.        
        
        Parameters
        ----------
        np_array : numpy array that have 3 axis(H, W, B)

        Returns
        -------
        None.

        """
        std = np.std(np_array, axis=(0,1))
        max_line = np.max(np_array, axis=0).reshape(204)
        min_line = np.min(np_array, axis=0).reshape(204)
        avg_line = np.mean(np_array, axis=0).reshape(204)
        plt.fill_between(self.bands, max_line, min_line, alpha=0.5)
        plt.plot(self.bands, avg_line, C='r')
        plt.text(800,0.5,f'std={std}')
        plt.show()
        
    def read_hdr(self, path):
        hdr = ''
        with open(path) as f:
            content = f.read()
            hdr += content
        return hdr
    
    def false_rgb(self, hs_cube, bands=(24, 18, 7)):
        image_array = hs_cube[:,:,bands]
        return image_array

    def draw_rectangle(self, image_array, upperleft, width, height):
        line_width = 4
        color = [1, 0, 0]
        x, y = upperleft

        y_ind, x_ind = np.indices(image_array.shape[:2])
        rectangle_inner = (x < x_ind) & (x_ind < x+width) & (y < y_ind) & (y_ind < y+height)
        rectangle_outer = (x-line_width < x_ind) & (x_ind < x+width+line_width) & (y-line_width < y_ind) & (y_ind < y+height+line_width)
        border_mask = np.logical_and(rectangle_outer, ~rectangle_inner)
        image_array[:,:][border_mask] = color[0]
        #image_array[:,:,1][border_mask] = color[1]
        #image_array[:,:,2][border_mask] = color[2]
        
        return image_array
    
    #extract roi area, area should be tuple or list that have 4 elements x1, y1(upper left), and x2, y2(lower right) 
    def roi_hs(self, hs_cube, area, fig_show=False):
        x1, y1, x2, y2 = area
        width = x2 - x1
        height = y2 - y1
        
        roi = hs_cube[y1:y1+height, x1:x1+width]
        
        if fig_show:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.set_title('image')
            ax1.imshow(self.draw_rectangle(hs_cube, (x1, y1), width, height))
            
            ax2 = fig.add_subplot(122)
            ax2.set_title('roi')
            ax2.imshow(roi)
            fig.show()
        return roi