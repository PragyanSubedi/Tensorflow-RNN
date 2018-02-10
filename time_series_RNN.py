import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TimeSeriesData():

    def __init__(self,num_points,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax,num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self,x_series):
        return np.sin(x_series)

    def next_batch(self,batch_size,steps,return_batch_ts):

        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size,1)
        
        # Convert to be on time series

        # Create batch time series on the x axis

        # Create the Y data for the time series x axis from previous steps

        # Formatting for RNN