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

    def next_batch(self,batch_size,steps,return_batch_ts=False):

        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size,1)

        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))

        # Create batch time series on the x axis
        batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution

        # Create the Y data for the time series x axis from previous steps
        y_batch = np.sin(batch_ts)

        # Formatting for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1)

ts_data = TimeSeriesData(250,0,10)

# TIME SERIES VISUALIZATION

# ORIGINAL DATA
# plt.plot(ts_data.x_data,ts_data.y_true)
# plt.show()

num_time_steps =30

y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)
# plt.plot(ts.flatten()[1:],y2.flatten(),'*', label = "Single Training Instance")
# plt.legend()
# plt.tight_layout()
# plt.show()

# Training Data

train_inst = np.linspace(5, 5 + ts_data.resolution*(num_time_steps+1), num_time_steps+1)
# plt.title('A training instance')
# plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]),'bo',markersize=15,alpha=0.5,label="INSTANCE")
# plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),'ko',markersize=7,label='TARGET')
#
# plt.show()

num_inputs =1

num_neurons = 100

num_outputs = 1

learning_rate = 0.001

num_train_iterations = 2000

batch_size = 1

#PLACEHOLDERS

X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])

# RNN CELL LAYER

cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,activation=tf.nn.relu)
#TO GET 1 output
cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=num_outputs)
# WHILE Loop operation to run over the cells
outputs,states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#MEAN SQUARE ERROR AS COST FUNCTION
loss = tf.reduce_mean(tf.square(outputs-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)

init =tf.global_variables_initializer()

#SESSION FOR CREATING MODEL

saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for iteration in range(num_train_iterations):
#
#         X_batch,y_batch = ts_data.next_batch(batch_size,num_time_steps,False)
#
#         sess.run(train,feed_dict={X:X_batch,y:y_batch})
#
#         if iteration%100 ==0:
#             mse =loss.eval(feed_dict={X:X_batch,y:y_batch})
#             print(iteration,"\tMSE",mse)
#
#     saver.save(sess,"./RNN_model/rnn_time_series_model")

# MODEL RESTORE
# with tf.Session() as sess:
#     saver.restore(sess,"./RNN_model/rnn_time_series_model")
#
#     X_new =np.sin(np.array(train_inst[:-1].reshape(-1,num_time_steps,num_inputs)))
#     y_pred = sess.run(outputs, feed_dict={X:X_new})

# plt.title("TESTING THE MODEL")
#
# # TRAINING INSTANCE
# plt.plot(train_inst[:-1],np.sin(train_inst[:-1]),"bo",markersize=15,alpha=0.5,label="Training Instance")
#
# # TARGET TO PREDICT (Correct test values np.sin(train))
# plt.plot(train_inst[1:],np.sin(train_inst[1:]),'ko',markersize=10,label='TARGET')
#
# # Models prediction
# plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize =10,label="Predictions")
#
# plt.xlabel('TIME')
#
# plt.legend()
# plt.tight_layout()
# plt.show()


#GENERATING NEW SEQUENCE

with tf.Session() as sess:
    saver.restore(sess,"./RNN_model/rnn_time_series_model")
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(ts_data.x_data)-num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1,num_time_steps,1)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})

        training_instance.append(y_pred[0,-1,0])

plt.plot(ts_data.x_data,training_instance,'b-')
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps],'r',linewidth=3)
plt.xlabel('TIME')
plt.ylabel('Y')
plt.show()