"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import deepxde as dde
import numpy as np
import tensorflow as tf

import deepxde as dde
import numpy as np
from scipy import *
#tf.compat.v1.enable_eager_execution()

# Define spatial and temporal domains
a0 = 1
Tmax = 18
geom = dde.geometry.Interval(0, a0)
timedomain = dde.geometry.TimeDomain(0, Tmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#num_pred_times = 30  
#pred_times = np.linspace(0, Tmax, num_pred_times)

# Define constant
gf = 9.8
gs = 27.8
e0 = 8.76

# Define the residuals of partial differential equations
def gibson_eqn(x, u, param):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
    
    k_u = param[0] * 10 ** ((u - e0) / param[1])
    s_u = param[2] * 10 ** ((e0 - u) / param[3])
    
    dk_du = dde.grad.jacobian(k_u, u, i=0, j=0)
    ds_du = dde.grad.jacobian(s_u, u, i=0, j=0)

    #dk_du = param[0] / param[1] * 10 ** ((u - e0) / param[1])
    #ds_du = - param[2] /param[3] * 10 ** ((e0 - u) / param[3])
    
    #print ("test derivative: ", dde.grad.jacobian(test(u), x, i=0)[1][0])
    #term0 = dk_du*(1 + e0) / (gf * (1 + u)) - k_u * (1+e0) / gf / (1+u) ** 2
    term0 = dde.grad.jacobian(k_u * (1 + e0) / (gf * (1 + u)), u, i=0, j=0)
    
    term1 = u_t
    term2 = dde.grad.jacobian(k_u * (1 + e0)**2 / (gf * (1 + u)) * ds_du * u_x, x, i=0, j=0)
    term3 = (gs - gf) * term0 * u_x
    
    return term1 + term2 + term3

# Initial condition u(x, 0) = e0
ic = dde.icbc.IC(geomtime, lambda x: e0, lambda _, on_initial: on_initial)

# Boundary condition u(a0, t) = e0
bc1 = dde.icbc.DirichletBC(geomtime, lambda x: e0, lambda x, on_boundary: on_boundary and np.isclose(x[0], a0))

# Boundary condition ∂u/∂x (0) = -(gs - gf) / (1 + e0) * du / (d(s(u)))
def boundary_condition(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def custom_bc2(x, u, ux, param):
    s_u = param[2] * 10 ** ((e0 - u) / param[3])
    #ds_du = dde.grad.jacobian(s_u, u, i=0, j=0)
    ds_du = - param[2] / param[3] * 10 ** ((e0 - u) / param[3])
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    return ux[0]  + (gs - gf) / (1 + e0) / ds_du

data1 = np.loadtxt("settlement_data.dat")  # contains [time, displacement]

# Define the observation points of the model (t, settlement)
observe_times = data1[:, 0]
observe_settlement = data1[:, 1]

def calculate_settlement(u_pred, a0):
    # Dynamically get the number of spatial points in u_pred
    num_x = tf.shape(u_pred)[0]  # Dynamically retrieve the size of the tensor
    dx = a0 / tf.cast(num_x + 1, tf.float32)  # Compute the spatial step size
    # Numerical integration using the trapezoidal rule to calculate settlement
    settlement = tf.reduce_sum((e0 - u_pred) * dx)  # Numerical integration over x

    return settlement

def custom_loss(y_true, y_pred, x_input, observe_times):
    # y_true is actual settlement observation data
    # y_pred is the porosity distribution predicted by the model
    # observe_times is the actual observation time
    
    settlements_pred = []
    x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
    observe_times = tf.convert_to_tensor(observe_times, dtype=tf.float32)
    
    def compute_settlement(t_obs):
        # Find the index of the predicted time closest to the observation time
        time_indices = tf.where(tf.abs(x_input[:, 1] - t_obs) <= 0.1)

        # Retrieve the corresponding value from y_pred based on time_indice
        u_pred_at_time = tf.gather(y_pred, time_indices)

        # Calculate the predicted settlement value
        u_pred_at_time_tensor = tf.convert_to_tensor(u_pred_at_time, dtype=tf.float32)
        settlement = calculate_settlement(u_pred_at_time_tensor, a0) 
        return settlement

    # Use tf.map_fn to iterate on observab_times and calculate the settlement value at each observation moment
    settlements_pred = tf.map_fn(compute_settlement, observe_times, dtype=tf.float32)

    # Calculate the loss between the predicted settlement value and the actual observed value
    loss = tf.reduce_mean(tf.square(settlements_pred - y_true))
    return loss

# 在自定义损失函数中输出 y_pred 和 u_pred_at_time
# def custom_loss(y_true, y_pred, pred_times, observe_times):
#     settlements_pred = []

#     for t in observe_times:
#         closest_idx = np.argmin(np.abs(pred_times - t))
#         y_pred = tf.reshape(y_pred, [-1, tf.size(y_pred)])
#         u_pred_at_time = y_pred[:, closest_idx]  # 预测的孔隙比

#         settlement = calculate_settlement(u_pred_at_time, a0)
#         settlements_pred.append(settlement)

#     settlements_pred = tf.convert_to_tensor(settlements_pred, dtype=tf.float32)
#     loss = tf.reduce_mean(tf.square(settlements_pred - y_true))
#     return loss

# def custom_loss(y_true, y_pred, x_input, observe_times):
#     settlements_pred = []

#     # Loop through each observation time
#     for t_obs in observe_times:
#         # Find the indices of the x_input time values within 0.1 of the observation time
#         time_indices = np.where(np.abs(x_input[:, 1] - t_obs) <= 0.25)[0]
        
#         if len(time_indices) > 0:
#             # Get the corresponding predicted u values for those time indices
#             u_pred_at_times = tf.gather(y_pred, time_indices, axis=0)

#             # Calculate the settlement for each predicted time
#             settlement = calculate_settlement(u_pred_at_times, a0)
#             settlements_pred.append(settlement)

#     settlements_pred = tf.convert_to_tensor(settlements_pred, dtype=tf.float32)

#     # Calculate loss between predicted settlements and actual observed settlements
#     loss = tf.reduce_mean(tf.square(settlements_pred - y_true))
#     return loss

# 定义 PointSetBC 用于观测沉降
#observe = dde.icbc.PointSetBC(np.hstack((observe_times.reshape(-1, 1), np.zeros_like(observe_times.reshape(-1, 1)))), observe_settlement.reshape(-1, 1), component=0)

# 训练数据集
#observe = dde.icbc.PointSetBC(np.hstack((data1[:, :1], np.full((data1.shape[0], 1), Tmax))), data1[:, 1:], component=0)

# Define initial parameter values for optimizing processes
k_0 = dde.Variable(1.0e-3)  # k_0 初值
alpha = dde.Variable(1.0)  # α 初值
f_0 = dde.Variable(1.0)   # f_0 初值
beta = dde.Variable(1.0)  # β 初值

# Put the parameters into the list for subsequent inversion
params = [k_0, alpha, f_0, beta]

bc2 = dde.icbc.OperatorBC(geomtime, lambda x, u, ux: custom_bc2(x, u, ux, params), boundary_condition)

# Definitive data
pde = dde.data.TimePDE(geomtime, lambda x, u: gibson_eqn(x, u, params), [ic, bc1, bc2], num_domain=1000, num_boundary=80, num_initial=30)

# Define neural networks
net = dde.maps.FNN([2] + [50] * 3 + [1], "tanh", "Glorot uniform")

# Define model
model = dde.Model(pde, net)

# Compile and train the model
model.compile("adam", lr=0.001, external_trainable_variables=[k_0, alpha, f_0, beta], 
              loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, model.data.train_x, observe_times))  #model.data.train_x y_pred = model.predict(x_input, batch_size=2260)

# # 在训练开始前设置 TensorBoard 日志路径
# log_dir = "logs/"
# writer = tf.summary.create_file_writer(log_dir)

# # 自定义回调函数来记录数据到 TensorBoard
# class TensorBoardCallback(dde.callbacks.Callback):
#     def __init__(self, writer, period=100):
#         super().__init__()
#         self.writer = writer
#         self.period = period
#         self.epoch = 0

#     def on_epoch_end(self):
#         self.epoch += 1
#         if self.epoch % self.period == 0:
#             # 获取某些数据流，例如预测值或损失值
#             with self.writer.as_default():
#                 u_pred = self.model.predict(self.model.data.train_x)
#                 #loss = self.model.evaluate()
#                 # 将数据写入 TensorBoard 日志
#                 #tf.summary.scalar("loss", loss[0], step=self.epoch)
#                 tf.summary.scalar("predicted_u", u_pred.mean(), step=self.epoch)
#                 self.writer.flush()

# 在训练时使用自定义回调
#tensorboard_callback = TensorBoardCallback(writer, period=100)

# class CustomCallback(dde.callbacks.Callback):
#     def __init__(self, filename, period=1000):
#         super().__init__()
#         self.filename = filename
#         self.period = period  # 控制多久记录一次数据
#         self.epoch = 0

#     def on_epoch_end(self):
#         self.epoch += 1
#         if self.epoch % self.period == 0:
#             # 获取当前预测值或模型参数的值
#             u_pred = self.model.predict(self.model.data.train_x)
#             # 记录数据到文件或打印到控制台
#             print(f"Epoch {self.epoch}, u_pred: {self.model.data.train_x}, len: {self.model.data.train_x.shape}")
#             with open(self.filename, "a") as f:
#                 f.write(f"Epoch {self.epoch}, u_pred: {u_pred}\n")

class CustomCallback(dde.callbacks.Callback):
    def __init__(self, filename, x_input, pred_times, observe_times, period=1000):
        super().__init__()
        self.filename = filename
        self.x_input = x_input  # 输入数据
        self.pred_times = pred_times  # 模型预测的时间点
        self.observe_times = observe_times  # 观测的时间点
        self.period = period  # 控制多久记录一次数据
        self.epoch = 0

    def on_epoch_end(self):
        self.epoch += 1
        if self.epoch % self.period == 0:
            # 获取模型预测的孔隙比 u_pred
            y_pred = self.model.predict(self.x_input)
            #y_pred = tf.reshape(y_pred, [-1, tf.size(y_pred)])

            u_pred_at_time = tf.constant([], dtype=tf.float32)
            
            # 根据observe_times动态选择u_pred_at_time
            for t_obs in self.observe_times:
                # 找到与观测时刻最接近的预测时刻的索引
                time_indices = np.where(np.abs(self.x_input[:, 1] - t_obs) <= 0.1)[0]
                y_pred_shape = tf.cast(tf.shape(y_pred)[0], tf.int64)

                for i in time_indices:
                    # 使用 tf.cond 来判断索引 i 是否超过 y_pred 的维度
                    idx = tf.cast(tf.squeeze(i), tf.int64)
                    def add_u_pred():
                        # 去除 y_pred[i] 的多余维度，确保其为一维张量
                        pred_value = tf.squeeze(y_pred[i])
                        pred_value = tf.expand_dims(pred_value, axis=0)
                        return tf.concat([u_pred_at_time, y_pred[i]], axis=0)

                    def no_op():
                        return u_pred_at_time

                    u_pred_at_time = tf.cond(idx < y_pred_shape, add_u_pred, no_op)

                with tf.compat.v1.Session() as sess:
                    u_pred_at_time_np = sess.run(u_pred_at_time)

                    # 根据自定义函数计算沉降量
                    settlement = calculate_settlement(u_pred_at_time, a0)
                    settlements_pred.append(settlement)

            # 将结果转为tensor以便记录
            settlements_pred = tf.convert_to_tensor(settlements_pred, dtype=tf.float32)

            # 打印或记录预测结果
            print(f"Epoch {self.epoch}, u_pred_at_time: {u_pred_at_time_np}, len: {settlements_pred.shape}")
            
            with open(self.filename, "a") as f:
                f.write(f"Epoch {self.epoch}, settlements_pred: {u_pred_at_time}\n")


# 在训练时使用回调
#custom_callback = CustomCallback("data_flow_log.txt", model.data.train_x, pred_times, observe_times, period=1000)

variable = dde.callbacks.VariableValue([k_0, alpha, f_0, beta], period=1000, filename="variables.dat")
losshistory, train_state = model.train(epochs=5000, batch_size=2000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#k_0_pred, alpha_pred, f_0_pred, beta_pred = [p.numpy() for p in params]

# with tf.compat.v1.Session() as sess:
#     # 初始化所有变量
#     sess.run(tf.compat.v1.global_variables_initializer())

#     # 获取并打印 dde.Variable 的值
#     k_0_value = sess.run(params[0].value())
#     alpha_value = sess.run(params[1].value())
#     f_0_value = sess.run(params[2].value())
#     beta_value = sess.run(params[3].value())
    
#     print("k_0 value:", k_0_value)
#     print("alpha value:", alpha_value)
#     print("f_0_value:", f_0_value)
#     print("beta_value:", beta_value)


#print(f"k_0 = {k_0_pred}, α = {alpha_pred}, f_0 = {f_0_pred}, β = {beta_pred}")

er())

#     # 获取并打印 dde.Variable 的值
#     k_0_value = sess.run(params[0].value())
#     alpha_value = sess.run(params[1].value())
#     f_0_value = sess.run(params[2].value())
#     beta_value = sess.run(params[3].value())
    
#     print("k_0 value:", k_0_value)
#     print("alpha value:", alpha_value)
#     print("f_0_value:", f_0_value)
#     print("beta_value:", beta_value)


#print(f"k_0 = {k_0_pred}, α = {alpha_pred}, f_0 = {f_0_pred}, β = {beta_pred}")

