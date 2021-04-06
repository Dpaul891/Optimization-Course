import numpy as np
import matplotlib.pyplot as plt
import time

def f1(x):
    return 3 + x[0] + ((1 - x[1]) * x[1] - 2) * x[1]
def f2(x):
    return 3 + x[0] + (x[1] - 3) * x[1]
def f(x):
    return f1(x) * f1(x) + f2(x) *f2(x)
def df(x):
    grad = np.zeros(2).reshape(2,1)
    grad[0] = 2 * f1(x) + 2 * f2(x)
    grad[1] = 2 * f1(x) * (2*x[1] - 3*(x[1]**2) - 2) + 2 * f2(x) * (2*x[1] - 3)
    return grad
def norm(x):
    return np.sqrt(x[0]**2 + x[1]**2)

# exact line search
color_list = ['blue', 'green', 'red', 'cyan', 'purple', 'yellow', 'orange', 'teal',
              'coral', 'darkred', 'brown', 'black']
def plot_contour():
    X = np.arange(-10.2,10.21,0.05)
    Y = np.arange(-2.045,2.55,0.01125)
    X,Y = np.meshgrid(X,Y)
    Z = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = []
            x.append(X[i][j])
            x.append(Y[i][j])
            x = np.array(x)
            Z[i][j] = f(x)
    plt.contourf(X, Y, Z, 20, alpha=0.3, cmap=plt.cm.hot)
    plt.contour(X, Y, Z, 20, colors='grey')
def plot_line(xk_list, subfig_num):
    x = []
    y = []
    for i in range(xk_list.shape[0]):
        x.append(xk_list[i][0][0])
        y.append(xk_list[i][1][0])
    plt.plot(x,y, color = color_list[subfig_num-1], linewidth=1.5)
    plt.scatter(x, y, s=3, color='black')

def gradient_method(initial, subfig_num):
    tol = 1e-8
    xk = initial
    gradient = df(xk)
    num_iteration = 0
    tol_golden = 1e-6
    maxit = 100
    a = 2
    xk_list = []
    xk_list.append(xk)
        
    while norm(gradient) > tol:
        dk = -df(xk)
        alphal = 0
        alphar = a
        num_iteration_golden = 0
        last_update = 0 #0:left; 1:right
        fi = (3 - np.sqrt(5)) / 2
        while (alphar - alphal) >= tol_golden and num_iteration_golden < maxit:
            if num_iteration_golden == 0:
                new_alphal = fi * alphar + (1 - fi) * alphal
                new_alphar = (1 - fi) * alphar + fi * alphal
            elif last_update == 0:
                new_alphal = new_alphar
                new_alphar = (1 - fi) * alphar + fi * alphal
            else:
                new_alphar = new_alphal
                new_alphal = fi * alphar + (1 - fi) * alphal
            
            f_alphal = f(xk + new_alphal*dk)
            f_alphar = f(xk + new_alphar*dk)
            
            if f_alphal < f_alphar:
                alphar = new_alphar
                last_update = 1
            else:
                alphal = new_alphal
                last_update = 0
            num_iteration_golden = num_iteration_golden + 1
        alphak = (alphal + alphar) / 2
        xk = xk + alphak * dk
        xk_list.append(xk)
        gradient = df(xk)
        num_iteration = num_iteration + 1
    
    Number_iterations.append(num_iteration)

    xk_list = np.array(xk_list)
    plot_line(xk_list, subfig_num)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-10.2, 10.2)
    plt.ylim(-2.045, 2.545)
    
x1 = np.arange(-10, 11, 4)
x2 = np.arange(-2, 3, 4)

Number_iterations = []
time_list = []

plt.figure(1, figsize=(10, 5))
plot_contour()
subfig_num = 1
for i in range(6):
    for j in range(2):
        initial = np.zeros(2).reshape(2,1)
        initial[0] = x1[i]
        initial[1] = x2[j]
                
        start = time.clock()
        gradient_method(initial, subfig_num)
        end = time.clock()
        
        time_list.append(end - start)
        subfig_num = subfig_num + 1
plt.savefig('exact', dpi=700)    

print()
print('Number of iterations from different initial points:', Number_iterations)
print('Average number of iterations from different initial points:', 
      sum(Number_iterations)/len(Number_iterations))

print('Calculating time from different initial points:', time_list)
print('Average calculating time from different initial points:', 
      sum(time_list)/len(time_list)) 
       