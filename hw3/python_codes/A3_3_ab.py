import numpy as np
import matplotlib.pyplot as plt

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
color_list = ['blue', 'green', 'red', 'cyan']
def plot_gradient(y, method, subfig_num):
    plt.figure(1, figsize=(8,10))
    plt.subplot(3,1,subfig_num)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label=method, color = color_list[subfig_num-1], linewidth=2)
    plt.legend()
    plt.xlabel('number of iteration')
    plt.ylabel('$log||gradient||$')
    plt.tight_layout()
    plt.savefig('gradient', dpi=300)
    
def plot_xk(y, method, subfig_num):
    plt.figure(2, figsize=(8,10))
    plt.subplot(3,1,subfig_num)
    n = y.size
    x = np.arange(n)
    y = np.log(y)
    plt.plot(x, y, label = method, color = color_list[subfig_num-1], linewidth=2)
    plt.legend()
    plt.xlabel('number of iteration')
    plt.ylabel('$log||(x^k-x^*)||$')
    plt.tight_layout()
    plt.savefig('xk', dpi=300)
    
#backtracking method
x_star = np.array([-1, 1]).reshape(2, 1)
s = 1
sigma = 0.5
gamma = 0.1
alpha = s
tol = 1e-5
gradient_list = []
xk_list = []

initial = np.array([0, 0]).reshape(2, 1)
xk = initial
gradient = df(xk)
num_iteration = 0
gradient_list.append(norm(gradient))
xk_list.append(norm(xk-x_star))

while norm(gradient) > tol:
    alphak = s
    dk = -df(xk)
    while True:
        if f(xk + alphak*dk) - f(xk) <= gamma * alphak * np.dot(df(xk).T, dk):
            break
        alphak = alphak * sigma
    xk = xk + alphak * dk
    xk_list.append(norm(xk-x_star))
    gradient = df(xk)
    gradient_list.append(norm(gradient))
    num_iteration = num_iteration + 1

xk_list = np.array(xk_list)
gradient_list = np.array(gradient_list)
subfig_num = 1
method = 'backtracking'
plot_gradient(gradient_list, method, subfig_num)
plot_xk(xk_list, method, subfig_num)

print('--------backtracking---------')
print('xk:\n', xk)
print('number of iterations:', num_iteration)
print()

#diminishing step size
def alpha_k(k):
    return 0.01/(np.log(k+2))

num_iteration = 0
k = 1
xk = initial
gradient = df(xk)
gradient_list = []
xk_list = []
gradient_list.append(norm(gradient))
xk_list.append(norm(xk-x_star))

while norm(gradient) > tol:
    dk = -df(xk)
    alphak = alpha_k(k)
    xk = xk + alphak * dk
    xk_list.append(norm(xk-x_star))
    gradient = df(xk)
    gradient_list.append(norm(gradient))
    num_iteration = num_iteration + 1
    k = k + 1
xk_list = np.array(xk_list)
gradient_list = np.array(gradient_list)  
subfig_num = 2      
method = 'diminishing step size'
plot_gradient(gradient_list, method, subfig_num)  
plot_xk(xk_list, method, subfig_num)
print('--------diminishing step size---------')
print('xk:\n', xk)
print('number of iterations:', num_iteration)
print()

# exact line search
tol_golden = 1e-6
maxit = 100
a = 2
num_iteration = 0
xk = initial
gradient = df(xk)
gradient_list = []
xk_list = []
gradient_list.append(norm(gradient))
xk_list.append(norm(xk-x_star))

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
    xk_list.append(norm(xk-x_star))
    gradient = df(xk)
    gradient_list.append(norm(gradient))
    num_iteration = num_iteration + 1
xk_list = np.array(xk_list)
gradient_list = np.array(gradient_list)
subfig_num = 3
method = 'exact line search'
plot_gradient(gradient_list, method, subfig_num)  
plot_xk(xk_list, method, subfig_num)

print('--------eaxct line search---------')
print('xk:\n', xk)
print('number of iterations:', num_iteration)
print()  


