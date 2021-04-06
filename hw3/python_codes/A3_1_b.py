import numpy as np

# f(x) = x^2/10 - 2*sin(x)

def f(x):
    return np.e**(-x) - np.cos(x)

def df(x):
    return -np.e**(-x) + np.sin(x)

xl = 0
xr = 1
gap = 1e-5

#golden section
fi = (3 - np.sqrt(5)) / 2

num_iteration = 0
last_update = 0 #0:left; 1:right
while (xr - xl) >= gap:
    if num_iteration == 0:
        new_xl = fi * xr + (1 - fi) * xl
        new_xr = (1 - fi) * xr + fi * xl
    elif last_update == 0:
        new_xl = new_xr
        new_xr = (1 - fi) * xr + fi * xl
    else:
        new_xr = new_xl
        new_xl = fi * xr + (1 - fi) * xl
    
    f_xl = f(new_xl)
    f_xr = f(new_xr)
    
    if f_xl < f_xr:
        xr = new_xr
        last_update = 1
    else:
        xl = new_xl
        last_update = 0
    num_iteration = num_iteration + 1

ans = (xl + xr) / 2
print('------golden section---------')
print('x=', ans)
print('number of iterations:', num_iteration)

# bisection
num_iteration = 0
xl = 0
xr = 1

while (xr - xl) >= gap:
    xm = (xl + xr) / 2
    if df(xm) == 0:
        break
    if df(xm) > 0:
        xr = xm
    else:
        xl = xm
    num_iteration = num_iteration + 1
print('------bisection---------')
print('x=', ans)
print('number of iterations:', num_iteration)


    






    
