import numpy as np

# f(x) = x^2/10 - 2*sin(x)

def f(x):
    return x**2/10 - 2*np.sin(x)

xl = 0
xr = 4

fi = (3 - np.sqrt(5)) / 2
gap = 1e-5

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
print('x=', ans)
print('number of iterations:', num_iteration)









    
