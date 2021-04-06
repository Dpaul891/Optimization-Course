from matplotlib import pyplot as plt
import numpy as np
figure = plt.figure(1)
ax = plt.subplot(111, projection='3d')
X = np.arange(-10,10,0.3)
Y = np.arange(-10,10,0.3)
X,Y = np.meshgrid(X,Y)
Z = X**4 + 2 * (X - Y) * (X**2) + 4 * (Y**2)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='hot', zorder=2, alpha=0.75)
point_x = np.array([0, -2])
point_y = np.array([0, 1])
point_z = np.array([0, -4])
ax.scatter(point_x, point_y, point_z, c='green', s=30, zorder=1)
ax.set_xlabel("$x_1$", fontsize = 13, labelpad = 5)
ax.set_ylabel("$x_2$", fontsize = 13, labelpad = 5)
ax.set_zlabel("$f_3(x)$", fontsize = 13, labelpad = 10)

my_x_ticks = np.arange(-10, 15, 5)
my_y_ticks = np.arange(-10, 15, 5)
my_z_ticks = np.arange(-100, 16000, 3000)

plt.tick_params(labelsize=12)

ax.set_xticks(my_x_ticks)
ax.set_yticks(my_y_ticks)
ax.set_zticks(my_z_ticks)

text = ['A', 'B']
for i in range(len(point_x)):
    ax.text(point_x[i], point_y[i], point_z[i], text[i], fontsize=15, c='green', zorder=3)

plt.savefig('fig1.png', dpi=500)
plt.show()


figure = plt.figure(2)
def f(x, y):
    return x**4 + 2 * (x - y) * (x**2) + 4 * (y**2)
ax2 = plt.subplot(111)
plt.contourf(X, Y, f(X,Y), 8, alpha=0.75, cmap=plt.cm.hot, zorder=1)
ax2.scatter(point_x, point_y, c='green', s=30, zorder=2)

C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.clabel(C, inline=1, fontsize=8)
plt.xticks(my_x_ticks), plt.yticks(my_y_ticks)
ax2.set_xlabel("$x_1$", fontsize = 13, labelpad = 5)
ax2.set_ylabel("$x_2$", fontsize = 13, labelpad = -8)

plt.tick_params(labelsize=12)

for i in range(2):
    plt.annotate(chr(ord('A')+i), xy=(point_x[i], point_y[i]), fontsize=12, xycoords='data',
                 xytext=(30, 30), textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
plt.savefig('fig2.png', dpi=500)
plt.show()