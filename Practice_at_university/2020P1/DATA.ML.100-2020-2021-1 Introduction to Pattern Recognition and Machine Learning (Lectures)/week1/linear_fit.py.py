# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np

#class to manage mouse event
class dotBuilder:
    def __init__(self, dot):
        self.dot = dot
        self.xs = list(dot.get_xdata())
        self.ys = list(dot.get_ydata())
        self.cid = dot.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.dot.axes: return
        if event.button==1: #鼠标左键点击
            print("my position:" ,event.xdata, event.ydata)
            
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.dot.set_data(self.xs, self.ys)
            
            x2 = np.array(self.xs)
            y2 = np.array(self.ys)
            plot_XY(x2, y2)
            self.dot.figure.canvas.draw()
        elif event.button==3: #鼠标右键点击
            if  event.xdata in self.xs:
                if event.ydata in self.ys:
                    self.xs.remove(event.xdata)
                    self.ys.remove(event.ydata)
                    self.dot.set_data(self.xs, self.ys)
                    
                    x2 = np.array(self.xs)
                    y2 = np.array(self.ys)
                    plot_XY(x2, y2)
                    self.dot.figure.canvas.draw()
                    print(f"clear positions ({event.xdata},{event.ydata}) ")



# Linear solver
def my_linfit(x,y):
    sumx = sum(x);
    sumsqx = sum(np.multiply(x,x));
    sumy = sum(y);
    sumyx = sum(np.multiply(x,y));
    m = len(x)
    b = (sumsqx*sumy-sumyx*sumx)/(-sumx*sumx+m*sumsqx)
    a = (sumyx-b*sumx)/sumsqx
    return a,b

# plot y = ax + b
def plot_XY(x,y):
        length =len(ax.lines)
        if length>1:
            ax.lines.remove(ax.lines[1])
        a,b = my_linfit(x,y)
        xp = np.arange(-2,5,0.1)
        
        ax.plot(xp,a*xp+b,'r-')
        
        print(f"My fit: a={a} and b={b} length ={len(ax.lines)}")
        plt.show()


#Main
x = np.random.uniform(-2,5,10)
x1 = list(x)
y = np.random.uniform(0,3,10)
y1 = list(y)

fig = plt.figure()

ax = fig.add_subplot()

dot, = ax.plot(x, y,'kx')  # empty line
dotbuilder = dotBuilder(dot)
plt.show()

plot_XY(x,y)




