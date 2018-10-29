from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


xs = np.array([1,2,3,4,5], dtype=np.float64) #由於是斜率，可能會有float型態資料
ys = np.array([5,4,6,5,6], dtype=np.float64)

#plot
#plt.scatter(xs ,ys)
#plt.show()
#定義
def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    
    b = mean(ys) - m * mean(xs)
    return m,b

#給定 m 與 b
m ,b = best_fit_slope(xs,ys)
#print(m ,b) #斜率與節距

regression_line = []

for x in xs:
    regression_line.append((m*x)+b)

plt.scatter(xs ,ys ,color = '#003F72')
# y = m*x + b，這邊將畫出input = x ， output = y的圖
plt.plot(xs ,regression_line)
#plt.show()


#R^2：判斷Regression的合適度
#(y - y^)總和
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

#公式推導
def coefficient_of_determination(ys_orig,ys_line):
    
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

