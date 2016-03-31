import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from numpy.linalg import inv
def normal_equation(x,y):

	#1- Inverse 
	return inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
	 
	#a = inv(x.transpose().dot(x))
	#c = a.dot(x.transpose())
	#theta = c.dot(y)
	#return theta

def gradient_descent(x,y,theta,alpha, n_iter):
	m = len(y)
	for _ in range(n_iter):
		prediction = x.dot(theta)
		loss = (prediction-y)
		#cost  = (loss**2)/(2*m)
		gradient = (x.transpose().dot(loss))/m
		theta = theta - (alpha*gradient)
		
		#print 'loss:' , loss
		#print '\nCost', sum(cost)
		#print x.transpose(),np.shape(x.transpose())
		#print '\n',loss,np.shape(loss)
		#print '\nGradient:',gradient,np.shape(gradient)
		#print '\nalpha * gradient:',alpha*gradient
		
	return theta

x = [[1,2,3,4],[2,4,7,1],[4,6,8,2],[7,9,6,3],[5,6,5,4],[1,7,8,5],[2,8,5,6],[6,9,4,7],[2,6,4,5],[5,5,3,9],[8,1,6,9],[9,2,3,8],[3,3,9,7],[4,4,0,7],[8,5,2,1],[9,2,5,5],[7,6,6,4],[1,8,5,3],[2,7,7,2],[4,5,1,2],[7,2,1,4],[6,6,4,6],[5,5,5,8],[4,2,6,9],[8,8,2,4],[9,7,4,4],[1,6,3,6],[4,6,3,7],[3,3,5,9],[8,2,7,1]]
y=  [22.7,25.2,36.8,47.4,39.7,44.8,46.9,55.6,37.7,49.9,48.8,44.7,45.7,35.7,27.2,38.4,43.6,36.4,35.2,24.0,26.9,45.6,49.8,46.3,43.4,45.3,38.1,44.8,45.9,27.8]
theta = [0,0,0,0]

print 'Gradient_Descent:',gradient_descent(np.asarray(x),np.asarray(y),np.asarray(theta),0.01,388)
print 'Normal_Equation: ',normal_equation(np.asarray(x),np.asarray(y))



old_min = 0
temp_min = 15
step_size = 0.01
precision = 0.001
 
def f_derivative(x):
    return 2*x -6


while abs(temp_min - old_min) > precision:
    old_min = temp_min 
    gradient = f_derivative(old_min) 
    move = gradient * step_size
    temp_min = old_min - move
    #print ('%.2f'%temp_min,'%.2f'%old_min,'%.2f'%move,'%.2f'%gradient)
 
#print("Local minimum occurs at {}.".format(round(temp_min,2)))













