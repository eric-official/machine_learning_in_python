import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

x = np.linspace(-5,5, 50)

y=1.0/(1+np.exp(-x))
y2=(1+np.tanh(x))/2

print(y)

plt.figure(figsize=(10,10))
plt.plot(x,y, lw=5, label="logit")
plt.plot(x,y2, "*", label="tanh")
plt.plot([-5,5],[1,1], "k:")
plt.plot([-5,5],[0,0], "k:")
plt.plot([0,0],[-0.2,1.2], "k-.")
plt.legend()
plt.xlabel("Parameter")
plt.ylabel("Funktionswert")
plt.title("Aktivierungsfunktion")
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.savefig("nicefunktion.png")
plt.show()

iris=datasets.load_iris()
Xiris = iris.data
yiris=iris.target

nr=0
plt.figure(figsize=(15,15))
for i in range(4):
    for k in range(i, 4):
        nr+=1
        plt.subplot(4 ,4, nr)
        plt.scatter(Xiris[:,i], Xiris[:,k], c=yiris, label="Datenpunkt")
        plt.colorbar()
        plt.legend()
plt.show()

dfIris=pd.DataFrame(Xiris)
pd.plotting.scatter_matrix(dfIris, c=yiris, figsize=(15,15))

plt.hist(Xiris[:,0])

plt.boxplot(Xiris)

corr=np.corrcoef(Xiris.T)
print(corr)
plt.matshow(corr, cmap="rainbow")
plt.colorbar()

x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
xx, yy=np.meshgrid(x,y)
p=xx+yy*1j
zz=np.angle((p-1)/(p*p+p+1))
print(zz.shape)
plt.contour(xx,yy,zz)
plt.colorbar()



