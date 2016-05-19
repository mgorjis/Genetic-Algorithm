import numpy as np
import sympy
import random

from utils import Individual, FunctionEvaluation

#s = str('x1**2+sin(x2)')
#ListOfVars=['x1','x2']
#f = sympy.sympify(s)
Function='x1^2*sin(x2)+cos(x3)'
ListOfVars=['x1','x2','x3']
R=np.array([[1,5],[0,1],[0,2]])
NumGen=50000
PopSize=10


s=str(Function)
f=sympy.sympify(s)
dim=len(ListOfVars)

Population=[]

Fitness=np.empty(shape=[1, PopSize]) 
MinFit=np.empty(shape=[1, NumGen])
MaxFit=np.empty(shape=[1, NumGen])
AverageFit=np.empty(shape=[1, NumGen])
    
for i in range(0,PopSize):
    p=Individual()
    p.Create(dim,R)
    Population.append(p)  #.Chromosome

    ListofVals=p.Chromosome[0]
    Fitness[0,i]=FunctionEvaluation(f,ListOfVars,np.ndarray.tolist(ListofVals))
    
   
for n in range(0,NumGen):
    
    index1=random.randint(0,PopSize-1)
    index2=random.randint(0,PopSize-1)
       
    parent1=Population[index1]
    parent2=Population[index2]
    
    fitness_parent1=Fitness[0,index1]
    fitness_parent2=Fitness[0,index2]
    
    child1, child2 , alpha = parent1.Crossover(parent2)
    
    child1=child1.Mutation(R,dim)
    child2=child2.Mutation(R,dim)
    
   
    ListofVals=child1.Chromosome[0]
    fitness_child1=FunctionEvaluation(f,ListOfVars,np.ndarray.tolist(ListofVals))
    ListofVals=child2.Chromosome[0]
    fitness_child2=FunctionEvaluation(f,ListOfVars,np.ndarray.tolist(ListofVals))
    
    
    #DC
    dp1c1=parent1.Dissimilarity(child1)
    dp1c2=parent1.Dissimilarity(child2)
    dp2c1=parent2.Dissimilarity(child1)
    dp2c2=parent2.Dissimilarity(child2)
    
    if dp1c1+dp2c2<=dp1c2+dp2c1:
#----------------------------------
        if  fitness_child1<=fitness_parent1:
            Population[index1]=child1;
            Fitness[0,index1]=fitness_child1           
        #else:
            # Do nothing
#----------------------------------           
        if  fitness_child2<=fitness_parent2:
            Population[index2]=child2;
            Fitness[0,index2]=fitness_child2   
        #else:
            # Do nothing
            
    
    
    if dp1c1+dp2c2>dp1c2+dp2c1:
#----------------------------------   
        if  fitness_child1<=fitness_parent2:
            Population[index2]=child1;
            Fitness[0,index2]=fitness_child1 
        #else:
            # Do nothing
            
#----------------------------------          
        if  fitness_child2<=fitness_parent1:
            Population[index1]=child2;
            Fitness[0,index1]=fitness_child2 
    
   
    
    
    
    
    MinFit[0,n]=np.min(Fitness)
    MaxFit[0,n]=np.max(Fitness)
    AverageFit[0,n]=np.average(Fitness)
    
    
    
import matplotlib.pyplot as plt

t = np.arange(0.0, NumGen, 1)

plt.plot(t,MinFit[0,:])
plt.plot(t,MaxFit[0,:])
plt.plot(t,AverageFit[0,:])

a=np.min(Fitness)
b=np.ndarray.tolist(Population[np.argmin(Fitness)].Chromosome[0])

print("The minimim of function  ", f, ' is ', a, 'occured in  ', b)