def DCRW(Function,ListOfVars,R,NumGen,PopSize): 

    import numpy as np
    import sympy
    import random
    import matplotlib.pyplot as plt
    from utils import Individual, FunctionEvaluation


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
            
        child1, child2= parent1.Crossover(parent2)
            
        child1=child1.Mutation(R,dim)
            #child2=child2.Mutation(R,dim)
            
        
        ListofVals=child1.Chromosome[0]
        fitness_child1=FunctionEvaluation(f,ListOfVars,np.ndarray.tolist(ListofVals))
            #ListofVals=child2.Chromosome[0]
            #fitness_child2=FunctionEvaluation(f,ListOfVars,np.ndarray.tolist(ListofVals))
            
            
            #DCRW
            
        counter=0;
        contribution_diversity_worse=[]
        contribution_diversity_worse_indices=[]
        for i in range(0,PopSize):
            if fitness_child1<Fitness[0,i]:
                counter=counter+1;
                contribution_diversity_worse=np.hstack([contribution_diversity_worse, Population[i].contribution_diversity(i,Population,PopSize,"Regualr")])
                contribution_diversity_worse_indices=np.hstack([contribution_diversity_worse_indices, i])
            
        if counter==0:   # replace worse
            index=np.argmax(Fitness)
            Population[index]=child1;
            Fitness[0,index]=fitness_child1;
            
            
        if counter!=0:   #Case 1
            index_cmin=np.argmin(contribution_diversity_worse)
            cmin_contribution_diversity=np.min(contribution_diversity_worse)
            index_cmin=contribution_diversity_worse_indices[index_cmin]
            cmin=Population[int(index_cmin)]
                
                #now compare contr_diver of this elemet to the contr_div of the child to the pop (removing cmin from pop)
                #find the con_divversity of child to the pop (where cmin removed frm pop
                                        
            #pop_cmin_removed=Population
            #del pop_cmin_removed[int(index_cmin)]
            contribution_diversity_child = child1.contribution_diversity(index_cmin,Population,PopSize,"cmin-removed")
            #contribution_diversity_child=Population[i].contribution_diversity(PopSize,Population,PopSize)

            if contribution_diversity_child>cmin_contribution_diversity:
                Population[int(index_cmin)]=child1;
                Fitness[0,int(index_cmin)]=fitness_child1
                
            else: #replace worse
                
                index=np.argmax(Fitness)
                Population[index]=child1;
                Fitness[0,index]=fitness_child1;   
                
        

        MinFit[0,n]=np.min(Fitness)
        MaxFit[0,n]=np.max(Fitness)
        AverageFit[0,n]=np.average(Fitness)
            
            
        #if MinFit[0,n]==MaxFit[0,n]:
            #break

    a=np.min(Fitness)
    b=np.ndarray.tolist(Population[np.argmin(Fitness)].Chromosome[0])

    print("The minimim of function  ", f, ' is ', a, 'occured in  ', b)

    plt.clf()
    if dim==1:
        for i in range(0,PopSize):
            plt.plot(Population[i].Chromosome,Fitness[0,i],marker='o',linestyle='',color='r')

        t=np.arange(R[0,0],R[0,1],.01)
        y=np.empty(shape=[1, len(t)]) 
        for i in range(0,len(t)):
            y[0,i]=FunctionEvaluation(f,ListOfVars,np.array([t[i]]))
        plt.plot(t,y[0,:])

        
    return MinFit,MaxFit, AverageFit,a,b,n
        