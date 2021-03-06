class Individual(object):
    
    
    def __init__(self,Chromosome=0,Fitness=0):
        self.Chromosome=Chromosome
        self.Fitness=Fitness
    
    def Create(self,dim,R):   # create chromosome of dim elements in the range of a and b
        import numpy as np
        self.Chromosome=np.empty([1,dim])
        for i in range(0,dim):
            self.Chromosome[0,i]=(R[i,1]-R[i,0])*np.random.rand()+R[i,0]
    
    def Crossover(self,other):
        import numpy as np
        child1=Individual()
        child2=Individual()
        alpha=np.random.rand()
        child1.Chromosome=(alpha)*(self.Chromosome)+(1-alpha)*(other.Chromosome)
        child2.Chromosome=(1-alpha)*(self.Chromosome)+(alpha)*(other.Chromosome)
        return child1, child2
        
    
    def Mutation(self,R,dim):
        import numpy as np
        import random
        
        mutant=self
        d=random.randint(0,dim-1)
        #d=np.random.random_integers(0,dim-1)  #dim_of_mutation=
        mut_range=0.1*(R[d,1]-R[d,0])
        self.Chromosome[0,d]= max(R[d,0]  , min (R[d,1], self.Chromosome[0,d]+mut_range*(np.random.rand()-0.5) )  )
        return self 
    
    def Dissimilarity(self,other):
        import numpy as np
        return np.linalg.norm(self.Chromosome-other.Chromosome)
        
        
    def contribution_diversity(self,index_cmin,Population,PopSize,case):
        import numpy as np
        
        if case=="Regualr":
            cd=np.empty(shape=[1, PopSize]) 
            for i in range(0,PopSize):#1
                #print(i)
                cd[0,i]=self.Dissimilarity(Population[i])
                
                #cd[0,index_of_self]=np.inf
            #cd[0,PopSize]=np.inf
            
        if case=="cmin-removed": 
            cd=np.empty(shape=[1, PopSize-1]) 
            pop_cmin_removed=Population[:]  #deep copy
            #print(int(index_cmin))
            del pop_cmin_removed[int(index_cmin)]
            for i in range(0,PopSize-1):
                cd[0,i]=self.Dissimilarity(pop_cmin_removed[i])           
        return np.min(cd)
    
    
    def __str__(self):
        return str((self.Chromosome,self.Fitness))
    
def FunctionEvaluation(f,ListOfVars,ListOfVals):
        Dim=len(ListOfVars)
        for i in range(0,Dim):
            f=f.subs({ListOfVars[i]:ListOfVals[i]})
        return f