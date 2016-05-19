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
    
    
    def __str__(self):
        return str((self.Chromosome,self.Fitness))
    
def FunctionEvaluation(f,ListOfVars,ListOfVals):
        Dim=len(ListOfVars)
        for i in range(0,Dim):
            f=f.subs({ListOfVars[i]:ListOfVals[i]})
        return f