#This file contains 8 evolutionary algorithms

# 2 deterministic models (individualistic ou communautary) both with Integer and with percentages :  EvolDeterInt EvolDeterReal (ex M2)
 
# 2 stochastic models where we choose 2 individuals randomly and where we increase population depending of the scores : EvolEncounter (ex M3)

# 2 stochastic models based on Moran process, where a fitness is computed for each individual, and where we use this fitness as a probability to choose one individual on which we increase or decrease by 1, the population: EvolMoran (ex M4)

# 2 stochastic models based on Fermi process where we choose 2 individuals based on their fitness, then they make a meeting, and we increase or decrease the population depending of the meeting result:"EvolFermi" (M5)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#class M2Reels :
class EvolDeterReal:
    def __init__(self, tournament, prefix, population, itermax, algo, resilience=0, diag=0) :
        nbstrats = len(tournament.strategies)
        self.nomstrats=tournament.matrix.index.values
        self.scores = tournament.matrix.iloc[0:nbstrats,0:nbstrats].values.copy()
        self.population=np.array(population)
        self.prefix = prefix
        self.itermax=itermax
        self.resilience=resilience
        self.algo=algo
        if (algo != 'com' and algo != 'ind'):   # m2sr=com(munautary)  m2=ind(ividualistic)
            raise ValueError("Algo inconnu :",algo)
        if (algo=='com'):
            np.fill_diagonal(self.scores,diag)  # A MULTIPLIER PAR LONGUEUR
        self.historic=[population.copy()]
    def run(self) :
        ident=False
        i=1
        while ident==False and i<self.itermax :
            fitness = self.population*np.dot(self.scores,self.population)
            #self.population = fitness/sum(fitness)
            self.population = (self.resilience*self.population) + ((1-self.resilience)*fitness/sum(fitness))
            self.historic.append(self.population.copy())
            # calcul de l'arret
            diff=self.historic[-1] - self.historic[-2]
            ident = list(filter(lambda x : abs(x)>(1/100000.0) , diff)) == []
            i=i+1
    def plot(self) :
        df = pd.DataFrame(self.historic, columns=self.nomstrats)
        df.plot(grid=True) # , title="M2Reels "+self.algo , figsize=(10,6))  # Defaut figsize=(6,4)
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')

        tmillis = int(round(time.time() * 1000))    
        plt.savefig(self.prefix+'_M2Reel_'+self.algo+'_res'+str(self.resilience)+'_'+str(tmillis), dpi=500)
        plt.show()
        plt.close()

        
#class M2Entiers :
class EvolDeterInt:
    def __init__(self, tournament, prefix, population, itermax, algo, strict=True , resilience=0 , diag=0) :
        self.tournament=tournament
        nbstrats = len(tournament.strategies)
        self.nomstrats=tournament.matrix.index.values
        self.scores = tournament.matrix.iloc[0:nbstrats,0:nbstrats].values.copy()
        self.population=np.array(population)
        self.base=sum(population)
        self.itermax=itermax
        self.resilience=resilience
        self.cooperationList=[] # decalage de 1 par rapport aux pops
        self.cooperationMatrix=tournament.cooperations.iloc[0:nbstrats,0:nbstrats].values
        self.strict=strict
        self.prefix=prefix
        self.algo=algo
        if (algo != 'com' and algo != 'ind'):   # m2sr=com(munautary)  m2=ind(ividualistic)
            raise ValueError("Algo inconnu :",algo)
        if (algo=='com'):
            np.fill_diagonal(self.scores,diag)    # A MULTIPLIER PAR LONGUEUER
        self.historic=[population.copy()]
    def run(self) :
        identiques=False
        i=1
        while identiques==False and i<self.itermax :
            nbSurvivants = np.count_nonzero(self.population)
             
            # le coeur du calcul de l'evolution
            if self.strict==True:
                fitness = np.dot(self.scores,self.population)
                # On enleve le score que j'ai obtenu contre moi-même
                fitness -= np.diag(self.scores)
                fitness = self.population * fitness              
            else :
                # version simple pour la pop
                fitness = self.population*np.dot(self.scores,self.population)
            
            # Le If est là pour éviter avec com (m2sr) une div par zero quand il ne reste qu'une famille
            # car quand il ne reste qu'une famille la somme du fitness est nulle
            if (nbSurvivants > 1) :
                #self.population = (self.base*fitness)//sum(fitness)
                self.population = (self.resilience*self.population) + ((1-self.resilience)*(self.base*fitness))//sum(fitness)
                
            self.historic.append(self.population.copy())
            # calcul cooperation avec la nouvelle pop
                       
            coops = np.dot(self.cooperationMatrix, self.population)
            if self.strict==True:
                coops -= np.diag(self.cooperationMatrix)
            #for j in range(self.cooperationMatrix.shape[0]) : coops[j] -=self.cooperationMatrix[j,j]
            totalCooperations = sum(self.population*coops)
            
            # version simple cooperation
            # totalCooperations = sum(self.population*np.dot(self.cooperationMatrix, self.population))

            effectifexact = self.population.sum()
            if self.strict==True:
                pcoop = 100 * totalCooperations / (effectifexact * (effectifexact-1) * self.tournament.length)
            else:
                pcoop = 100 * totalCooperations / (effectifexact * effectifexact * self.tournament.length)
                               
            # assert pcoop <= 100 , [s.name for s in bag]
            self.cooperationList.append(pcoop)
            
            # calcul de l'arret : 2 pop consecutives identiques
            identiques = np.all(self.historic[-1] == self.historic[-2])
            identiques = identiques or (nbSurvivants==1) # 1 seul non nul (pb avec com (m2sr))
            i=i+1
    def plot(self) :
        df = pd.DataFrame(self.historic, columns=self.nomstrats)
        df.plot(grid=True) # , title="M2Entier "+self.algo , figsize=(10,6))  # Defaut figsize=(6,4)
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')

        tmillis = int(round(time.time() * 1000))    
        plt.savefig(self.prefix+'_M2Entier_'+self.algo+'_res'+str(self.resilience)+'_'+str(tmillis)+".png"  , dpi=500)
        plt.show()
        plt.close()
