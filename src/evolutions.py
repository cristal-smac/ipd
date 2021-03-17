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
    def __init__(self, algo, tournament, population, itermax, prefix, resilience=0, diag=0) :
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
        self.historic=[self.population.copy()]
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
    def drawPlot(self) :
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
    def __init__(self, algo, tournament, population, itermax, prefix, strict=True , resilience=0 , diag=0) :
        nbstrats = len(tournament.strategies)
        self.tournament=tournament
        self.nomstrats=tournament.matrix.index.values
        self.scores = tournament.matrix.iloc[0:nbstrats,0:nbstrats].values.copy()
        # soit c'est la meme pop pour tout le monde, soit c'est un tableau
        if (type(population)==int) :
            self.population=np.array([population]*nbstrats  , dtype=int)
        else :
            self.population=np.array(population, dtype=int)    # un npArray pour pouvoir faire /total
        self.base=sum(self.population)
        self.itermax=itermax
        self.resilience=resilience
        self.cooperationList=[] # decalage de 1 par rapport aux pops
        self.cooperationMatrix=tournament.cooperations.iloc[0:nbstrats,0:nbstrats].values
        self.strict=strict
        self.prefix=prefix
        self.algo=algo
        if (algo != 'com' and algo != 'ind'):   # com(munautary) ex m2sr ;   ind(ividualistic) ex m2
            raise ValueError("Algo inconnu :",algo)
        if (algo=='com'):
            np.fill_diagonal(self.scores,diag)    # A MULTIPLIER PAR LONGUEUER
        self.historic=[self.population.copy()]
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
    def drawPlot(self) :
        df = pd.DataFrame(self.historic, columns=self.nomstrats)
        df.plot(grid=True) # , title="M2Entier "+self.algo , figsize=(10,6))  # Defaut figsize=(6,4)
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')

        tmillis = int(round(time.time() * 1000))    
        plt.savefig(self.prefix+'_M2Entier_'+self.algo+'_res'+str(self.resilience)+'_'+str(tmillis)+".png"  , dpi=500)
        plt.show()
        plt.close()



#Ex M3
class EvolEncounter:
    def __init__(self, algo, tournament, population, itermax, prefix=''):
        self.nbstrats = len(tournament.strategies)
        self.idxstrats=list(range(self.nbstrats))
        self.nomstrats=tournament.matrix.index.values
        self.scores = tournament.matrix.iloc[0:self.nbstrats,0:self.nbstrats].values.copy()
        # soit c'est la meme pop pour tout le monde, soit c'est un tableau
        if (type(population)==int) :
            self.population=np.array([population]*self.nbstrats  , dtype=int)
        else :
            self.population=np.array(population, dtype=int)    # un npArray pour pouvoir faire /total
        self.base=sum(self.population)
        self.itermax=itermax
        self.algo=algo
        self.prefix=prefix
        if (self.algo != 'com' and self.algo != 'ind'):   # com(munautary) ex m2sr ;   ind(ividualistic) ex m2
            raise ValueError("Algo inconnu :",algo)
        # diagonale à zero sur la matrice
        if (self.algo=='com'):
            np.fill_diagonal(self.scores,0)    # A MULTIPLIER PAR LONGUEUR
        self.historic=[self.population.copy()]     # un copy sinon on pointe toujours sur la même population !!
    def run(self) :
        #  abstraction du n-1 quand un individu joue contre sa propre famille    
        for iter in range(self.itermax) :
            #fitnessInd = np.dot(self.scores,self.population)
            #fitnessStr = self.population * fitnessInd
            # print(self.population)  # TRACE A SUPPRIMER
            # -------- La seule ligne differente entre m3 et m5 ------------
            if (self.algo=='com') :
                # 2 choisis fonction de la population par une roue de la fortune, sans remise
                players = np.random.choice(self.idxstrats, size=2, replace=False, p=self.population/self.base)
                #print(str(players[0])+" "+str(players[1]))
            else :
                # 2 choisis fonction de la population par une roue de la fortune, avec remise
                players = np.random.choice(self.idxstrats, size=2, replace=True, p=self.population/self.base)
                #print(str(players[0])+" "+str(players[1]))
            #---------------------------------------------------------------
            # chaque population est incrémentee de son score respectif
            self.population[players[0]]+=self.scores[players[0],players[1]]
            self.population[players[1]]+=self.scores[players[1],players[0]]
            # destruction des individus selon la population : il faut verifier qu'on ne tombe pas dans le negatif
            # on les enleve d'1 coup au lieu de les enlever 1 par 1 : approximation de la methode pure
            toDestroy = self.scores[players[0],players[1]] + self.scores[players[1],players[0]]
            coeff = toDestroy / (sum(self.population) + toDestroy)
            # pop2 = [x-int(x/coeff) for x in population]
            self.population = np.array(list(map(lambda x : int(x*(1-coeff)) , self.population)))
            # il se peut que sum(population) < total .... dans CompetEcolo on s'en fout
            ecart = self.base - sum(self.population)
            # population[np.random.randint(self.nbstrats)] += ecart
            n = np.random.randint(self.nbstrats)
            while (self.population[n]==0):
                n = np.random.randint(self.nbstrats)
            self.population[n] +=ecart
            assert(self.base == sum(self.population))
            # On ne garde que 1000 points sur la totalité des itérations
            # if (self.itermax>2000 and iter%(self.itermax//1000)==1):
            self.historic.append(self.population.copy())
        # j'affiche juste le resultat final
        print(self.population ,'\t-> ', self.nomstrats[np.array(self.population).argmax()] )
    def drawPlot(self):
        df=pd.DataFrame(self.historic, columns=self.nomstrats)
        tmillis = int(round(time.time() * 1000))
        filename=self.prefix+'_EvolEncounter_'+self.algo+'_'+str(tmillis)+'.png'
        # print(df)
        # df.to_csv(filename+".csv", index=None, sep=';')
        # chgt des noms de colonnes pour ajouter les moyennes
        nblig=int(df.shape[0]/2)
        moy=df[nblig:].sum()//nblig
        moy=list(map (lambda x : int(x/self.base*1000) , moy))
        names=list(map (lambda x : x[0]+' '+str(x[1]) , list(zip(df.columns,moy))))
        df.columns=names
        df.plot(markevery=self.itermax//1000) # x=list(range(1,itermax,(itermax//1000)))
        plt.savefig(filename+"_legende.png", dpi=500)
        #df.plot(legend=False)
        #plt.savefig(filename+".png", dpi=1000)
        plt.show()
        plt.close()



# Ex M5
# EvolEncounter et EvolFermi sont batis exactement sur le même code
class EvolFermi:
    def __init__(self, algo, tournament, population, itermax, prefix=''):
        self.nbstrats = len(tournament.strategies)
        self.idxstrats=list(range(self.nbstrats))
        self.nomstrats=tournament.matrix.index.values
        self.scores = tournament.matrix.iloc[0:self.nbstrats,0:self.nbstrats].values.copy()
        # soit c'est la meme pop pour tout le monde, soit c'est un tableau
        if (type(population)==int) :
            self.population=np.array([population]*self.nbstrats  , dtype=int)
        else :
            self.population=np.array(population, dtype=int)    # un npArray pour pouvoir faire /total
        self.base=sum(self.population)
        self.itermax=itermax
        self.algo=algo
        self.prefix=prefix
        if (self.algo != 'com' and self.algo != 'ind' and self.algo != 'pls'):   # com(munautary) ex m2sr ;   ind(ividualistic) ex m2
            raise ValueError("Algo inconnu :",algo)
        # diagonale à zero sur la matrice
        if (self.algo=='com' or self.algo=='pls'):
            np.fill_diagonal(self.scores,0)    # A MULTIPLIER PAR LONGUEUR
        self.historic=[self.population.copy()]     # un copy sinon on pointe toujours sur la même population !!
    def run(self) :
        #  abstraction du n-1 quand un individu joue contre sa propre famille    
        for iter in range(self.itermax) :
            fitnessStr = self.population * np.dot(self.scores,self.population)
            # print(self.population)  # TRACE A SUPPRIMER
            # ----- La seule ligne differente entre m3 et m5 -------
            if (self.algo=='com') :
                # 2 choisis fonction de la population par une roue de la fortune, sans remise
                players = np.random.choice(self.idxstrats, size=2, replace=False, p=fitnessStr/sum(fitnessStr))
                #print(str(players[0])+" "+str(players[1]))
            else :
                                           # 2 choisis fonction de la population par une roue de la fortune, avec remise
                players = np.random.choice(self.idxstrats, size=2, replace=True, p=fitnessStr/sum(fitnessStr))
                #print(str(players[0])+" "+str(players[1]))
            #-------------------------------------------------------    
            # chaque population est incrémentee de son score respectif
            self.population[players[0]]+=self.scores[players[0],players[1]]
            self.population[players[1]]+=self.scores[players[1],players[0]]
            # destruction des individus selon la population : il faut verifier qu'on ne tombe pas dans le negatif
            # on les enleve d'1 coup au lieu de les enlever 1 par 1 : approximation de la methode pure
            toDestroy = self.scores[players[0],players[1]] + self.scores[players[1],players[0]]
            coeff = toDestroy / (sum(self.population) + toDestroy)
            # pop2 = [x-int(x/coeff) for x in population]
            self.population = np.array(list(map(lambda x : int(x*(1-coeff)) , self.population)))
            # il se peut que sum(population) < total .... dans CompetEcolo on s'en fout
            ecart = self.base - sum(self.population)
            # population[np.random.randint(self.nbstrats)] += ecart
            n = np.random.randint(self.nbstrats)
            while (self.population[n]==0):
                n = np.random.randint(self.nbstrats)
            self.population[n] +=ecart
            assert(self.base == sum(self.population))
            # On ne garde que 1000 points sur la totalité des itérations
            # if (self.itermax>2000 and iter%(self.itermax//1000)==1):
            self.historic.append(self.population.copy())
        # j'affiche juste le resultat final
        print(self.population ,'\t-> ', self.nomstrats[np.array(self.population).argmax()] )
    def drawPlot(self):
        df=pd.DataFrame(self.historic, columns=self.nomstrats)
        tmillis = int(round(time.time() * 1000))
        filename=self.prefix+'_EvolEncounter_'+self.algo+'_'+str(tmillis)+'.png'
        # print(df)
        # df.to_csv(filename+".csv", index=None, sep=';')
        # chgt des noms de colonnes pour ajouter les moyennes
        nblig=int(df.shape[0]/2)
        moy=df[nblig:].sum()//nblig
        moy=list(map (lambda x : int(x/self.base*1000) , moy))
        names=list(map (lambda x : x[0]+' '+str(x[1]) , list(zip(df.columns,moy))))
        df.columns=names
        df.plot(markevery=self.itermax//1000) # x=list(range(1,itermax,(itermax//1000)))
        plt.savefig(filename+"_legende.png", dpi=500)
        #df.plot(legend=False)
        #plt.savefig(filename+".png", dpi=1000)
        plt.show()
        plt.close()
