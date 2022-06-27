#This file contains 8 evolutionary algorithms

# 2 deterministic models (individualistic ou communautary) both with Integer and with percentages :  EvolDeterInt EvolDeterReal (ex M2)
 
# 2 stochastic models where we choose 2 individuals randomly and where we increase population depending of the scores : EvolEncounter (ex M3)

# 2 stochastic models based on Moran process, where a fitness is computed for each individual, and where we use this fitness as a probability to choose one individual on which we increase or decrease by 1, the population: EvolMoran (ex M4)

# 2 stochastic models based on Fermi process where we choose 2 individuals based on their fitness, then they make a meeting, and we increase or decrease the population depending of the meeting result:"EvolFermi" (M5)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ipd import *




# Wrapper sur Tournament pour faire ind et com
class Classic:
    def __init__(self, algo, tournament, diag=0) :
        self.tournament = tournament
        self.matrix=tournament.matrix.iloc[:,:-1].copy()
        self.algo=algo
        if (algo != 'com' and algo != 'ind'):   # m2sr=com(munautary)  m2=ind(ividualistic)
            raise ValueError("Algo inconnu :",algo)
        if (algo=='com'):
            np.fill_diagonal(self.matrix.values,diag)  # A MULTIPLIER PAR LONGUEUR
    def run(self) :
        self.matrix["Total"] = self.matrix.sum(axis=1)
    def getRanking(self) :
        name='ClaT_'+self.algo
        df=pd.DataFrame(self.matrix['Total']) #, index=self.nomstrats,columns=[name]) # .sort_values(by='m2',ascending=False)
        df.columns=[name]
        df['rank_'+name] = df[name].rank(axis=0, ascending=False, method='average')  # min,max,first,dense
        return df


        

#class M2Reels :
class EvolDeterReal:
    def __init__(self, algo, tournament, population, itermax, resilience=0, diag=0) :
        nbstrats = len(tournament.strategies)
        self.nomstrats=tournament.matrix.index.values
        self.scores = tournament.matrix.iloc[0:nbstrats,0:nbstrats].values.copy()
        self.population=np.array(population)
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
    def drawPlot(self, filename=None) :
        df = pd.DataFrame(self.historic, columns=self.nomstrats)
        df.plot(grid=True) # , title="DetReels "+self.algo , figsize=(10,6))  # Defaut figsize=(6,4)
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')
        if filename==None:
            tmillis = int(round(time.time() * 1000))    
            filename=str(len(self.nomstrats))+'str_DetReal_'+self.algo+'_res'+str(self.resilience)+'_'+str(tmillis)+'.png'
        plt.savefig(filename, dpi=500)
        print('figure sauvée : ' +filename)
        plt.show()
        plt.close()

        
#class M2Entiers :
class EvolDeterInt:
    def __init__(self, algo, tournament, population, itermax, strict=True , resilience=0 , diag=0) :
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
            identiques = np.all(self.historic[-1] == self.historic[-2])  # or sum(self.historic[-1] - self.historic[-2]) == 0
            identiques = identiques or (nbSurvivants==1) # 1 seul non nul (pb avec com (m2sr))
            i=i+1
    def drawPlot(self,filename=None) :
        df = pd.DataFrame(self.historic, columns=self.nomstrats)
        df.plot(grid=True) # , title="M2Entier "+self.algo , figsize=(10,6))  # Defaut figsize=(6,4)
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')
        if filename==None:
            tmillis = int(round(time.time() * 1000))    
            filename=str(len(self.nomstrats))+'str_DetInt_'+self.algo+'_res'+str(self.resilience)+'_'+str(tmillis)+'.png'
        plt.savefig(filename, dpi=500)
        print('figure sauvée : ' +filename)
        plt.show()
        plt.close()
    def getRanking(self) :
        name='Det_'+self.algo
        df=pd.DataFrame(self.historic[-1], index=self.nomstrats,columns=[name]) # .sort_values(by='m2',ascending=False)
        df['rank_'+name] = df[name].rank(axis=0, ascending=False, method='average')  # min,max,first,dense
        return df
        

        

#Ex M3
class EvolEncounter:
    def __init__(self, algo, tournament, population, itermax):
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
                     # toDestroy = self.scores[players[0],players[1]] + self.scores[players[1],players[0]]
                     # coeff = 1 - (toDestroy / (sum(self.population) + toDestroy))
            coeff = self.base / sum(self.population)
            # pop2 = [x-int(x/coeff) for x in population]
            self.population = np.array(list(map(lambda x : int(x*coeff) , self.population)))
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
    def drawPlot(self, filename=None):
        df=pd.DataFrame(self.historic, columns=self.nomstrats)
        # print(df)
        # df.to_csv(filename+".csv", index=None, sep=';')
        # chgt des noms de colonnes pour ajouter les moyennes
        moy=list(map (lambda x : int(x/self.base*1000) , self.getFinalPop()))
        names=list(map (lambda x : x[0]+' '+str(x[1]) , list(zip(df.columns,moy))))
        df.columns=names
        df.plot(markevery=self.itermax//1000 , grid=True) # x=list(range(1,itermax,(itermax//1000)))  # We take 1000 points, not more
        #df.plot(legend=False)   # Pour PLS
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')
        if filename==None:
            tmillis = int(round(time.time() * 1000))    
            filename=str(len(self.nomstrats))+'str_EvolEncounter_'+self.algo+'_'+str(tmillis)+'.png'
        plt.savefig(filename, dpi=500)
        print('figure sauvée : ' +filename)
        plt.show()
        plt.close()
    def getFinalPop(self):
        # classically, final pop is
        # self.historic[-1] ou pd.DataFrame(m2.historic[-1:], columns=m2.nomstrats)
        # but here we average the lasts
        nLast=int(len(self.historic)/4)       # start at 3/4 to compute avg
        indLast=len(self.historic)-nLast # index of last
        moy=pd.DataFrame(self.historic[indLast:],columns=self.nomstrats).sum()//nLast
        return moy
    def getRanking(self) :               #for stochastic methods, tanking is based on the avg of the 3/4 last pop
        name='Enc_'+self.algo
        # it's a Serie, not a df
        # rank computed on getFinalPop, not historic[-1] to be more robust
        df = pd.DataFrame(self.getFinalPop().transpose(), columns=[name])
        df['rank_'+name] = df[name].rank(axis=0, ascending=False, method='average')  # min,max,first,dense
        return df
        


# Ex M5
# EvolEncounter et EvolFermi sont batis exactement sur le même code
class EvolFermi:
    def __init__(self, algo, tournament, population, itermax):
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
                    # toDestroy = self.scores[players[0],players[1]] + self.scores[players[1],players[0]]
                    #coeff = toDestroy / (sum(self.population)) # + toDestroy)
            coeff = self.base / sum(self.population)
            # pop2 = [x-int(x/coeff) for x in population]
            self.population = np.array(list(map(lambda x : int(x*coeff) , self.population)))
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
    def drawPlot(self,filename=None):
        df=pd.DataFrame(self.historic, columns=self.nomstrats)
        # print(df)
        # df.to_csv(filename+".csv", index=None, sep=';')
        # chgt des noms de colonnes pour ajouter les moyennes
        moy=list(map (lambda x : int(x/self.base*1000) , self.getFinalPop()))
        names=list(map (lambda x : x[0]+' '+str(x[1]) , list(zip(df.columns,moy))))
        df.columns=names
        df.plot(markevery=self.itermax//1000 , grid=True) # x=list(range(1,itermax,(itermax//1000)))  # We take 1000 points, not more
        #df.plot(legend=False)  # pour PLS
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')
        if filename==None:
            tmillis = int(round(time.time() * 1000))    
            filename=str(len(self.nomstrats))+'str_EvolFermi_'+self.algo+'_'+str(tmillis)+'.png'
        plt.savefig(filename, dpi=500)
        print('figure sauvée : ' +filename)
        plt.show()
        plt.close()
    def getFinalPop(self):
        # classically, final pop is
        # self.historic[-1] ou pd.DataFrame(m2.historic[-1:], columns=m2.nomstrats)
        # but here we average the lasts
        nLast=int(len(self.historic)/4)       # start at 3/4 to compute avg
        indLast=len(self.historic)-nLast # index of last
        moy=pd.DataFrame(self.historic[indLast:],columns=self.nomstrats).sum()//nLast
        return moy
    def getRanking(self) :               #for stochastic methods, tanking is based on the avg of the 3/4 last pop
        name='Fer_'+self.algo
        # it's a Serie, not a df
        # rank computed on getFinalPop, not historic[-1] to be more robust
        df = pd.DataFrame(self.getFinalPop().transpose(), columns=[name])
        df['rank_'+name] = df[name].rank(axis=0, ascending=False, method='average')  # min,max,first,dense
        return df




# Ex M4
# EvolMoran
class EvolMoran:
    def __init__(self, algo, tournament, population, itermax):
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
        if (self.algo != 'com' and self.algo != 'ind'):   # com(munautary) ex m2sr ;   ind(ividualistic) ex m2
            raise ValueError("Algo inconnu :",algo)
        # diagonale à zero sur la matrice
        if (self.algo=='com'):
            np.fill_diagonal(self.scores,0)    # A MULTIPLIER PAR LONGUEUR
        self.historic=[self.population.copy()]     # un copy sinon on pointe toujours sur la même population !!
    def run(self) :
        #  abstraction du n-1 quand un individu joue contre sa propre famille    
        for iter in range(self.itermax) :
            fitnessStr = self.population * np.dot(self.scores,self.population)
            # print(self.population)  # TRACE A SUPPRIMER
            # ----- La seule ligne differente entre m3 et m4 -------
            winner = np.random.choice(self.idxstrats, size=1, replace=False, p=fitnessStr/sum(fitnessStr))
            looser = np.random.choice(self.idxstrats, size=1, replace=False, p=self.population/self.base)          
            # chaque population est incrémentee de +1/-1
            self.population[winner]+=1
            self.population[looser]-=1
            #-------------------------------------------------------
            assert(self.base == sum(self.population))
            # On ne garde que 1000 points sur la totalité des itérations
            # if (self.itermax>2000 and iter%(self.itermax//1000)==1):
            self.historic.append(self.population.copy())
        # j'affiche juste le resultat final
        print(self.population ,'\t-> ', self.nomstrats[np.array(self.population).argmax()] )
    def drawPlot(self, filename=None):
        df=pd.DataFrame(self.historic, columns=self.nomstrats)
        # print(df)
        # df.to_csv(filename+".csv", index=None, sep=';')
        # chgt des noms de colonnes pour ajouter les moyennes
        moy=list(map (lambda x : int(x/self.base*1000) , self.getFinalPop()))
        names=list(map (lambda x : x[0]+' '+str(x[1]) , list(zip(df.columns,moy))))
        df.columns=names
        df.plot(markevery=self.itermax//1000 , grid=True) # x=list(range(1,itermax,(itermax//1000)))  # We take 1000 points, not more
        #df.plot(legend=False)  # pour PLS
        ax=plt.gca()  # gca : get current axes
        ax.set_facecolor('#F0F0F0')
        if filename==None:
            tmillis = int(round(time.time() * 1000))    
            filename=str(len(self.nomstrats))+'str_EvolMoran_'+self.algo+'_'+str(tmillis)+'.png'
        plt.savefig(filename, dpi=500)
        print('figure sauvée : ' +filename)     
        plt.show()
        plt.close()
    def getFinalPop(self):
        # classically, final pop is
        # self.historic[-1] ou pd.DataFrame(m2.historic[-1:], columns=m2.nomstrats)
        # but here we average the lasts
        nLast=int(len(self.historic)/4)       # start at 3/4 to compute avg
        indLast=len(self.historic)-nLast # index of last
        moy=pd.DataFrame(self.historic[indLast:],columns=self.nomstrats).sum()//nLast
        return moy
    def getRanking(self) :               #for stochastic methods, tanking is based on the avg of the 3/4 last pop
        name='Mor_'+self.algo
        # it's a Serie, not a df
        # rank computed on getFinalPop, not historic[-1] to be more robust
        df = pd.DataFrame(self.getFinalPop().transpose(), columns=[name])
        df['rank_'+name] = df[name].rank(axis=0, ascending=False, method='average')  # min,max,first,dense
        return df


# Calcule et cumule les distances entre soit des algos déterministes (5), soit tout (11)
# afin de permettre in fine d'afficher un dendogramme de comparaison de ces méthodes
# Attention : si methods='all' ça lance toutes les méthodes stochastiques autant de fois qu'il y a de bags !
# Près de 2h30 de calcul !
def cumulativeDistanceMatrix(bags, repeat=5, methods='Deterministic'):  # all  or deterministic)
    print(methods)
    cumulresult=pd.DataFrame()
    cumulrank=pd.DataFrame()
    for i in range(len(bags)) :
        print("------------- ",i," -----------------")
        res,rank = resultMatrix(bags[i],repeat=repeat,methods=methods)
        # dist sur les resultats
        dres = dist(res)
        print('-- result normé : --\n',res)
        print('-- dist result :--\n',dres)
        # dist sur les rangs
        drank = rank.corr(method='spearman')
        drank=1-drank
        print('-- rank : --\n',rank)
        print('-- dist rank :--\n',drank)
        # cumul
        if cumulresult.empty:
            cumulresult=dres
            cumulrank=drank
        else:
            cumulresult += dres
            cumulrank += drank
    return (cumulresult, cumulrank)


from math import sqrt

def dist(df):
    x = pd.DataFrame(columns=df.columns, index=df.columns)
    for i in (range(df.shape[1])):
        for j in (range(df.shape[1])):
            x.iloc[i,j]= sqrt(sum((df.iloc[:,i]-df.iloc[:,j])**2))
    return x


# renvoie une matrice de rangs normalisée Les rangs sont calculés
# selon la méthode des moyennes en cas d'aexeco
# Les méthodes stochastiques sont répétées Repeat fois leurs classements cumulés
def resultMatrix(bag,repeat=5,methods='Deterministic',normalized=True):  # ADSTECI
    methods=methods[0].capitalize()
    print(methods)
    #(A)ll, (D)eterministic, (S)tochastics, (T)tournaments, (E)volutionaries, (C)ommunautary evol,  (I)ndividualistic evol
    
    t=Tournament(g,bag,100)
#    g = game.Game([(2, 2), (0, 3), (3, 0), (1, 1)], ["C", "D"])
#    t=Tournament(g,bag,100)
    t.run()

    results=pd.DataFrame(index=[s.name for s in bag])
    ranks=pd.DataFrame(index=[s.name for s in bag])
        
    if methods in 'ADT' :
        tb = TournamentVictory(t)
        tb.run()
        x= pd.DataFrame(tb.matrix.loc[:,['Total']]+len(bag) )      # pour score
        x.columns=['VicT']
        if normalized:
            x.iloc[:,0] = (x.iloc[:,0]*1)/sum(x.iloc[:,0])
        results = results.join(x, how='inner')
        y= pd.DataFrame(tb.matrix.loc[:,['Total']])                 # pour rank
        y['rank_VicT'] = y['Total'].rank(axis=0, ascending=False, method='average')
        ranks = ranks.join(y.iloc[:,1], how='inner')  # keep only rang
        
    if methods in 'ADT' :
        c=Classic('ind',t)
        c.run()
        x = c.getRanking() # first col=score, second=rank
        if normalized:
            x.iloc[:,0] = (x.iloc[:,0]*1)/sum(x.iloc[:,0])
        results=results.join(x.iloc[:,0].to_frame() , how='inner')
        ranks=ranks.join(x.iloc[:,1] , how='inner')      
        
    if methods in 'ADT' :    
        c=Classic('com',t)
        c.run()
        x = c.getRanking() # first col=score, second=rank
        if normalized:
            x.iloc[:,0] = (x.iloc[:,0]*1)/sum(x.iloc[:,0])
        results=results.join(x.iloc[:,0].to_frame() , how='inner')
        ranks=ranks.join(x.iloc[:,1] , how='inner')      

    if methods in 'ADEI' :    
        m2=EvolDeterInt('ind', t , 100 , 1000, '')
        m2.run()
        x = m2.getRanking() # first col=score, second=rank
        if normalized:
            x.iloc[:,0] = (x.iloc[:,0]*1)/sum(x.iloc[:,0])
        results=results.join(x.iloc[:,0].to_frame() , how='inner')
        ranks=ranks.join(x.iloc[:,1] , how='inner')      

    if methods in 'ADEC' :        
        m2=EvolDeterInt('com', t , 100 , 1000, '')
        m2.run()
        x = m2.getRanking() # first col=score, second=rank
        if normalized:
            x.iloc[:,0] = (x.iloc[:,0]*1)/sum(x.iloc[:,0])
        results=results.join(x.iloc[:,0].to_frame() , how='inner')
        ranks=ranks.join(x.iloc[:,1] , how='inner')      

    # LIMIT OF DETERMINISTIC METHODS
        
    if methods in 'ASEI' :            
        m2=EvolEncounter('ind', t , 200000 , 200000, '')
        x=pd.Series(index=m2.nomstrats, dtype=np.int64)
        for i in range(repeat):
            m2.run()
            x = x+m2.getFinalPop()
        if normalized :
            x = (x*1)/sum(x)
        results = results.join(x.to_frame(name='Enc_ind'), how='inner')
        # then we compute the rank on the cumulation, and not the converse
        r=x.rank(axis=0,ascending=False, method='average')
        ranks = ranks.join(r.to_frame(name='Rank_Enc_Ind') , how='inner')
        print('evolEncounterInd done')

    if methods in 'ASEC' :
        m2=EvolEncounter('com', t , 200000 , 200000, '')
        x=pd.Series(index=m2.nomstrats, dtype=np.int64)
        for i in range(repeat):
            m2.run()
            x = x+m2.getFinalPop()
        if normalized :
            x = (x*1)/sum(x)
        results = results.join(x.to_frame(name='Enc_com'), how='inner')
        # then we compute the rank on the cumulation, and not the converse
        r=x.rank(axis=0,ascending=False, method='average')
        ranks = ranks.join(r.to_frame(name='Rank_Enc_com') , how='inner')
        print('evolEncounterCom done')

    if methods in 'ASEI' :                
        m2=EvolFermi('ind', t , 200000 , 200000, '')
        x=pd.Series(index=m2.nomstrats, dtype=np.int64)
        for i in range(repeat):
            m2.run()
            x = x+m2.getFinalPop()
        if normalized :
            x = (x*1)/sum(x)
        results = results.join(x.to_frame(name='Fer_ind'), how='inner')
        # then we compute the rank on the cumulation, and not the converse
        r=x.rank(axis=0,ascending=False, method='average')
        ranks = ranks.join(r.to_frame(name='Rank_Fer_ind') , how='inner')
        print('evolFermiInd done')

    if methods in 'ASEC' :                
        m2=EvolFermi('com', t , 200000 , 200000, '')
        x=pd.Series(index=m2.nomstrats, dtype=np.int64)
        for i in range(repeat):
            m2.run()
            x = x+m2.getFinalPop()
        if normalized :
            x = (x*1)/sum(x)
        results = results.join(x.to_frame(name='Fer_com'), how='inner')
        # then we compute the rank on the cumulation, and not the converse
        r=x.rank(axis=0,ascending=False, method='average')
        ranks = ranks.join(r.to_frame(name='Rank_Fer_com') , how='inner')
        print('evolFermiCom done')

    if methods in 'ASEI' :                
        m2=EvolMoran('ind', t , 1000 , 200000, '')
        x=pd.Series(index=m2.nomstrats, dtype=np.int64)
        for i in range(repeat):
            m2.run()
            x = x+m2.getFinalPop()
        if normalized :
            x = (x*1)/sum(x)
        results = results.join(x.to_frame(name='Mor_ind'), how='inner')
        # then we compute the rank on the cumulation, and not the converse
        r=x.rank(axis=0,ascending=False, method='average')
        ranks = ranks.join(r.to_frame(name='Rank_Mor_ind') , how='inner')
        print('evolMoranInt done')

    if methods in 'ASEC' :
        m2=EvolMoran('com', t , 1000 , 200000, '')
        x=pd.Series(index=m2.nomstrats, dtype=np.int64)
        for i in range(repeat):
            m2.run()
            x = x+m2.getFinalPop()
        if normalized :
            x = (x*1)/sum(x)
        results = results.join(x.to_frame(name='Mor_com'), how='inner')
        # then we compute the rank on the cumulation, and not the converse
        r=x.rank(axis=0,ascending=False, method='average')
        ranks = ranks.join(r.to_frame(name='Rank_Mor_com') , how='inner')
        print('evolMoranCom done')

    return (results,ranks)




# ex: pops=oneVsAll(len(bag),1000,[20,50,100])
def oneVsAll(size,oneAt,otherStarts) :
    pops=[]
    for n in otherStarts :
        for i in range(size) :
            p1=[n]*size
            p1[i]=oneAt
            pops.append(p1)
    return pops



# bag : the bag to use (list of strategies)
# meth : the method to use  (DeterInt, Encounter, Moran,Fermi)
# algo : ind (individualistic) or com (communautarian)
# pops : list of lists of pop starting points (of size=len(bag))
def draw_polygon(bag,meth,algo,pops,name=None) :
    m1=Tournament(g,bag,100)
    m1.run()
    
    # tracé du polygone
    fig = plt.figure(figsize=(10, 10))
    nbstrats = len(m1.strategies)
    xs=[]
    ys=[]
    for k in range(nbstrats) :
        xs.append(np.sin(k*2*np.pi/nbstrats))
        ys.append(np.cos(k*2*np.pi/nbstrats))
        plt.text(np.sin(k*2*np.pi/nbstrats) , np.cos(k*2*np.pi/nbstrats) ,m1.matrix.columns[k] , size='xx-large')
    plt.fill(xs+xs[:1], ys+ys[:1] , 'mistyrose')
    plt.plot(xs+xs[:1], ys+ys[:1], 'black')

    # Calcul des trajectoires    
    for pop in pops :
        if (meth=='DeterReal'):
            m2=EvolDeterReal(algo,m1, np.array(pop)/sum(pop), 10000, resilience=0)
        elif (meth=='DeterInt'):
            m2=EvolDeterInt(algo, m1,pop,1000)
        elif (meth=='Encounter'):
            m2=EvolEncounter(algo,m1,pop,100000)
        elif (meth=='Moran'):
            m2=EvolMoran(algo,m1,pop,100000)
        elif (meth=='Fermi'):
            m2=EvolFermi(algo,m1,pop,100000)
        else :
            plt.close()
            raise ValueError("Methode inconnue :",meth)
        m2.run()
        
        # pourcentages finaux : print(m2.historic[-1] *1.0 / sum(m2.historic[-1]))
        # print("duree d'evolution ",len(m2.historic))
        # Trace des convergences dans le cas Reel : print(m2.historic[-1])
        # print("taux coop final en entiers : ",m2.cooperationList[-1])
        
        # On trace au maximum 500 segments par trajectoire
        df = pd.DataFrame(m2.historic[::max(1,len(m2.historic)//500)], columns=m2.nomstrats)
        #df = pd.DataFrame(m2.historic[::], columns=m2.nomstrats)
        for i in range(df.shape[0]) :
            # df.loc[i,'abs']= sum(np.array(df.loc[i])[0:3] * [a,b,c])  / sum(df.loc[i][0:3])
            df.loc[i,'abs']= np.inner(np.array(df.loc[i])[0:nbstrats] , xs)  / sum(df.loc[i][0:nbstrats])
            df.loc[i,'ord']= sum(np.array(df.loc[i])[0:nbstrats] * ys) / sum(df.loc[i][0:nbstrats])
            plt.plot(df.iloc[-1,-2],df.iloc[-1,-1], 'bo')
        plt.plot(df['abs'], df['ord'])
    # eventuellement on retrace le dernier point s'il est effacé 
    print(df.iloc[-1,:])
    plt.plot(df.iloc[-1,-2],df.iloc[-1,-1], 'bo')

    plt.axis('off')
    if name==None:
        tmillis = int(round(time.time() * 1000))  # millisec 
        name=str(len(bag))+'str_polygon_'+meth+'_'+algo+'_'+str(tmillis)
    plt.savefig(name, dpi=500)
    print('figure sauvée : ' +name)
    plt.show()
    plt.close()

