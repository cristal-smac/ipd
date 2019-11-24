import pandas as pd
import numpy as np
import pandas
import copy
import math
import matplotlib.pyplot as plt
import statistics
import random
import itertools
import Game



class Meeting :      
    def __init__(self,game,s1,s2,length=1000):
        self.game = game
        self.s1=s1.clone()
        self.s2=s2.clone()
        self.length=length
        self.nb_cooperation_s1 = 0
        self.nb_cooperation_s2 = 0
        
    def reinit(self):
        self.s1_score=0
        self.s2_score=0
        self.s1_rounds=""
        self.s2_rounds=""
    
    def run(self):
        self.reinit()
        for tick in range(0,self.length):
            c1=self.s1.getAction(tick).upper()
            c2=self.s2.getAction(tick).upper()
            if (c1 == "C"):
                self.nb_cooperation_s1 +=1
            if (c2 == "C"):
                self.nb_cooperation_s2 +=1
            self.s1_rounds+=c1
            self.s2_rounds+=c2
            self.s1.update(c1,c2)
            self.s2.update(c2,c1)
            act=self.game.actions
            self.s1_score+=self.game.scores['x'][act.index(c1),act.index(c2)]
            self.s2_score+=self.game.scores['y'][act.index(c1),act.index(c2)]

class Tournament:
    def __init__(self, game, strategies, length=1000, repeat=1):
        self.strategies = strategies
        self.game = game
        self.length=length
        self.repeat=repeat
        size=len(strategies);
        df = pd.DataFrame(np.zeros((size,size+1),dtype=np.int32))
        df.columns, df.index = [s.name for s in self.strategies]+["Total"], [s.name for s in self.strategies]
        self.matrix = df
        df2 = pd.DataFrame(np.zeros((size,size+1),dtype=np.int32))
        df2.columns, df2.index = [s.name for s in self.strategies]+["Total"], [s.name for s in self.strategies]
        self.cooperations = df2

    def run(self):
        for k in range(self.repeat):
            for i in range (0,len(self.strategies)):
                for j in range (i,len(self.strategies)):
                    meet = Meeting(self.game, self.strategies[i], self.strategies[j], self.length)
                    meet.run()
                    self.matrix.at[self.strategies[i].name, self.strategies[j].name] = meet.s1_score
                    self.matrix.at[self.strategies[j].name, self.strategies[i].name] = meet.s2_score
                    self.cooperations.at[self.strategies[i].name, self.strategies[j].name] = meet.nb_cooperation_s1
                    self.cooperations.at[self.strategies[j].name, self.strategies[i].name] = meet.nb_cooperation_s2
        self.matrix["Total"] = self.matrix.sum(axis=1)
        self.matrix.sort_values(by='Total', ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)
        self.cooperations["Total"] = self.cooperations.sum(axis=1)
        self.cooperations.sort_values(by='Total', ascending=False, inplace=True)
        rows = list(self.cooperations.index) + ["Total"]
        self.cooperations = self.cooperations.reindex(columns=rows)

class Ecological:
    def __init__(self, game, strategies, length=1000, repeat=1, pop=100):
        self.strategies = strategies
        self.pop = pop
        self.game = game
        self.length = length
        self.generation = 0 #Numéro de la génération actuelle
        self.base = pop*len(strategies)
        self.historic = pandas.DataFrame(columns = [strat.name for strat in strategies])
        self.historic.loc[0] = [pop for x in range (len(strategies))]
        self.extinctions = dict((s.name,math.inf) for s in strategies)
        self.cooperations =  dict((s.name,0) for s in strategies)
        self.listeCooperations = list()
        self.scores = dict((s.name,0) for s in strategies)
        self.tournament = Tournament(self.game, self.strategies,length,repeat)
        self.tournament.run()
        
    def run(self):
        dead = 0
        stab = False
        while ((self.generation < 1000) and (stab==False)):
            parents = list(copy.copy(self.historic.loc[self.generation]))
            for i in range (len(self.strategies)):
                strat=self.strategies[i].name
                if (self.historic.at[self.generation, strat] != 0):
                    score = 0
                    cooperations = 0
                    for j in range(len(self.strategies)): 
                        strat2 = self.strategies[j].name
                        if (self.historic.at[self.generation, strat2] != 0):
                            if i==j:
                                score+=(self.historic.at[self.generation, strat]-1)*self.tournament.matrix.at[strat,strat2]
                                cooperations+=(self.historic.at[self.generation, strat]-1)*self.tournament.cooperations.at[strat,strat2]
                            else:
                                score+=self.historic.at[self.generation, strat2]*self.tournament.matrix.at[strat,strat2]
                                cooperations+=self.historic.at[self.generation, strat2]*self.tournament.cooperations.at[strat,strat2]
                        self.scores[strat] = score
                        self.cooperations[strat] = cooperations
                    
            total = 0
            totalCooperations = 0
            for strat in self.strategies:
                total+=self.scores[strat.name]*self.historic.at[self.generation, strat.name]
                totalCooperations += self.cooperations[strat.name]*self.historic.at[self.generation, strat.name]
            for strat in self.strategies:        
                parent = self.historic.at[self.generation, strat.name]
                if (self.scores[strat.name] != 0):
                    self.historic.at[self.generation+1, strat.name] = math.floor(self.base*parent*self.scores[strat.name]/total)
                elif (self.scores[strat.name] == 0):
                    self.historic.at[self.generation+1, strat.name] = 0
                    dead += 1
                if ((parent!=0) and (self.historic.at[self.generation+1, strat.name] == 0)):
                    self.extinctions[strat.name] = self.generation+1
                elif (self.historic.at[self.generation+1, strat.name] != 0):
                    self.extinctions[strat.name] = self.historic.at[self.generation+1, strat.name]*1000
                if (dead == len(self.strategies) - 1):
                    stab = True
            self.listeCooperations.append(totalCooperations/(self.base*self.length*len(self.strategies)))
            self.generation+=1
            if (parents == list(self.historic.loc[self.generation])):stab = True
        trie = sorted(self.extinctions.items(), key=lambda t:t[1], reverse=True)
        df_trie = pandas.DataFrame()
        for t in trie :
            df_trie[t[0]]=self.historic[t[0]]
        self.historic = df_trie
        return self.historic

    def saveData(self):
        date = datetime.datetime.now()
        self.historic.to_csv(str(date)+'.csv', sep=';', encoding='utf-8')

    def drawPlot(self,nbCourbes=None,nbLegends=None):
        nbCourbes = len(self.strategies) if (nbCourbes==None) else nbCourbes
        nbLegends = len(self.strategies) if (nbLegends==None) else nbLegends
        strat = self.historic.columns.tolist()
        for i in range(nbCourbes):
            plt.plot(self.historic[strat[i]], label=strat[i] if (i<nbLegends) else '_nolegend_')
        plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)
        plt.ylabel('Population')
        plt.xlabel('Generation')
        plt.show()
        #date = datetime.datetime.now()
        #plt.savefig(str(date)+'.png', dpi=1000)
    
    def drawCooperations(self):
        plt.plot(self.listeCooperations)
        plt.ylabel('Pourcentage de cooperations')
        plt.xlabel('Generation')
        plt.ylim(0, 101)
        plt.show()


def subClasses(soupe, n):
    dip =[(3,3),(0,5),(5,0),(1,1)]   # Dilemme du prisonnier
    g = Game.Game(dip,['C','D'])

    if (n > len(soupe)):
        print ("the soup size must be smaller than n")
        return   
    res = pd.DataFrame(np.nan,[s.name for s in soupe], ["BestRank","WorstRank", "RankAvg", "RankStd"])   
    for s in soupe:
        res.at[s.name, "BestRank"] = len(soupe)
    ranks = dict()
    sousEnsembles = list(itertools.combinations(soupe, n))
    for s in sousEnsembles:
        e = Ecological(g, s)
        e.run()
        classements = e.historic.iloc[e.generation].rank(0, method="min", ascending=False)
        for strat in s : 
            classement = classements[strat.name]
            if (math.isnan(res.at[strat.name, "BestRank"]) or classement < res.at[strat.name, "BestRank"]):
                res.at[strat.name, "BestRank"] = classement
            if (math.isnan(res.at[strat.name, "WorstRank"]) or classement > res.at[strat.name, "WorstRank"]):
                res.at[strat.name, "WorstRank"] = classement  
            if (strat.name in ranks.keys()):
                ranks[strat.name].append(classement)
            if (strat.name not in ranks.keys()):
                ranks[strat.name] = [classement]      
    for strat in soupe : 
        res.at[strat.name, "RankAvg"] = statistics.mean(ranks[strat.name])
        res.at[strat.name, "RankStd"] = statistics.stdev(ranks[strat.name])
    print(res.sort_values(by = ['RankAvg', 'BestRank', 'RankStd', 'WorstRank'],  ascending = [True, True, True, True ]))
          

def subClassesWithOneStrat(soupe, n, strategy, printAll = False):

    dip =[(3,3),(0,5),(5,0),(1,1)]   # Dilemme du prisonnier
    g = Game.Game(dip,['C','D'])

    if (n > len(soupe)):
        print ("the soup size must be smaller than n")
        return     
    res = pd.DataFrame(np.nan,[s.name for s in soupe+[strategy]], ["BestRank", "WorstRank", "RankAvg", "RankStd"])   
    sousEnsembles = list(itertools.combinations(soupe, n))
    ranks = dict()
    bestComp = []
    worstComp = []
    for s in sousEnsembles:  
        e = Ecological(g, list(s) + [strategy])
        e.run()
        classements = e.historic.iloc[e.generation].rank(0, method="min", ascending=False)
        for strat in  list(s) + [strategy] : 
            classement = classements[strat.name]
            if (math.isnan(res.at[strat.name, "BestRank"]) or classement < res.at[strat.name, "BestRank"]):
                res.at[strat.name, "BestRank"] = classement
                if (strat == strategy):
                    bestComp = list(s) + [strategy]
            if (math.isnan(res.at[strat.name, "WorstRank"]) or classement > res.at[strat.name, "WorstRank"]):
                res.at[strat.name, "WorstRank"] = classement 
                if (strat == strategy):
                    worstComp = list(s) + [strategy]
            if (strat.name in ranks.keys()):
                ranks[strat.name].append(classement)
            if (strat.name not in ranks.keys()):
                ranks[strat.name] = [classement]
    for s in soupe+[strategy]:
        if (s.name in ranks.keys()):
            res.at[s.name, "RankAvg"] = statistics.mean(ranks[s.name])
            if (len(ranks[s.name]) > 1):
                res.at[s.name, "RankStd"] = statistics.stdev(ranks[s.name])
    if (printAll) :       
        print(res.sort_values(by = ['RankAvg', 'BestRank', 'RankStd', 'WorstRank'],  ascending = [True, True, True, True ]))
    else : 
        print("Strategy ranking : "+strategy.name)
        print(res.loc[strategy.name,:])
    return bestComp, worstComp, strategy



def subClassesRandomWithOneStrat(p, soupe, n, strategy, printAll = False ):
    dip =[(3,3),(0,5),(5,0),(1,1)]   # Dilemme du prisonnier
    g = Game.Game(dip,['C','D'])
    if (n > len(soupe)):
        "the soup size must be smaller than n"
        return  
    res = pd.DataFrame(np.nan,[s.name for s in soupe+[strategy]], ["BestRank","WorstRank", "RankAvg", "RankStd"])
    ranks = dict()
    bestComp = []
    worstComp = []
    for i in range (0, p) : 
        #print("Competition "+str(i+1)+ "/"+str(p))
        strategies = []
        strategies.append(strategy)
        indice = [i for i in range (0, len(soupe))]
        for i in range (0, n):
            indiceStrat = random.choice(indice)
            indice.remove(indiceStrat)
            strategies.append(soupe[indiceStrat])
        #print("Les stratégies qui jouent sont : ")
        #for s in strategies :
            #print(s.name)
        e = Ecological(g, strategies)
        e.run()
        classements = e.historic.iloc[e.generation].rank(0, method="min", ascending=False)
        for strat in strategies : 
            classement = classements[strat.name]
            if (math.isnan(res.at[strat.name, "BestRank"]) or classement < res.at[strat.name, "BestRank"]):
                res.at[strat.name, "BestRank"] = classement
                if (strat == strategy):
                    bestComp = strategies
            if (math.isnan(res.at[strat.name, "WorstRank"]) or classement > res.at[strat.name, "WorstRank"]):
                res.at[strat.name, "WorstRank"] = classement  
                if (strat == strategy):
                    worstComp = strategies
            if (strat.name in ranks.keys()):
                ranks[strat.name].append(classement)
            if (strat.name not in ranks.keys()):
                ranks[strat.name] = [classement]
    for s in soupe+[strategy]:
        if (s.name in ranks.keys()):
            res.at[s.name, "RankAvg"] = statistics.mean(ranks[s.name])
            if (len(ranks[s.name]) > 1):
                res.at[s.name, "RankStd"] = statistics.stdev(ranks[s.name])
    if (printAll) :   
        print(res.sort_values(by = ['RankAvg', 'BestRank', 'RankStd', 'WorstRank'],  ascending = [True, True, True, True ]))
    else : 
        print("Strategy ranking : "+strategy.name)
        print(res.loc[strategy.name,:])
    return bestComp, worstComp, strategy

def getAllMemory(x,y):
    if (x+y > 4):
        return "Pas calculable"
    len_genome = max(x,y)+2**(x+y)
    permut = [p for p in itertools.product(['C','D'], repeat=len_genome)]
    genomes = [''.join(p) for p in permut]
    return [Mem(x,y,gen) for gen in genomes]





        
