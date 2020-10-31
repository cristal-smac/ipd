import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import statistics
import random
import itertools
#import datetime
import multiprocessing as multip


import game

scores = [(3, 3), (0, 5), (5, 0), (1, 1)]
g = game.Game(scores, ["C", "D"])

class Evaluator:
    def run(self):
        pass


class Meeting(Evaluator):
    def __init__(self, game, s1, s2, length=1000):
        self.game = game
        self.s1 = s1.clone()
        self.s2 = s2.clone()
        self.length = length
        self.nb_cooperation_s1 = 0
        self.nb_cooperation_s2 = 0

    def reinit(self):
        self.s1_score = 0
        self.s2_score = 0
        self.s1_rounds = []
        self.s2_rounds = []

    def run(self):
        self.reinit()
        for tick in range(0, self.length):
            c1 = self.s1.getAction(tick).upper()
            c2 = self.s2.getAction(tick).upper()
            if c1 == "C":
                self.nb_cooperation_s1 += 1
            if c2 == "C":
                self.nb_cooperation_s2 += 1
            self.s1_rounds.append(c1)
            self.s2_rounds.append(c2)
            self.s1.update(c1, c2)
            self.s2.update(c2, c1)
            act = self.game.actions
            self.s1_score += self.game.scores["x"][act.index(c1), act.index(c2)]
            self.s2_score += self.game.scores["y"][act.index(c1), act.index(c2)]

    def prettyPrint(self,max=20) :
        print("{:8}\t{} = {}".format(self.s1.name, ' '.join(map(str, self.s1_rounds)) , self.s1_score))
        print("{:8}\t{} = {}".format(self.s2.name, ' '.join(map(str, self.s2_rounds)) , self.s2_score))


#  For use only with deterministic strategy sets. Nothing is repeated
class Tournament(Evaluator):
    def __init__(self, game, strategies, length=1000):
        self.strategies = strategies
        self.game = game
        self.length = length
        size = len(strategies)
        df = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int64))
        df.columns, df.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.matrix = df
        df2 = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int64))
        df2.columns, df2.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.cooperations = df2
        # indicate whether "self.run()" was called or not
        self.ran = False

    def run(self):
        self.ran = True
        # on ne calcule que la diagonale de la matrice
        for i in range(0, len(self.strategies)):
            for j in range(i, len(self.strategies)):
                meet = Meeting(
                    self.game, self.strategies[i], self.strategies[j], self.length
                )
                meet.run()
                # on range le score d'une rencontre
                self.matrix.at[
                    self.strategies[i].name, self.strategies[j].name
                ] = meet.s1_score
                # et si les 2 sont différentes, on range aussi le symétrique
                if (i != j):
                    self.matrix.at[
                        self.strategies[j].name, self.strategies[i].name
                    ] = meet.s2_score
                # Idem pour les cooperations
                self.cooperations.at[
                    self.strategies[i].name, self.strategies[j].name
                ] = meet.nb_cooperation_s1
                if (i != j):
                    self.cooperations.at[
                        self.strategies[j].name, self.strategies[i].name
                    ] = meet.nb_cooperation_s2
        # On calcule le Total des gains pour pouvoir trier
        self.matrix["Total"] = self.matrix.sum(axis=1)
        self.matrix.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)
        self.cooperations["Total"] = self.cooperations.sum(axis=1)
        self.cooperations.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.cooperations.index) + ["Total"]
        self.cooperations = self.cooperations.reindex(columns=rows)

        
#  To use only with deterministic strategy sets. Nothing is repeated
class Ecological(Evaluator):
    def __init__(self, tournament, pop=100, max_iter=1000):
        self.pop = pop
        self.max_iter = max_iter
        self.tournament = tournament
        if not tournament.ran:
            print("warning: running passed Tournament. This step may take a while.")
            self.tournament.run()
        self.generation = 0  # Numéro de la génération actuelle
        self.historic = pd.DataFrame(columns= tournament.matrix.index)
        # Modifié pour que ça prenne l'ordre du resultat du Tournament comme dans Moran
        # avant c'était columns= [strat.name for strat in tournament.strategies])
        # Si on passe un entier, c'est la même population pour toutes les stratégies
        if type(pop) == int:
            self.historic.loc[0] = [pop for x in range(len(tournament.strategies))]
            self.base = pop * len(tournament.strategies)
        else :
        # sinon on utilise les populations passées en paramètre pour chaque stratégie
            assert len(pop)==len(tournament.strategies)
            self.historic.loc[0] = pop
            self.base = sum(pop)
        # On initialise les différents tableaux    
        self.extinctions = dict((s.name, math.inf) for s in tournament.strategies)
        self.cooperations = dict((s.name, 0) for s in tournament.strategies)
        self.listeCooperations = list()
        self.scores = dict((s.name, 0) for s in tournament.strategies)

    def run(self):
        dead = 0
        stab = False
        while (self.generation < self.max_iter) and (not stab):
            parents = list(copy.copy(self.historic.loc[self.generation]))
            # Calcul de la descendance d'une stratégie i face à toutes les autres
            for i in range(len(self.tournament.strategies)):
                strat = self.tournament.strategies[i].name
                if self.historic.at[self.generation, strat] != 0:
                    score = 0
                    cooperations = 0
                    # On cumule les points que l'on peut obtenir contre chaque famille, y compris la sienne
                    for j in range(len(self.tournament.strategies)):
                        strat2 = self.tournament.strategies[j].name
                        if self.historic.at[self.generation, strat2] != 0:
                            if i == j:
                                # quand on joue contre ses semblables, on ne joue pas contre sois-même
                                score += (
                                    self.historic.at[self.generation, strat] - 1
                                ) * self.tournament.matrix.at[strat, strat2]
                                cooperations += (
                                    self.historic.at[self.generation, strat] - 1
                                ) * self.tournament.cooperations.at[strat, strat2]
                            else:
                                # par contre, on joue contre tous les autres
                                score += (
                                    self.historic.at[self.generation, strat2]
                                    * self.tournament.matrix.at[strat, strat2]
                                )
                                cooperations += (
                                    self.historic.at[self.generation, strat2]
                                    * self.tournament.cooperations.at[strat, strat2]
                                )
                        # le tableau scores contient pour chaque
                        # stratégie le score cumulé qu'un représentant
                        # de cette stratégie obtient contre tous les
                        # représentants de toutes les autres
                        # stratégies
                        self.scores[strat] = score
                        self.cooperations[strat] = cooperations
            
            # total : tous les points distribués sur la population globale
            total = 0
            totalCooperations = 0
            for strat in self.tournament.strategies:
                total += (
                    self.scores[strat.name]
                    * self.historic.at[self.generation, strat.name]
                )
                totalCooperations += (
                    self.cooperations[strat.name]
                    * self.historic.at[self.generation, strat.name]
                )
            # calcul des nouvelles populations
            # Une fois qu'on a fait tous les cumuls on fait une règle de 3 pour se ramener à la même base
            for strat in self.tournament.strategies:
                parent = self.historic.at[self.generation, strat.name]
                if self.scores[strat.name] != 0:
                    self.historic.at[self.generation + 1, strat.name] = math.floor(
                        self.base * parent * self.scores[strat.name] / total
                    )
                # fitness est le score qu'une famille a obtenue
                # newpop = base * fitness / total
                elif self.scores[strat.name] == 0:
                    self.historic.at[self.generation + 1, strat.name] = 0
                    dead += 1
                if (parent != 0) and (
                    self.historic.at[self.generation + 1, strat.name] == 0
                ):
                    self.extinctions[strat.name] = self.generation + 1
                elif self.historic.at[self.generation + 1, strat.name] != 0:
                    self.extinctions[strat.name] = (
                        self.historic.at[self.generation + 1, strat.name] * 1000
                    )
                if dead == len(self.tournament.strategies) - 1:
                    stab = True
            #
            self.listeCooperations.append(
                totalCooperations / (self.base * self.tournament.length * len(self.tournament.strategies))
            )
            self.generation += 1
            if parents == list(self.historic.loc[self.generation]):
                stab = True
        # end of generation loop
        if self.generation==self.max_iter :
            print("Warning : max_iter reached")   
        trie = sorted(self.extinctions.items(), key=lambda t: t[1], reverse=True)
        df_trie = pd.DataFrame()
        for t in trie:
            df_trie[t[0]] = self.historic[t[0]]
        self.historic = df_trie

    def saveData(self):
        date = datetime.datetime.now()
        self.historic.to_csv(str(date) + ".csv", sep=";", encoding="utf-8")

    def drawPlot(self, nbCourbes=None, nbLegends=None, file='', title=''):
        nbCourbes = len(self.tournament.strategies) if (nbCourbes == None) else nbCourbes
        nbLegends = len(self.tournament.strategies) if (nbLegends == None) else nbLegends
        strat = self.historic.columns.tolist()
        for i in range(nbCourbes):
            plt.plot(
                self.historic[strat[i]],
                label=strat[i] if (i < nbLegends) else "_nolegend_",
            )
        plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.0)
        plt.ylabel("Populations")
        plt.xlabel("Generations")
        if title != '' :
            plt.title(title) 
        if file == '' :
            plt.show()
        # date = datetime.datetime.now()
        # plt.savefig(str(date)+'.png', dpi=1000)
        else : plt.savefig(file, dpi=1000)
        plt.close()
        

    def drawCooperation(self, file='', title=''):
        plt.plot(self.listeCooperations)
        if title != '' :
            plt.title(title)
        plt.ylabel("% of cooperation")
        plt.xlabel("Generations")
        plt.ylim(0, 101)
        if file == '' : plt.show()
        else : plt.savefig(file,dpi=1000)
        plt.close()
        


# Example
# t=Tournament(g,getClassicals()[0:6])
# t.run()
# t.matrix
# e=Ecological(t)
# e.run()
# e.historic
# e.historic.iloc[-1][0]
# e.drawPlot()
# e.drawCooperation()




#=========================================================================================
#
# Cette classe est quasi identique à Tournament excepté le faite
# qu'elle ne stocke pas les scores mais les victoires
#
# Ces deux classes pourraient sans doute être fusionnées histoire de
# factoriser le code, mais cela les rendraient sans doute plus
# difficiles à comprendre
# =========================================================================================


        
class TournamentVictory(Evaluator):
    def __init__(self, tournament):
        self.tournament = tournament
        self.matrix=tournament.matrix.iloc[:,:-1].copy()
        
    def run(self):
        for i in range(self.matrix.shape[0]) :
            for j in range(i+1) :
                self.matrix.iloc[i,j] = np.sign(self.tournament.matrix.iloc[i,j]-self.tournament.matrix.iloc[j,i])
                self.matrix.iloc[j,i] = -1 * self.matrix.iloc[i,j] 
        self.matrix["Total"] = self.matrix.sum(axis=1)
        self.matrix.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)



# Example
# t=Tournament(g,getClassicals()[3:7]
# t.run()
# v=TournamentVictory2(t)
# v.run()
# v.matrix



        

#=========================================================================================
# Repeated versions
#
# Quand on a des stratégies aléatoires, il est important de moyenner
# les résultats.
#
# Ces 2 classes permettent de calculer les tableaux de répétitions
#
# Attention dans la verion non répétée (précédente) la matrice
# contient un entier dans chaque case, tandis que dans cette version
# la matrice contient une liste de n valeurs Par ailleurs, ces
# versions sont paralléllisées
#=========================================================================================



#  For use only when there are some non-deterministic strategies in the set, and you want to repeat
class TournamentRepeat(Evaluator):
    def __init__(self, game, strategies, length=1000, repeat=1, flagCooperations=False):
        self.strategies = strategies
        self.game = game
        self.length = length
        self.repeat = repeat
        self.flagCooperations = flagCooperations
        self.size = len(strategies)
        # get coordinates of the upper triangular matrix:
        triU = np.triu(np.ones((self.repeat, self.size, self.size)))
        self.indexTriUp = np.argwhere(triU)
        # init the DataFrames:
        data = np.zeros((self.size, self.size+1), dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                data[i][j] = np.zeros(repeat)
        col = [s.name for s in self.strategies] + ["Total"]
        ind = [s.name for s in self.strategies]
        self.matrix = pd.DataFrame(data, columns=col, index=ind, dtype=object)
        if flagCooperations:
            self.cooperations = pd.DataFrame(deepcopy(data), columns=col, index=ind, dtype=object)
        # indicate whether "self.run()" was called or not
        self.ran = False

    def dispatched_meeting(self, index):
        # unpack index:
        x = index[0] # repeat index
        y = index[1] # strategy A
        z = index[2] # strategy B
        # init return struct:
        res_score = []
        if self.flagCooperations:
            res_cooperations = []
        # perform the meeting:
        meet = Meeting(self.game, self.strategies[y],
                       self.strategies[z], self.length)
        meet.run()
        # store score of the meeting:
        res_score.append(meet.s1_score)
        if (z != y):
            res_score.append(meet.s2_score)
        # store cooperations of the meeting:
        if self.flagCooperations:
            res_cooperations.append(meet.nb_cooperation_s1)
            if (z != y):
                res_cooperations.append(meet.nb_cooperation_s2)
        # return:
        if self.flagCooperations:
            return (index, res_score, res_cooperations)
        else:
            return (index, res_score)

    def run(self):
        self.ran = True
        # initialize workers:
        pool = multip.Pool()
        # launch & wait for processes:
        data = list(pool.map(self.dispatched_meeting, self.indexTriUp))
        # fill the data frames:
        for pack in data:
            # untpack:
            x = pack[0][0] # repeat index
            y = pack[0][1] # strategy A
            z = pack[0][2] # strategy B
            # add scores:
            self.matrix.at[self.strategies[y].name,
                           self.strategies[z].name][x] += pack[1][0]
            if (y != z):
                self.matrix.at[self.strategies[z].name,
                               self.strategies[y].name][x] += pack[1][1]
            if self.flagCooperations:
                self.cooperations.at[self.strategies[y].name,
                                     self.strategies[z].name][x] += pack[2][0]
                if (y != z):
                    self.cooperations.at[self.strategies[z].name,
                                         self.strategies[y].name][x] += pack[2][1]
        # manually terminate workers:
        pool.close()
        # compute totals:
        val_matrix = self.matrix[[s.name for s in self.strategies]].to_numpy()
        total_matrix = val_matrix.sum(axis=1)
        # statistics:
        for l in range(len(self.strategies)):
            self.matrix["Total"][l] = total_matrix[l].sum()
        # sort and reorganize the DF:
        self.matrix.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)
        # Repeat for cooperations
        if self.flagCooperations:
            # totals:
            val_cooperations = self.cooperations[[s.name for s in self.strategies]].to_numpy()
            total_cooperations = val_cooperations.sum(axis=1)
            # stats:
            for l in range(len(self.strategies)):
                self.cooperations["Total"][l] = total_cooperations[l].sum()
            # sort:
            self.cooperations.sort_values(by="Total", ascending=False, inplace=True)
            rows = list(self.cooperations.index) + ["Total"]
            self.cooperations = self.cooperations.reindex(columns=rows)
        
    def compute_statistics(self, func):
        """
        To aggregate pass an aggregation function from the library numpy as func.
        
        examples:
        >>> mean_matrix, mean_cooperations = selt.aggregate(np.mean)
        """
        assert callable(func)
        # aggregate data from the matrix:
        per_repeat_matrix = self.matrix[[s.name for s in self.strategies]].to_numpy().sum(axis=1)
        res_matrix = pd.DataFrame(index=[s.name for s in self.strategies])
        # apply desired function:
        for i in range(len(self.strategies)):
            res_matrix.at[self.strategies[i].name, 0] = func(per_repeat_matrix[i])
        # Repeat for cooperations:
        if self.flagCooperations:
            per_repeat_cooperations = self.cooperations[[s.name for s in self.strategies]].to_numpy().sum(axis=1)
            res_cooperations = pd.DataFrame(index=[s.name for s in self.strategies])
            for i in range(len(self.strategies)):
                res_cooperations.at[self.strategies[i].name, 0] = func(per_repeat_cooperations[i])
        # return
        if self.flagCooperations:
            return res_matrix, res_cooperations
        else:
            return res_matrix


#  For use only when there are some non-deterministic strategies in the set, and you want to repeat
class EcologicalRepeat(Evaluator):
    def __init__(self, tournament, pop=100, max_iter=1000):
        # parameters inherited from tournament:
        self.game = tournament.game
        self.strategies = tournament.strategies
        self.length = tournament.length
        self.flagCooperations = tournament.flagCooperations
        self.repeat = tournament.repeat
        self.tournament = tournament
        if not tournament.ran:
            print("warning: running passed Tournament. This step may take a while.")
            self.tournament.run()
        # parameters for the ecological competition
        self.pop = pop
        self.max_iter = max_iter
        self.generation = 0  # Numéro de la génération actuelle
        self.historic = pd.DataFrame(columns=[strat.name for strat in self.strategies], dtype=object)
        if type(pop) == int:
            self.historic.loc[0] = [np.full(self.repeat, pop) for x in range(len(self.strategies))]
            self.base = pop * len(self.strategies)
        else :
            assert len(pop)==len(self.strategies)
            self.historic.loc[0] = pop
            self.base = sum(pop)
        self.extinctions = dict((s.name, np.array([np.inf for k in range(self.repeat)])) for s in self.strategies)
        self.scores = dict((s.name, 0) for s in self.strategies)
        if self.flagCooperations:
            self.listeCooperations = list()
            self.cooperations = dict((s.name, 0) for s in self.strategies)
        
    def run(self):
        dead = np.zeros(self.repeat)
        stab = np.full(self.repeat, False)
        # the 'generation' loop:
        while self.generation<self.max_iter and not stab.all():
            parents = [np.array([strat[k] for strat in self.historic.loc[self.generation].to_numpy()])
                       for k in range(self.repeat)]              
            for i in range(len(self.strategies)):
                strat = self.strategies[i].name
                score = 0
                cooperations = 0
                # compute scores and cooperations values for strat:
                for j in range(len(self.strategies)):
                    strat2 = self.strategies[j].name
                    if i == j:
                        # when playing against the same strategy,
                        # an individual cannot play with itself --> '-1'
                        score += (np.clip(self.historic.at[self.generation, strat]-1, 0, None)
                            * self.tournament.matrix.at[strat, strat2])
                        if self.flagCooperations:
                            cooperations += (np.clip(self.historic.at[self.generation, strat]-1, 0, None)
                                * self.tournament.cooperations.at[strat, strat2])
                    else:
                        score += (self.historic.at[self.generation, strat2]
                            * self.tournament.matrix.at[strat, strat2])
                        if self.flagCooperations:
                            cooperations += (self.historic.at[self.generation, strat2]
                                * self.tournament.cooperations.at[strat, strat2])
                # store the computed values (overwriten each generation):
                self.scores[strat] = score
                if self.flagCooperations:
                    self.cooperations[strat] = cooperations

            # compute total scores and cooperations values:
            total = np.zeros(self.repeat)
            if self.flagCooperations:
                totalCooperations = np.zeros(self.repeat)
            for strat in self.strategies:
                total += (self.scores[strat.name]
                    * self.historic.at[self.generation, strat.name])
                if self.flagCooperations:
                    totalCooperations += (self.cooperations[strat.name]
                        * self.historic.at[self.generation, strat.name])
            
            # compute new population distribution:
            for strat in self.strategies:
                parent = self.historic.at[self.generation, strat.name]
                # 'regle de trois':
                self.historic.at[self.generation + 1, strat.name] = np.floor(
                    self.base * parent * self.scores[strat.name] / total) # Runtime Warning: scalar overflow
                # count deaths:
                dead += self.scores[strat.name]==0
                ### TODO: vectorize computations:
                for k in range(self.repeat):
                    # count extinctions:
                    if parent[k] != 0 and self.historic.at[self.generation + 1, strat.name][k] == 0:
                        self.extinctions[strat.name][k] = self.generation + 1
                    elif self.historic.at[self.generation + 1, strat.name][k] != 0:
                        self.extinctions[strat.name][k] = (
                            self.historic.at[self.generation + 1, strat.name][k] * 1000)
                    # stability check:
                    if dead[k] == len(self.strategies) - 1:
                        stab[k] = True
            if self.flagCooperations:
                # update listeCooperations:
                self.listeCooperations.append(
                    totalCooperations / (self.base * self.length * len(self.strategies)))
            # update generation counter:
            self.generation += 1
            # stability check:
            for k in range(self.repeat):
                new_generation = np.array([strat[k] for strat in self.historic.loc[self.generation].to_numpy()])
                stab[k] = np.array_equal(parents[k], new_generation)
        # end of generation loop
        if self.generation==self.max_iter :
            print("Warning : max_iter reached")   
        # sort by extinction date:
        trie = sorted(self.extinctions.items(), key=lambda t: t[1].mean(), reverse=True)
        df_trie = pd.DataFrame()
        for t in trie:
            df_trie[t[0]] = self.historic[t[0]]
        self.historic = df_trie
        return self.historic

    def saveData(self):
        date = datetime.datetime.now()
        self.historic.to_csv(str(date) + ".csv", sep=";", encoding="utf-8")

    def drawPlot(self, nbCourbes=None, nbLegends=None, beam=None, file='', title=''):
        # init param:
        nbCourbes = len(self.strategies) if (nbCourbes == None) else nbCourbes
        nbLegends = len(self.strategies) if (nbLegends == None) else nbLegends
        if beam==None:
            beam = [True for k in range(nbCourbes)]
        elif isinstance(beam, int):
            beam = [True if k<nbLegends else False for k in range(nbLegends)]
        elif len(beam) < nbCourbes:
            beam += [False]*(nbCourbes-len(beam))
        # extract strategies sorted:
        strats = self.historic.columns.tolist()
        # figure:
        for i in range(nbCourbes):
            # get average population evolution and std:
            avg = np.zeros(self.generation)
            std = np.zeros(self.generation)
            for j in range(self.generation):
                avg[j] = np.mean(self.historic.at[j, strats[i]])
                std[j] = np.std(self.historic.at[j, strats[i]])
            # plot:
            line = plt.plot(avg, label=strats[i] if i<nbLegends else "_nolegend_")
            if beam[i]:
                # plot beam:
                color = line[0].get_color()
                plt.fill_between(np.arange(self.generation), avg+std, avg-std, color=color, alpha=0.2)
        # style:
        plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.0)
        plt.ylabel("Populations")
        plt.xlabel("Generations")
        if title != '' :
            plt.title(title)
        plt.tight_layout() 
        # save/display:
        if file == '' :
            plt.show()
        else:
            plt.savefig(file, dpi=1000)
            plt.close()

    def drawBoxplot(self, nbBoxes=None, file='', title=''):
        # init params:
        nbBoxes = len(self.strategies) if (nbBoxes == None) else nbBoxes
        # extract data:
        strats = self.historic.columns.tolist()
        to_plot=[]
        ticks=[]
        # init figure:
        fig,ax = plt.subplots()
        for i in range(nbBoxes):
            to_plot.append(self.historic.at[self.generation, strats[i]])
            ticks.append(strats[i])
        # box plot & style:
        bx = ax.boxplot(to_plot, showmeans=True, meanline=True)
        # fig style:
        ax.set_xticklabels(ticks, rotation='vertical', fontsize='xx-small')
        ax.set(xlabel='strategies', ylabel='population')
        if title != '' :
            fig.suptitle(title)
        fig.tight_layout()
        # save/display:
        if file == '' :
            fig.show()
        else :
            fig.savefig(file, dpi=1000)
            plt.close()
        
    def drawCooperation(self, file='', title=''):
        assert self.flagCooperations
        #
        plt.plot(self.listeCooperations)
        if title != '' :
            plt.title(title)
        plt.ylabel("% of cooperation")
        plt.xlabel("Generations")
        plt.ylim(0, 101)
        if file == '' : plt.show()
        else : plt.savefig(file,dpi=1000)
        plt.close()

    def compute_statistics(self, func):
        """
        To aggregate pass an aggregation function from the library numpy as func.
        
        examples:
        >>> mean_matrix, mean_cooperations = selt.aggregate(np.mean)
        """
        assert callable(func)
        # extract sorted strategies:
        strats = self.historic.columns.tolist()
        # init data structure
        res = pd.DataFrame(index=[s for s in strats])
        # apply desired function:
        for s in strats:
            res.at[s, 0] = func(self.historic.at[self.generation, s])
        # return
        return res



# Example
# Avec repeat=1 et des strats déterministes, ça doit donner exactement le même résultat que les versions classiques
# t=Tournament(g,getClassicals()[0:6])
# r=TournamentRepeat(g,getClassicals()[0:6])
# t.run()
# r.run()
# t.matrix['Total']
# r.matrix['Total']

# Et dès qu'on rajoute une non-deterministe on peut afficher les faisceaux
# r=TournamentRepeat(g,getClassicals()[0:6]+[Lunatic()], repeat=100)
# f=EcologicalRepeat(r)
# f.run()
# f.historic
# f.historic.iloc[-1][0]
# f.drawPlot(nbCourbes=len(f.strategies), nbLegends=len(strategies),beam=[True]*3)


