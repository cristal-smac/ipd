import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import statistics
import random
import itertools
import datetime

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


class Tournament(Evaluator):
    def __init__(self, game, strategies, length=1000):
        self.strategies = strategies
        self.game = game
        self.length = length
        size = len(strategies)
        df = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int32))
        df.columns, df.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.matrix = df
        df2 = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int32))
        df2.columns, df2.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.cooperations = df2

    def run(self):
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
        # A MON AVIS Y'A UN BUG ICI : ON DEVRAIT TRIER SUR LE TOTAL DES SCORES (et pas des cooperations) ?????  A REVOIR
        self.cooperations["Total"] = self.cooperations.sum(axis=1)
        self.cooperations.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.cooperations.index) + ["Total"]
        self.cooperations = self.cooperations.reindex(columns=rows)

        
class TournamentVictory(Evaluator):
    def __init__(self, game, strategies, length=1000, repeat=1):
        self.strategies = strategies
        self.game = game
        self.length = length
        self.repeat = repeat
        size = len(strategies)
        df = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int32))
        df.columns, df.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )
        self.matrix = df
        df2 = pd.DataFrame(np.zeros((size, size + 1), dtype=np.int32))
        df2.columns, df2.index = (
            [s.name for s in self.strategies] + ["Total"],
            [s.name for s in self.strategies],
        )

    def run(self):
        for k in range(self.repeat):
            for i in range(0, len(self.strategies)):
                for j in range(i+1, len(self.strategies)):
                    meet = Meeting(
                        self.game, self.strategies[i], self.strategies[j], self.length
                    )
                    meet.run()
                    if meet.s1_score > meet.s2_score :
                        self.matrix.at[
                            self.strategies[i].name, self.strategies[j].name
                        ] += 1
                        self.matrix.at[
                            self.strategies[j].name, self.strategies[i].name
                        ] -= 1
                    elif  meet.s2_score > meet.s1_score : 
                        self.matrix.at[
                            self.strategies[j].name, self.strategies[i].name
                        ] += 1
                        self.matrix.at[
                            self.strategies[i].name, self.strategies[j].name
                        ] -= 1               
        self.matrix["Total"] = self.matrix.sum(axis=1)
        self.matrix.sort_values(by="Total", ascending=False, inplace=True)
        rows = list(self.matrix.index) + ["Total"]
        self.matrix = self.matrix.reindex(columns=rows)

        
        
class Ecological(Evaluator):
    def __init__(self, tournament, pop=100, max_iter=1000):
        self.pop = pop
        self.max_iter = max_iter
        self.tournament = tournament
        self.generation = 0  # Numéro de la génération actuelle
        self.historic = pd.DataFrame(columns=[strat.name for strat in tournament.strategies])
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
            self.listeCooperations.append(
                totalCooperations / (self.base * self.tournament.length * len(self.tournament.strategies))
            )
            self.generation += 1
            if parents == list(self.historic.loc[self.generation]):
                stab = True
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
        



