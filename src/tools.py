import pandas as pd
import numpy as np
import random
import itertools
from ipd import *


# Toutes les combinaisons de n stratégies prises dans la soupe
# une stratégie ne participe donc pas à tout, mais toutes les stratégies participent exactement à autant de compétitions au total.
def subClasses(soup, n, length=1000):
    if n > len(soup):
        print("The soup size must be smaller than n")
        return
    res = pd.DataFrame(
        np.nan, [s.name for s in soup], ["BestRank", "WorstRank", "RankAvg", "RankStd"]
    )
    for s in soup:
        res.at[s.name, "BestRank"] = len(soup)
    ranks = dict()
    sousEnsembles = list(itertools.combinations(soup, n))
    for s in sousEnsembles:
        e = Ecological(g, s, length)
        e.run()
        classements = e.historic.iloc[e.generation].rank(
            0, method="min", ascending=False
        )
        for strat in s:
            classement = classements[strat.name]
            if (
                math.isnan(res.at[strat.name, "BestRank"])
                or classement < res.at[strat.name, "BestRank"]
            ):
                res.at[strat.name, "BestRank"] = classement
            if (
                math.isnan(res.at[strat.name, "WorstRank"])
                or classement > res.at[strat.name, "WorstRank"]
            ):
                res.at[strat.name, "WorstRank"] = classement
            if strat.name in ranks.keys():
                ranks[strat.name].append(classement)
            if strat.name not in ranks.keys():
                ranks[strat.name] = [classement]
    for strat in soup:
        res.at[strat.name, "RankAvg"] = statistics.mean(ranks[strat.name])
        res.at[strat.name, "RankStd"] = statistics.stdev(ranks[strat.name])
    print(
        res.sort_values(
            by=["RankAvg", "BestRank", "RankStd", "WorstRank"],
            ascending=[True, True, True, True],
        )
    )


# toutes les combinaisons de n stratégies prises dans la soupe avec systématiquement une stratégie ajoutée
# La stratégie ajoutée est toujours présente et est donc confrontée à des environnements variés
def subClassesWithOneStrat(soup, n, strategy, printAll=False, length=1000):
    if n > len(soup):
        print("The soup size must be smaller than n")
        return
    res = pd.DataFrame(
        np.nan,
        [s.name for s in soup + [strategy]],
        ["BestRank", "WorstRank", "RankAvg", "RankStd"],
    )
    sousEnsembles = list(itertools.combinations(soup, n))
    ranks = dict()
    bestComp = []
    worstComp = []
    for s in sousEnsembles:
        e = Ecological(g, list(s) + [strategy], length)
        e.run()
        classements = e.historic.iloc[e.generation].rank(
            0, method="min", ascending=False
        )
        for strat in list(s) + [strategy]:
            classement = classements[strat.name]
            if (
                math.isnan(res.at[strat.name, "BestRank"])
                or classement < res.at[strat.name, "BestRank"]
            ):
                res.at[strat.name, "BestRank"] = classement
                if strat == strategy:
                    bestComp = list(s) + [strategy]
            if (
                math.isnan(res.at[strat.name, "WorstRank"])
                or classement > res.at[strat.name, "WorstRank"]
            ):
                res.at[strat.name, "WorstRank"] = classement
                if strat == strategy:
                    worstComp = list(s) + [strategy]
            if strat.name in ranks.keys():
                ranks[strat.name].append(classement)
            if strat.name not in ranks.keys():
                ranks[strat.name] = [classement]
    for s in soup + [strategy]:
        if s.name in ranks.keys():
            res.at[s.name, "RankAvg"] = statistics.mean(ranks[s.name])
            if len(ranks[s.name]) > 1:
                res.at[s.name, "RankStd"] = statistics.stdev(ranks[s.name])
    if printAll:
        print(
            res.sort_values(
                by=["RankAvg", "BestRank", "RankStd", "WorstRank"],
                ascending=[True, True, True, True],
            )
        )
    else:
        print("Ranking of " + strategy.name)
        print(res.loc[strategy.name, :])
    return bestComp, worstComp, ranks, strategy



# p compétitions de n stratégies prises au hasard, avec une stratégie systématiquement ajoutée
# Comparativement à la méthode précédente, celle ci est utile quand le nombre de combinaisons explose
def subClassesRandomWithOneStrat(p, soup, n, strategy, printAll=False, length=1000):
    if n > len(soup):
        "the soup size must be smaller than n"
        return
    res = pd.DataFrame(
        np.nan,
        [s.name for s in soup + [strategy]],
        ["BestRank", "WorstRank", "RankAvg", "RankStd"],
    )
    ranks = dict()
    bestComp = []
    worstComp = []
    for i in range(0, p):
        # print("Competition "+str(i+1)+ "/"+str(p))
        strategies = []
        strategies.append(strategy)
        indice = [i for i in range(0, len(soup))]
        for i in range(0, n):
            indiceStrat = random.choice(indice)
            indice.remove(indiceStrat)
            strategies.append(soup[indiceStrat])
        # print("Les stratégies qui jouent sont : ")
        # for s in strategies :
        # print(s.name)
        e = Ecological(g, strategies, length)
        e.run()
        classements = e.historic.iloc[e.generation].rank(
            0, method="min", ascending=False
        )
        for strat in strategies:
            classement = classements[strat.name]
            if (
                math.isnan(res.at[strat.name, "BestRank"])
                or classement < res.at[strat.name, "BestRank"]
            ):
                res.at[strat.name, "BestRank"] = classement
                if strat == strategy:
                    bestComp = strategies
            if (
                math.isnan(res.at[strat.name, "WorstRank"])
                or classement > res.at[strat.name, "WorstRank"]
            ):
                res.at[strat.name, "WorstRank"] = classement
                if strat == strategy:
                    worstComp = strategies
            if strat.name in ranks.keys():
                ranks[strat.name].append(classement)
            if strat.name not in ranks.keys():
                ranks[strat.name] = [classement]
    for s in soup + [strategy]:
        if s.name in ranks.keys():
            res.at[s.name, "RankAvg"] = statistics.mean(ranks[s.name])
            if len(ranks[s.name]) > 1:
                res.at[s.name, "RankStd"] = statistics.stdev(ranks[s.name])
    if printAll:
        print(
            res.sort_values(
                by=["RankAvg", "BestRank", "RankStd", "WorstRank"],
                ascending=[True, True, True, True],
            )
        )
    else:
        print("Strategy ranking  : " + strategy.name)
        print(res.loc[strategy.name, :])
    return bestComp, worstComp, strategy

