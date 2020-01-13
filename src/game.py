import numpy as np
import pandas as pd
import math


class Game:
    def __init__(self, tab, actions, actions2=[], asymetrical=False):
        self.actions = actions
        m = np.array(tab, dtype=[("x", object), ("y", object)])
        if asymetrical:
            self.size = len(actions)
            self.size2 = len(actions2)
            self.actions2 = actions2
            self.scores = m.reshape(self.size, self.size2)
        else:
            self.size = int(math.sqrt(len(tab)))
            self.scores = m.reshape(self.size, self.size)
        self.asymetrical = asymetrical

    def getNash(self):
        max_x = np.matrix(self.scores["x"].max(0)).repeat(self.size, axis=0)
        bool_x = self.scores["x"] == max_x
        if self.asymetrical:
            max_y = (
                np.matrix(self.scores["y"].max(1))
                .transpose()
                .repeat(self.size2, axis=1)
            )
        else:
            max_y = (
                np.matrix(self.scores["y"].max(1)).transpose().repeat(self.size, axis=1)
            )
        bool_y = self.scores["y"] == max_y
        bool_x_y = bool_x & bool_y
        result = np.where(bool_x_y == True)
        listOfCoordinates = list(zip(result[0], result[1]))
        return listOfCoordinates

    def isPareto(self, t, s):
        return (
            True
            if (len(s) == 0)
            else (s[0][0] <= t[0] or s[0][1] <= t[1]) and self.isPareto(t, s[1:])
        )

    def getPareto(self):
        x = 0
        y = 0
        res = list()
        liste = self.scores.flatten()
        for s in liste:
            if x == self.size:
                x = 0
                y = y + 1
            if self.isPareto(s, liste):
                res.append((x, y))
            x = x + 1
        return res

    def getDominantStrategies(self, strict=True):
        dominatedLines = []
        dominatedColumns = []
        findDominated = True
        while (
            findDominated
            and (len(dominatedLines) != self.size - 1)
            and (len(dominatedColumns) != self.size - 1)
        ):
            findDominated = False
            # on regarde les lignes dominées
            for i in range(self.size - 1):
                line1 = self.scores["x"][i]
                line2 = self.scores["x"][i + 1]
                if self.compare(line1, line2, dominatedColumns, strict):
                    if i not in dominatedLines:
                        dominatedLines += [i]
                        findDominated = True
                if self.compare(line2, line1, dominatedColumns, strict):
                    if i + 1 not in dominatedLines:
                        dominatedLines += [i + 1]
                        findDominated = True
            # on regarde les colonnes dominées

            if self.asymetrical:
                lenY = self.size2
            else:
                lenY = self.size
            for i in range(lenY - 1):
                c1 = self.scores["y"].transpose()[i]
                c2 = self.scores["y"].transpose()[i + 1]
                if self.compare(c1, c2, dominatedLines, strict):
                    if i not in dominatedColumns:
                        dominatedColumns += [i]
                        findDominated = True
                if self.compare(c2, c1, dominatedLines, strict):
                    if i + 1 not in dominatedColumns:
                        dominatedColumns += [i + 1]
                        findDominated = True
        return self.result(dominatedLines, dominatedColumns)

    def compare(self, l1, l2, tab, strict):
        dominated = True
        for i in range(self.size):
            if strict:
                if (l1[i] < l2[i] and i not in tab) or i in tab:
                    dominated = dominated and True
                else:
                    dominated = dominated and False
            else:
                if (l1[i] <= l2[i] and i not in tab) or i in tab:
                    dominated = dominated and True
                else:
                    dominated = dominated and False
        return dominated

    def result(self, dominatedLines, dominatedColumns):
        x = list()
        y = list()
        res = list()
        tab = list()
        colums = list()
        rows = list()
        for i in range(self.size):
            if i not in dominatedLines:
                x.append(i)
                colums.append(self.actions[i])
        if self.asymetrical:
            lenY = self.size2
        else:
            lenY = self.size
        for i in range(lenY):
            if i not in dominatedColumns:
                y.append(i)
                if self.asymetrical:
                    rows.append(self.actions2[i])
                else:
                    rows.append(self.actions[i])
        for indX in x:
            for indY in y:
                res.append((indX, indY))
                tab.append(self.scores[indX][indY])
        print(res)
        return Game(tab, colums, rows, True)

    def prettyPrint(self):
        game = pd.DataFrame(np.nan, self.actions, self.actions, dtype=object)
        for i in range(self.size):
            for j in range(self.size):
                game.iat[i, j] = self.scores[i][j]
        print(game)
