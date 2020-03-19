import itertools
import numpy as np
import collections


class Strategy:
    def setMemory(self, mem):
        pass

    def getAction(self, tick):
        pass

    def __copy__(self):
        pass

    def update(self, x, y):
        pass


class Periodic(Strategy):
    def __init__(self, sequence, name=None):
        super().__init__()
        self.sequence = sequence.upper()
        self.step = 0
        self.name = "per_" + sequence if (name is None) else name

    def getAction(self, tick):
        return self.sequence[tick % len(self.sequence)]

    def clone(self):
        object = Periodic(self.sequence, self.name)
        return object

    def update(self, x, y):
        pass


class Tft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "tft"
        self.hisPast = ""

    def getAction(self, tick):
        return "C" if (tick == 0) else self.hisPast[-1]

    def clone(self):
        return Tft()

    def update(self, my, his):
        self.hisPast += his


class Tf2t(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "tf2t"
        self.hisPast = ""

    def getAction(self, tick):
        if tick == 0 or tick == 1:
            return "C"
        else:
            if self.hisPast[-1] == "D" and self.hisPast[-2]:
                return "D"
            else:
                return "C"

    def clone(self):
        return Tf2t()

    def update(self, my, his):
        self.hisPast += his


class Hardtft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "hardtft"
        self.hisPast = ""

    def getAction(self, tick):
        if tick == 0 or tick == 1:
            return "C"
        else:
            if self.hisPast[-1] == "D" or self.hisPast[-2] == "D":
                return "D"
            else:
                return "C"

    def clone(self):
        return Hardtft()

    def update(self, my, his):
        self.hisPast += his


class Slowtft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "slowtft"
        self.hisPast = ""
        self.myPast = ""

    def getAction(self, tick):
        if tick == 0 or tick == 1:
            return "C"
        else:
            if self.hisPast[-1] == "D" and self.hisPast[-2] == "D":
                return "D"
            elif self.hisPast[-1] == "C" and self.hisPast[-2] == "C":
                return "C"
            else:
                return self.myPast[-1]

    def clone(self):
        return Slowtft()

    def update(self, my, his):
        self.hisPast += his
        self.myPast += my


class Spiteful(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "spiteful"
        self.hisPast = ""
        self.myPast = ""

    def getAction(self, tick):
        if tick == 0:
            return "C"
        if self.hisPast[-1] == "D" or self.myPast[-1] == "D":
            return "D"
        else:
            return "C"

    def clone(self):
        return Spiteful()

    def update(self, my, his):
        self.myPast += my
        self.hisPast += his


class Mistrust(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "mistrust"
        self.hisPast = ""

    def getAction(self, tick):
        if tick == 0:
            return "D"
        return self.hisPast[-1]

    def clone(self):
        return Mistrust()

    def update(self, my, his):
        self.hisPast += his


class SoftMajority(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "softmajo"
        self.nbCooperations = 0
        self.nbTrahisons = 0

    def getAction(self, tick):
        if self.nbCooperations >= self.nbTrahisons:
            return "C"
        else:
            return "D"

    def clone(self):
        return SoftMajority()

    def update(self, my, his):
        if his == "C":
            self.nbCooperations += 1
        elif his == "D":
            self.nbTrahisons += 1


class HardMajority(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "hardmajo"
        self.nbCooperations = 0
        self.nbTrahisons = 0

    def getAction(self, tick):
        if self.nbCooperations > self.nbTrahisons:
            return "C"
        else:
            return "D"

    def clone(self):
        return HardMajority()

    def update(self, my, his):
        if his == "C":
            self.nbCooperations += 1
        elif his == "D":
            self.nbTrahisons += 1


class Gradual(Strategy):
    def __init__(self, hard=True):
        super().__init__()
        self.name = "gradual_"+("hard" if hard else "soft")
        self.nbTrahisons = 0
        self.punish = 0
        self.calm = 0
        self.hisLast=""
        self.hard=hard

    def getAction(self, tick):
        if tick == 0:
            return "C"
        if self.punish > 0:
            self.punish -= 1
            if (self.hard and self.hisLast == "D") : self.nbTrahisons+=1
            return "D"
        if self.calm > 0:
            self.calm -= 1
            if (self.hard and self.hisLast == "D") : self.nbTrahisons+=1
            return "C"
        if self.hisLast == "D" : self.nbTrahisons+=1
        if self.hisLast == "D":
            self.punish = self.nbTrahisons - 1
            self.calm = 2
            return "D"
        return "C"

    def clone(self):
        return Gradual(self.hard)

    def update(self, my, his):
        self.hisLast = his


class Mem(Strategy):
    def __init__(self, x, y, genome, name=None):
        assert max(x,y)+(2**(x+y)) == len(genome), "incorrect genotype size"
        self.name = name
        self.x = x
        self.y = y
        self.genome = genome
        if name is None:  # Nom par défaut si l'utilisateur ne le définit pas
            self.name = genome
        self.myMoves = collections.deque(maxlen=x)  # contains my x last moves
        self.itsMoves = collections.deque(maxlen=y)  # contains its y last moves

    def clone(self, name=None):
        if name is None:
            return Mem(self.x, self.y, self.genome, self.name)
        else :
            return Mem(self.x, self.y, self.genome, name)

    def getAction(self, tick):
        if (tick == 0 and len(self.myMoves)>0) :
            print("Strategy not reinitialized", flush=True)
        if tick < max(self.x, self.y):
            return self.genome[tick]
        cpt = 0
        for i in range(0, self.x):
            cpt *= 2
            if self.myMoves[i] == "D":
                cpt += 1
        for i in range(0, self.y):
            cpt *= 2
            if self.itsMoves[i] == "D":
                cpt += 1
        cpt += max(self.x, self.y)
        return self.genome[cpt]

    def update(self, myMove, itsMove):
        self.myMoves.append(myMove)
        self.itsMoves.append(itsMove)


class Prober(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "prober"
        self.hisPast = ""

    def getAction(self, tick):
        if tick == 0:
            return "D"
        elif tick == 1 or tick == 2:
            return "C"
        else:
            if self.hisPast[1] == "C" and self.hisPast[2] == "C":
                return "D"
            else:
                return self.hisPast[-1]

    def clone(self):
        return Prober()

    def update(self, my, his):
        self.hisPast += his


class Pavlov(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "pavlov"
        self.hisPast = ""
        self.myPast = ""

    def getAction(self, tick):
        if tick == 0:
            return "C"
        else:
            if self.hisPast[-1] == self.myPast[-1]:
                return "C"
            else:
                return "D"

    def clone(self):
        return Pavlov()

    def update(self, my, his):
        self.hisPast += his
        self.myPast += my


#random.seed(0)
#np.random.seed(0)
class Lunatic(Strategy):
    def __init__(self, proba=0.5):
        super().__init__()
        self.name = "lunatic"
        self.proba = proba

    def getAction(self, tick):
        return np.random.choice(["C","D"],p=[self.proba, 1-self.proba])

    def clone(self):
        return Lunatic(self.proba)

#    def update(self, my, his):
#        return


def getMem(x, y):
    if x + y > 4:
        return "Pas calculable"
    len_genome = max(x, y) + 2 ** (x + y)
    permut = [p for p in itertools.product(["C", "D"], repeat=len_genome)]
    genomes = ["".join(p) for p in permut]
    return [Mem(x, y, gen) for gen in genomes]


def getPeriodics(n):
    cards = ["C", "D"]
    periodics = list()
    for i in range(n + 1):
        periodics += [p for p in itertools.product(cards, repeat=i)]
    strats = [Periodic("".join(p)) for p in periodics]
    return strats[1:]


def getClassicals():
    return [
        Periodic("C"),
        Periodic("D"),
        Tft(),
        Spiteful(),
        SoftMajority(),
        HardMajority(),
        Periodic("DDC"),
        Periodic("CCD"),
        Mistrust(),
        Periodic("CD"),
        Pavlov(),
        Tf2t(),
        Hardtft(),
        Slowtft(),
        Gradual(),
        Prober(),
    ]


class MetaStrategy(Strategy):
    def __init__(self, bag, n):
        super().__init__()
        self.name = "metastrat"
        self.bag = bag
        self.n = n
        self.scores = [0 for i in range(len(bag))]
        self.cpt = -1
        self.nbPlayed = [0 for i in range(len(bag))]

    def getAction(self, tick):
        if tick < self.n * len(self.bag):
            if tick % self.n == 0:
                self.cpt = (self.cpt + 1) % len(self.bag)
        else:
            if tick % self.n == 0:
                self.cpt = np.argmax(self.scores)
        #print("Playing : "+self.bag[self.cpt].name)
        #print(self.nbPlayed[self.cpt])
        return self.bag[self.cpt].getAction(self.nbPlayed[self.cpt])

    def clone(self):
        return MetaStrategy(self.bag, self.n)

    def update(self, my, his):
        if his == "C" and my == "C":
            self.scores[self.cpt] = self.scores[self.cpt] + 3
        elif his == "D" and my == "D":
            self.scores[self.cpt] = self.scores[self.cpt] + 1
        elif his == "D" and my == "C":
            self.scores[self.cpt] = self.scores[self.cpt] + 5
        self.bag[self.cpt].update(my, his)
        self.nbPlayed[self.cpt] += 1
