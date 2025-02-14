import itertools
import numpy as np
import collections


class Strategy:
    """Basic Strategy Interface"""
    def setMemory(self, mem):
        pass

    def getAction(self, tick: int) -> str:
        """
        Returns the action taken by the strategy at a given tick.
        tick (int) : turn number of the game
        """
        pass

    def __copy__(self):
        pass

    def update(self, x: str, y: str) -> None:
        pass


class Periodic(Strategy):
    """Periodic strategy
    -----------------
    Given a sequence, will give each character of the sequence and repeat itself.

    Parameters
    ----------
    sequence : `str`
        the sequence it should iterate through. Must be composed by "C" and "D".
    name : `str`, optional
        the name given to this strategy.

    Notes
    -----
    To make a "Always Defect" or "Always Cooperate" strategy, simply use `Periodic("D")` or `Periodic("C")`.
    """
    def __init__(self, sequence: str, name: str=None):
        super().__init__()
        self.sequence = sequence.upper()
        self.step = 0
        self.name = "per_" + sequence if (name is None) else name

    def getAction(self, tick: int) -> str:
        # for each tick, will return a char of the sequence, and loop itself
        return self.sequence[tick % len(self.sequence)]

    def clone(self):
        object = Periodic(self.sequence, self.name)
        return object

    def update(self, x, y):
        pass


class Tft(Strategy):
    """Tit-For-Tat strategy
    --------------------
    Will cooperate on the first turn.
    Then will copy the action done by the opponent's last turn.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |
    |-----------|-----|-----|-----|-----|-----|
    | Tft       |  C  |  C  |  D  |  C  |  C  |
    | Opponent  |  C  |  D  |  C  |  C  |  C  |
    """

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
    """Tit-For-2Tat strategy
    --------------------
    Will cooperate on the two first turn.
    If the opponent defects two consecutive times, will defect.
    Else, cooperate.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |
    |-----------|-----|-----|-----|-----|-----|
    | Tf2t      |  C  |  C  |  C  |  D  |  C  |
    | Opponent  |  C  |  D  |  D  |  C  |  C  |
    """
    def __init__(self):
        super().__init__()
        self.name = "tf2t"
        self.hisPast = ""

    def getAction(self, tick):
        if tick == 0 or tick == 1:
            return "C"
        else:
            if self.hisPast[-1] == "D" and self.hisPast[-2] == "D":
                return "D"
            else:
                return "C"

    def clone(self):
        return Tf2t()

    def update(self, my, his):
        self.hisPast += his


class HardTft(Strategy):
    """Hard Tit-For-Tat strategy
    --------------------
    Will cooperate on the first turn.
    If the opponent has defected on the last or the second-last turn, will defect.
    Else, cooperate.

    Notes
    -----
    If defected, he will always defect one more time than the opponent.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
    |-----------|-----|-----|-----|-----|-----|-----|-----|-----|
    | HardTft   |  C  |  C  |  D  |  D  |  D  |  C  |  D  |  D  |
    | Opponent  |  C  |  D  |  C  |  D  |  C  |  C  |  D  |  C  |
    """
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
        return HardTft()

    def update(self, my, his):
        self.hisPast += his


class SlowTft(Strategy):
    """Slow Tit-For-Tat strategy
    --------------------
    Will cooperate the first two turns.
    If the opponent's two last moves were the same, copy the opponent's last move.
    Else, do its last move.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |-----------|-----|-----|-----|-----|-----|-----|-----|
    | SlowTft   |  C  |  C  |  C  |  D  |  D  |  C  |  C  |
    | Opponent  |  C  |  D  |  D  |  C  |  C  |  D  |  C  |
    """

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
        return SlowTft()

    def update(self, my, his):
        self.hisPast += his
        self.myPast += my


class Spiteful(Strategy):
    """Spiteful strategy
    --------------------
    Will cooperate on the first turn.
    If the opponent defects just one time, he will defect every time.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |
    |-----------|-----|-----|-----|-----|-----|
    | Spiteful  |  C  |  C  |  D  |  D  |  D  |
    | Opponent  |  C  |  D  |  C  |  D  |  C  |
    """
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
    """Mistrust strategy
    --------------------
    Will defect on the first turn.
    Then will copy the action done by the opponent's last turn.

    Notes
    -----
    Is the "defect" counterpart of `Tit-For-Tat`.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |
    |-----------|-----|-----|-----|-----|-----|-----|
    | Mistrust  |  D  |  C  |  D  |  C  |  C  |  D  |
    | Opponent  |  C  |  D  |  C  |  C  |  D  |  D  |
    """

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
    """Soft Majority strategy
    -------------------------
    Will count the number of cooperation and defection of the opponent.
    If they are an equal number or more cooperation, will cooperate.
    Else, will defect.

    Notes
    -----
    Will always cooperate on the first turn.
    Is the "cooperate" counterpart of the `Hard Majority` strategy.

    Interactions
    ------------
    | Turn          |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |---------------|-----|-----|-----|-----|-----|-----|-----|
    | SoftMajority  |  C  |  C  |  C  |  D  |  C  |  D  |  C  |
    | Opponent      |  C  |  D  |  D  |  C  |  D  |  C  |  C  |
    """
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
    """Hard Majority strategy
    -------------------------
    Will count the number of cooperation and defection of the opponent.
    If there are strictly more cooperations, will cooperate.
    Else, will defect.

    Notes
    -----
    Will always defect on the first turn.
    Is the "defect" counterpart of the `Soft Majority` strategy.

    Interactions
    ------------
    | Turn          |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |---------------|-----|-----|-----|-----|-----|-----|-----|
    | HardMajority  |  D  |  C  |  D  |  C  |  C  |  C  |  D  |
    | Opponent      |  C  |  D  |  C  |  C  |  D  |  D  |  D  |
    """
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
    """Gradual strategy
    -------------------
    Will begin by cooperating.
    If the opponent defects, will punish with defections equal to the number of times the opponent defected so far, minus one.
    After punishing, will calm down by cooperating twice before resuming normal play.

    Parameters
    ----------
    hard : bool, optional
        If True, the strategy tracks and counts the opponent's defection during punishment and calming phases. Defaults to True.

    Notes
    -----
    A softer version of this strategy (`hard=False`) reduces tracking and acts less severely during punishment and calming phases.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |  10 |  11 |  12 |  13 |  14 |
    |-----------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
    | Gradual   |  C  |  C  |  C  |  D  |  C  |  C  |  D  |  D  |  C  |  C  |  C  |  C  |  D  |  D  |  D  |
    | Opponent  |  C  |  C  |  D  |  C  |  C  |  D  |  C  |  D  |  C  |  C  |  C  |  D  |  D  |  C  |  D  |
    """
    def __init__(self, hard=True):
        super().__init__()
        self.name = "gradual"+("" if hard else "_soft")
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
    """Mem strategy
    ---------------
    A strategy that remembers the previous moves of both players and adjusts its actions based on a given genome.

    Parameters
    ----------
    x : int
        The number of moves to remember for the player (my moves).
    y : int
        The number of moves to remember for the opponent (their moves).
    genome : str
        A string representing the strategy's genotype, which contains all possible actions based on the history of moves.
    name : str, optional
        A custom name for the strategy. If not provided, the name is generated from `x`, `y`, and the genome.

    Notes
    -----
    This strategy is based on a genotype that determines the action based on the history of moves of both players.
    The genome is structured such that each combination of moves leads to a different action ('C' or 'D').

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |-----------|-----|-----|-----|-----|-----|-----|-----|
    | Mem(0,0,"C")  |  C  |  C  |  C  |  C  |  C  |  C  |  C  |
    | Opponent  |  C  |  D  |  C  |  D  |  C  |  D  |  C  |

    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |-----------|-----|-----|-----|-----|-----|-----|-----|
    | Mem(0,1,"dCD") |  D  |  C  |  C  |  C  |  D  |  C  |  D  |
    | Opponent       |  C  |  C  |  C  |  D  |  C  |  D  |  C  |

    Notes on behavior:
    - `Mem(0,0, 'C', 'AllC')` will cooperate every time.
    - `Mem(0,1, 'cCD', 'Tft')` behaves like Tit-For-Tat, cooperating initially and then matching the opponent's previous move.
    """
    def __init__(self, x, y, genome, name=None):
        assert max(x,y)+(2**(x+y)) == len(genome), "incorrect genotype size"
        self.name = name
        self.x = x
        self.y = y
        self.genome = genome
        if name is None:  # Nom par défaut si l'utilisateur ne le définit pas
            self.name = "Mem"+str(x)+str(y)+"_"+genome[:max(x,y)].lower()+genome[max(x,y):].upper()
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
    """Prober strategy
    ------------------
    Will start by defecting on the first turn and cooperating on the next two turns.
    If the opponent cooperates on both the second and third turns, it assumes the opponent is exploitable and defects thereafter.
    Otherwise, it mirrors the opponent's last action.

    Notes
    -----
    This strategy aims to test whether the opponent is cooperative and, if so, exploit their behavior.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |-----------|-----|-----|-----|-----|-----|-----|-----|
    | Prober    |  D  |  C  |  C  |  D  |  D  |  D  |  D  |
    | Opponent  |  C  |  C  |  C  |  C  |  C  |  C  |  C  |
    """
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
    """Pavlov strategy
    ------------------
    Also known as "Win-Stay, Lose-Shift."
    Will start by cooperating on the first turn.
    If the previous round's outcome was mutual cooperation or mutual defection, it will cooperate.
    Otherwise, it will defect.

    Notes
    -----
    Pavlov is designed to exploit mutual cooperation while adapting to an opponent's defections.

    Interactions
    ------------
    | Turn      |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |-----------|-----|-----|-----|-----|-----|-----|-----|
    | Pavlov    |  C  |  C  |  D  |  D  |  C  |  C  |  D  |
    | Opponent  |  C  |  D  |  C  |  D  |  C  |  D  |  C  |
    """
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

        
class SpitefulCC(Strategy):
    """SpitefulCC strategy
    ----------------------
    Will cooperate on the first two turns.
    After that, behaves like the Spiteful strategy:
    If either the opponent or itself defected on the last turn, it will defect.
    Otherwise, it will cooperate.

    Notes
    -----
    Equivalent to the memory-based strategy Mem(1,2,'ccCDDDDDDD').
    It starts with cooperation but becomes unforgiving after any defection.

    Interactions
    ------------
    | Turn        |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |-------------|-----|-----|-----|-----|-----|-----|-----|
    | SpitefulCC  |  C  |  C  |  C  |  D  |  D  |  D  |  D  |
    | Opponent    |  D  |  C  |  D  |  C  |  C  |  D  |  D  |
    """
    def __init__(self):
        super().__init__()
        self.name = "spitefulCC"
        self.hisPast = ""
        self.myPast = ""

    def getAction(self, tick):
        if tick < 2:
            return "C"
        if self.hisPast[-1] == "D" or self.myPast[-1] == "D":
            return "D"
        else:
            return "C"

    def clone(self):
        return SpitefulCC()

    def update(self, my, his):
        self.myPast += my
        self.hisPast += his


class TftSpiteful(Strategy):
    """Tit-For-Tat Spiteful strategy
    --------------------------------
    Will cooperate on the first two turns.
    Then behaves like Tit-For-Tat unless the opponent defects on two consecutive turns.
    If two consecutive defections are detected, it switches to permanent defection (becomes spiteful).

    Notes
    -----
    Combines elements of Tit-For-Tat and Spiteful strategies, being forgiving at first but unforgiving after repeated betrayals.

    Interactions
    ------------
    | Turn          |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
    |---------------|-----|-----|-----|-----|-----|-----|-----|
    | TftSpiteful   |  C  |  C  |  D  |  C  |  D  |  D  |  D  |
    | Opponent      |  C  |  D  |  C  |  D  |  D  |  C  |  C  |
    """
    def __init__(self):
        super().__init__()
        self.name = "TftSpiteful"
        self.hisPast = ""
        self.trahison=False
        
    def getAction(self, tick):
        if tick < 2:
            return "C"
        if self.hisPast[-1] == "D" and self.hisPast[-2] == "D":
            self.trahison =True
        if (self.trahison):
            return "D"
        else:
            return self.hisPast[-1]
        
    def clone(self):
        return TftSpiteful()

    def update(self, my, his):
        self.hisPast += his
        

#random.seed(0)
#np.random.seed(0)
class Lunatic(Strategy):
    """Lunatic strategy
    ------------------
    Will choose to cooperate or defect at random, with a specified probability for cooperation.

    Parameters
    ----------
    proba : float, optional
        Probability of cooperating. Defaults to 0.5 (equal chance of cooperation and defection).

    Notes
    -----
    This strategy introduces randomness into the decision-making process, making it unpredictable.
    """
    def __init__(self, proba=0.5):
        super().__init__()
        self.name = "lunatic"
        self.proba = proba

    def getAction(self, tick):
        return np.random.choice(["C","D"],p=[self.proba, 1-self.proba])

    def clone(self):
        return Lunatic(self.proba)

    def update(self, my, his):
        pass


def getMem(x, y):
    if x + y > 4:
        return "Pas calculable"
    len_genome = max(x, y) + 2 ** (x + y)
    permut = [p for p in itertools.product(["C", "D"], repeat=len_genome)]
    genomes = ["".join(p) for p in permut]
    return [Mem(x, y, gen) for gen in genomes]


# same than getMem(1,1) but with synonyms
def getMem11():
    return [Mem(1,1,"cCCCC","mem11_cCCCC-allc"),
            Mem(1,1,"cCCCD","mem11_cCCCD-allc"),
            Mem(1,1,"cCCDC","mem11_cCCDC-allc"),
            Mem(1,1,"cCCDD","mem11_cCCDD-allc"),
            Mem(1,1,"cCDCC","mem11_cCDCC"),
            Mem(1,1,"cCDCD","mem11_cCDCD-tft"),
            Mem(1,1,"cCDDC","mem11_cCDDC-pavlov"),
            Mem(1,1,"cCDDD","mem11_cCDDD-spite"),
            Mem(1,1,"cDCCC","mem11_cDCCC"),
            Mem(1,1,"cDCCD","mem11_cDCCD"),
            Mem(1,1,"cDCDC","mem11_cDCDC"),
            Mem(1,1,"cDCDD","mem11_cDCDD"),
            Mem(1,1,"cDDCC","mem11_cDDCC-perCD"),
            Mem(1,1,"cDDCD","mem11_cDDCD"),
            Mem(1,1,"cDDDC","mem11_cDDDC"),
            Mem(1,1,"cDDDD","mem11_cDDDD"),
            Mem(1,1,"dCCCC","mem11_dCCCC"),
            Mem(1,1,"dCCCD","mem11_dCCCD"),
            Mem(1,1,"dCCDC","mem11_dCCDC"),
            Mem(1,1,"dCCDD","mem11_dCCDD-alld"),
            Mem(1,1,"dCDCC","mem11_dCDCC"),
            Mem(1,1,"dCDCD","mem11_dCDCD-mistrust"),
            Mem(1,1,"dCDDC","mem11_dCDDC"),
            Mem(1,1,"dCDDD","mem11_dCDDD-alld"),
            Mem(1,1,"dDCCC","mem11_dDCCC"),
            Mem(1,1,"dDCCD","mem11_dDCCD"),
            Mem(1,1,"dDCDC","mem11_dDCDC"),
            Mem(1,1,"dDCDD","mem11_dDCDD-alld"),
            Mem(1,1,"dDDCC","mem11_dDDCC-perDC"),
            Mem(1,1,"dDDCD","mem11_dDDCD"),
            Mem(1,1,"dDDDC","mem11_dDDDC"),
            Mem(1,1,"dDDDD","mem11_dDDDD-alld")
            ]

# 26 strategies if we remove identicals
def getSimplifiedMem11():
    return [Mem(1,1,"cCCCC","mem11_cCC**-allc"),
            Mem(1,1,"cCDCC","mem11_cCDCC"),
            Mem(1,1,"cCDCD","mem11_cCDCD-tft"),
            Mem(1,1,"cCDDC","mem11_cCDDC-pavlov"),
            Mem(1,1,"cCDDD","mem11_cCDDD-spite"),
            Mem(1,1,"cDCCC","mem11_cDCCC"),
            Mem(1,1,"cDCCD","mem11_cDCCD"),
            Mem(1,1,"cDCDC","mem11_cDCDC"),
            Mem(1,1,"cDCDD","mem11_cDCDD"),
            Mem(1,1,"cDDCC","mem11_cDDCC-perCD"),
            Mem(1,1,"cDDCD","mem11_cDDCD"),
            Mem(1,1,"cDDDC","mem11_cDDDC"),
            Mem(1,1,"cDDDD","mem11_cDDDD"),
            Mem(1,1,"dCCCC","mem11_dCCCC"),
            Mem(1,1,"dCCCD","mem11_dCCCD"),
            Mem(1,1,"dCCDC","mem11_dCCDC"),
            Mem(1,1,"dCDCC","mem11_dCDCC"),
            Mem(1,1,"dCDCD","mem11_dCDCD-mistrust"),
            Mem(1,1,"dCDDC","mem11_dCDDC"),
            Mem(1,1,"dDCCC","mem11_dDCCC"),
            Mem(1,1,"dDCCD","mem11_dDCCD"),
            Mem(1,1,"dDCDC","mem11_dDCDC"),
            Mem(1,1,"dDDCC","mem11_dDDCC-perDC"),
            Mem(1,1,"dDDCD","mem11_dDDCD"),
            Mem(1,1,"dDDDC","mem11_dDDDC"),
            Mem(1,1,"dDDDD","mem11_d**DD-alld")
            ]

def getPeriodics(n):
    assert n>=1, f"The length of the sequence should be >=1 (length given: {n})"

    cards = ["C", "D"]
    periodics = list()
    for i in range(n + 1):
        periodics += [p for p in itertools.product(cards, repeat=i)]
    strats = [Periodic("".join(p)) for p in periodics]
    return strats


def getClassicals():
    return [
        Periodic("C","allC"),
        Periodic("D","allD"),
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
        HardTft(),
        SlowTft(),
        Gradual(),
        Prober(),
        SpitefulCC(),
        TftSpiteful(),
        Mem(1,2,"ccCDCDDCDD","Winner12"),
        Mem(2,1,"dcCDCDCDDD","Winner21"),
        Gradual(False),
        MetaStrategy([Tft(),Periodic("C","allC"),Spiteful(),Periodic("CCD","perCCD")],4,"badbet")
    ]


class MetaStrategy(Strategy):
    def __init__(self, bag, n, name=None):
        super().__init__()
        if name is None:
            name = ''.join([s.name[:1] for s in bag])
        else:
            self.name = name
        self.bag = bag
        self.n = n
        self.scores = [0 for i in range(len(bag))]
        self.cpt = -1
        self.nbPlayed = [0 for i in range(len(bag))]

    def getAction(self, tick):
        # si c'est l'amorce , on les joue toutes tout à tour
        if tick < self.n * len(self.bag):
            if tick % self.n == 0:
                self.cpt = (self.cpt + 1)
                #print(self.scores, '->', self.cpt, '  ',end='')
        # ensuite on joue celle qui a eu le meilleur score toutes n étapes
        else:
            if tick % self.n == 0:
                self.cpt = np.argmax(self.scores)
                #print(self.scores,'->',self.cpt,'  ',  end='')
                #print(self.cpt,'  ',  end='')
                self.scores[self.cpt]=0
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
        elif his == "C" and my == "D":
            self.scores[self.cpt] = self.scores[self.cpt] + 5
        self.bag[self.cpt].update(my, his)
        self.nbPlayed[self.cpt] += 1



def getAllMemory(x, y):
    if x + y > 4:
        return "Not possible to calculate"
    len_genome = max(x, y) + 2 ** (x + y)
    permut = [p for p in itertools.product(["C", "D"], repeat=len_genome)]
    genomes = ["".join(p) for p in permut]
    return [Mem(x, y, gen) for gen in genomes]


class Proba(Strategy):
    """Proba strategy
    -----------------
    A probabilistic strategy that adjusts its actions based on the previous actions of both players.

    Parameters
    ----------
    first : str
        The action to play on the first turn, either 'C' (cooperate) or 'D' (defect).
    p1, p2, p3, p4 : float
        Probabilities for choosing 'C' over 'D' depending on the combination of previous actions.
    name : str, optional
        A custom name for the strategy. If not provided, the name is automatically generated.

    Notes
    -----
    This strategy uses the previous actions of both players to determine the probability of cooperation for the current turn.
    It has different probabilities based on four possible situations:
    - (C, C), (C, D), (D, C), (D, D).
    """
    def __init__(self, first,p1,p2,p3, p4, name=None):
        super().__init__()
        self.first=first
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4=p4
        self.name = "proba{}_{:1.1f}_{:1.1f}_{:1.1f}_{:1.1f}".format(first,p1,p2,p3,p4) if (name is None) else name

    def getAction(self, tick):
        if (tick == 0):
            return self.first

        rnd = np.random.uniform(0,1)

        if (self.myPrevious == 'C' and self.itsPrevious == 'C'):
            return 'C' if (rnd < self.p1) else 'D'

        if (self.myPrevious == 'C' and self. itsPrevious == 'D'):
            return 'C' if (rnd < self.p2) else 'D'

        if (self.myPrevious == 'D' and self.itsPrevious == 'C'):
            return 'C' if (rnd < self.p3) else 'D'

        # if (self.myPrevious == 'D' and self.itsPrevious == 'D'):
        return 'C' if (rnd < self.p4) else 'D'

        assert 1==2 , "Should never be here !"
        
    def clone(self):
        object = Proba(self.first,self.p1,self.p2,self.p3,self.p4, self.name)
        return object

    def update(self, my, his):
        self.myPrevious = my
        self.itsPrevious = his



# if K=5 we generate 0/5 1/5 2/5 3/5 4/5 5/5  , thus 2*6^4 = 2592
# if K=4 we generate 0/4 1/4 2/4 3/4 4/4      , thus 2*5^4 = 1250
# if K=2 we generate 0/2 1/2 2/2              , thus 2*3^4 =  162
def getAllProba(K, possibleFirst=['C','D']):
    strats = []
    for first in possibleFirst:
        for p1 in range(K+1):
            for p2 in range(K+1):
                for p3 in range(K+1):
                    for p4 in range(K+1):
                        strats.append(Proba(first,p1/K,p2/K,p3/K,p4/K))
    return strats

