import itertools
import random
import numpy as np

class Strategy():
    def setMemory(self,mem):
        pass
    
    def getAction(self,tick):
        pass
    
    def __copy__(self):
        pass

    def update(self,x,y):
        pass
    

class Periodic(Strategy):
    def __init__(self, sequence, name=None):
        super().__init__()
        self.sequence = sequence.upper()
        self.step = 0
        self.name = "per_"+sequence if (name == None) else name

    def getAction(self,tick):
        return self.sequence[tick % len(self.sequence)]

    def clone(self):
        object = Periodic(self.sequence, self.name)
        return object

    def update(self,x,y):
        pass

class Tft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "tft"
        self.hisPast=""
        
    def getAction(self,tick):
        return 'C' if (tick==0) else self.hisPast[-1]

    def clone(self):
        return Tft()

    def update(self,my,his):
        self.hisPast+=his

class Tf2t(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "tf2t"
        self.hisPast=""
        
    def getAction(self,tick):
        if (tick==0 or tick==1):
            return 'C'
        else :
            if (self.hisPast[-1] == 'D' and self.hisPast[-2]):
                return 'D'
            else :
                return 'C'

    def clone(self):
        return Tf2t()

    def update(self,my,his):
        self.hisPast+=his

class Hardtft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "hardtft"
        self.hisPast=""
        
    def getAction(self,tick):
        if (tick==0 or tick==1):
            return 'C'
        else :
            if (self.hisPast[-1] == 'D' or self.hisPast[-2] == 'D'):
                return 'D'
            else :
                return 'C'

    def clone(self):
        return Hardtft()

    def update(self,my,his):
        self.hisPast+=his

class Slowtft(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "slowtft"
        self.hisPast=""
        self.myPast=""
        
    def getAction(self,tick):
        if (tick==0 or tick==1):
            return 'C'
        else :
            if (self.hisPast[-1] == 'D' and self.hisPast[-2] == 'D' ):
                return 'D'
            elif (self.hisPast[-1] == 'C' and self.hisPast[-2] == 'C' ):
                return 'C'
            else :
                return self.myPast[-1]
            

    def clone(self):
        return Slowtft()

    def update(self,my,his):
        self.hisPast+=his
        self.myPast+=my
    
    
class Spiteful(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "spiteful"
        self.hisPast=""
        self.myPast=""
        
    def getAction(self,tick):
        if (tick==0):
                return 'C'
        if (self.hisPast[-1]=='D' or self.myPast[-1]=='D') :
            return 'D'
        else :
            return 'C'

    def clone(self):
        return Spiteful()

    def update(self,my,his):
        self.myPast+=my
        self.hisPast+=his

class Mistrust(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "mistrust"
        self.hisPast=""
        self.myPast=""
        
    def getAction(self,tick):
        if (tick==0):
                return 'D'
        if (self.hisPast[-1]=='D' or self.myPast[-1]=='D') :
            return 'D'
        else :
            return 'C'

    def clone(self):
        return Mistrust()

    def update(self,my,his):
        self.myPast+=my
        self.hisPast+=his

class SoftMajority(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "softmajo"
        self.nbCooperations = 0
        self.nbTrahisons = 0
        
    def getAction(self,tick):
        if (self.nbCooperations >= self.nbTrahisons):
            return 'C'
        else :
            return 'D'

    def clone(self):
        return SoftMajority()

    def update(self,my,his):
        if (his == 'C'):
            self.nbCooperations += 1
        elif (his == 'D'):
            self.nbTrahisons += 1

class HardMajority(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "hardmajo"
        self.nbCooperations = 0
        self.nbTrahisons = 0
        
    def getAction(self,tick):
        if (self.nbCooperations > self.nbTrahisons):
            return 'C'
        else :
            return 'D'

    def clone(self):
        return HardMajority()

    def update(self,my,his):
        if (his == 'C'):
            self.nbCooperations += 1
        elif (his == 'D'):
            self.nbTrahisons += 1
            

class Gradual(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "gradual"
        self.nbTrahisons = 0
        self.punish = 0
        self.calm = 0
    def getAction(self,tick):
        if (tick==0) : return 'C'
        if self.punish > 0 :
            self.punish-=1
            return 'D'
        if self.calm > 0 :
            self.calm-=1
            return 'C'
        if self.hisLast=='D' : 
            self.punish=self.nbTrahisons - 1
            self.calm=2
            return 'D'
        else: return 'C'

    def clone(self):
        return Gradual()

    def update(self,my,his):
        self.hisLast=his
        if (his == 'D'):
            self.nbTrahisons += 1

class Mem(Strategy):
    def __init__(self, x, y, genome, name=None):
        self.name = name
        self.x = x
        self.y = y
        self.genome = genome
        if (name == None): #Nom par défaut si l'utilisateur ne le définit pas
            self.name = genome
        self.myMoves = [] #contains my x last moves
        self.itsMoves = [] #contains its y last moves

    def clone(self):
        return Mem(self.x, self.y, self.genome, self.name)

    def getAction(self, tick):
        if (tick < max(self.x, self.y)):
            return self.genome[tick]
        cpt = 0
        for i in range(self.x-1,-1,-1):
            cpt*=2
            if (self.myMoves[i] == 'D'):
                cpt+=1
        for i in range(self.y-1,-1,-1):
            cpt*=2
            if (self.itsMoves[i] == 'D'):
                cpt+=1
        cpt += max(self.x, self.y)
        return self.genome[cpt]

    def update(self, myMove, itsMove):
        if (self.x > 0):
            if(len(self.myMoves) == self.x):
                del self.myMoves[0]
            self.myMoves.append(myMove)
        if (self.y > 0):
            if(len(self.itsMoves) == self.y):
                del self.itsMoves[0]
            self.itsMoves.append(itsMove)

class Prober(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "prober"
        self.hisPast=""
        
    def getAction(self,tick):
        if (tick==0):
            return 'D'
        elif (tick == 1 or tick == 2):
            return 'C'
        else :
            if (self.hisPast[1] == 'C' and self.hisPast[2] == 'C'):
                return 'D'
            else :
                return self.hisPast[-1]
                
    def clone(self):
        return Prober()

    def update(self,my,his):
        self.hisPast+=his

class Pavlov(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "pavlov"
        self.hisPast=""
        self.myPast=""
        
    def getAction(self,tick):
        if (tick==0):
            return 'C'
        else :
            if (self.hisPast[-1] == self.myPast[-1]):
                return 'C'
            else :
                return 'D'          
                
    def clone(self):
        return Pavlov()

    def update(self,my,his):
        self.hisPast+=his
        self.myPast+=my

class MetaStrategy(Strategy):
    def __init__(self, bag, n):
        super().__init__()
        self.name = "metastrat"
        self.bag = bag
        self.n = n
        self.scores = [0 for i in range(len(bag))]
        self.cpt = -1
        
    def getAction(self,tick):
        print(tick)
        
        if (tick < self.n * len(self.bag)):
            if (tick % self.n ==  0):
                self.cpt = (self.cpt + 1) % len(self.bag) 
            return self.bag[self.cpt].getAction(tick % self.n)
        else :
            if (tick % self.n ==  0):
                self.cpt = np.argmax(self.scores)
            return self.bag[self.cpt].getAction(tick % self.n)
                
    def clone(self):
        return MetaStrategy(self.bag, self.n)

    def update(self,my,his):
        if (his == 'C' and my == 'C'):
            self.scores[self.cpt] = self.scores[self.cpt] + 3
        elif (his == 'D' and my == 'D'):
            self.scores[self.cpt] = self.scores[self.cpt] + 1
        elif (his == 'D' and my == 'C'):
            self.scores[self.cpt] = self.scores[self.cpt] + 5
        self.bag[self.cpt].update(my,his)



def getMem(x,y):
    if (x+y > 4):
        return "Pas calculable"
    len_genome = max(x,y)+2**(x+y)
    permut = [p for p in itertools.product(['C','D'], repeat=len_genome)]
    genomes = [''.join(p) for p in permut]
    return [Mem(x,y,gen) for gen in genomes]

def getPeriodics(n):
    cards = ['C','D']
    periodics = list()
    for i in range (n+1):
        periodics += [p for p in itertools.product(cards, repeat=i)]
    strats = [Periodic(''.join(p)) for p in periodics]
    return strats[1:]

def getClassicals():
    return [Periodic('C'), Periodic('D'), Tft(), Spiteful(), SoftMajority(), HardMajority(), Periodic("DDC"), Periodic("CCD"), Mistrust(), Periodic("CD"), Pavlov(), Tf2t(), Hardtft(), Slowtft(), Gradual(), Prober()]

