# IPD : the Iterated Prisoner's Dilemma

This repository is a continuation of the old website [http://www.lifl.fr/IPD](https://web.archive.org/web/1998*/www.lifl.fr/ipd) created in 1998 and containing a simulator to carry out various experiments at the iterated prisoner's dilemma (thanks to the Wayback Machine). This repository has now been deleted to reappear here.

It contains Python code and Jupyter notebooks to understand easily how to experiment computational game theory, build and evaluate strategies at the iterated prisoner's dilemma (IPD) and gives many tools for sophisticated experiments. It is also useful to reproduce most of the experiences of our research articles cited in the bibliography.

Team : P Mathieu, JP Delahaye, B Beaufils, L Fodil, C Petitpre  ([CRISTAL Lab](http://www.cristal.univ-lille.fr), [SMAC team](https://www.cristal.univ-lille.fr/?rubrique27&eid=17), [Lille University](http://www.univ-lille.fr))

Contact : philippe.mathieu at univ-lille.fr

In this repository there are several jupyter notebooks in Python which contain both explanations and exercises on this field. Their objective is primarily pedagogical.

## Quick start

```python
from ipd import *
from strategies import *

bag=[Periodic("D"), HardMajority(), Tft(), Spiteful(),  Gradual()]
e= Ecological(g,bag, 1000, pop=[100])
e.run()
e.tournament.matrix
e.historic
e.drawPlot()
```

## Gallery
A [Gallery](Gallery#readme) of many remarkable curves is available. A [refCard.pdf](ipd_refCard.pdf) is also available for a really short introduction.

# Notebooks

## Game Theory
Français : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=FR/gametheory_fr.ipynb)
English : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=EN/gametheory_en.ipynb)

In this Jupyter notebook we explain the basics of game theory concerning simultaneous games, including payoff matrix notion and the different equilibrium notions. In this notebook the prisoner's dilemma is just one example among many.

## The basics of the Iterated Prisoner's dilemma
Français : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=FR/ipd_basics_fr.ipynb)
English : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=EN/ipd_basics_en.ipynb)

We focus now on the famous Iterated Prisoner's Dilemma Game which is the iterated version of the previous one. Strategies can now learn from the past. We show how to build a strategy and especially how to evaluate it. Several sets of objective strategies are provided. The two classic methods of evaluating a set of strategies are provided: the tournament and the ecological competition.

## How to simplify complete large classes?
Français : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=FR/ipd_tools_simplify_fr.ipynb)
English : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=EN/ipd_tools_simplify_en.ipynb)


Creating a large set of strategies is difficult. The computation of complete classes of strategies, as seen in the basics, thanks to a definition based on a genotype, easily allows to generate thousands of them. Nevertheless some of them are redundant. This sheet provides a reflection on how to test the equivalence of two strategies and how to simplify large sets. 

## A more robust evaluation tool: subclasses
Français : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=FR/ipd_tools_subclasses_fr.ipynb)

The tournament is a first way to evaluate a set of strategies. The ecological competition is more robust. But testing a strategy in various sets is even more robust. We propose here some tools to calculate the average performance of a strategy in all possible subclasses of a list of reference strategies.

## An approach using genetic algorithms
Français : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=FR/ipd_gas_fr.ipynb)

In this Jupyter Notebook we show how to use "genetic" technics to identify and produce new efficient strategies, mainly genetic algorithms. It is then very easy, thanks to this approach, to find the best possible strategy for a fixed set of strategies.


## The construction of meta-strategies
Français : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cristal-smac/ipd.git/master?filepath=FR/ipd_tools_metastrat_fr.ipynb)

Meta-strategies are strategies that build on a set of other strategies. They play a given strategy for a certain period of time and then, according to certain criteria, other strategies for another period of time. This makes it possible to construct scomplex and yet understandable behaviors, while maintaining a very high level of adaptability of the strategy to its opponent.


# Bibliography
- Philippe Mathieu, Jean-Paul Delahaye. **Experimental criteria to identify efficient probabilistic memory-one strategies for the iterated prisoner’s dilemma**. Simulation Modelling Practice and Theory, Elsevier, 2019
- Philippe Mathieu, Jean-Paul Delahaye. **New Winning Strategies for the Iterated Prisoner's Dilemma**. Journal of Artificial Societies and Social Simulation, SimSoc Consortium, 2017, 20 (4)
- Philippe Mathieu, Jean-Paul Delahaye. **New Winning Strategies for the Iterated Prisoner's Dilemma**. 14th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2015), May 2015, Istanbul, Turkey. pp.1665-1666
- Philippe Mathieu, Bruno Beaufils, Jean-Paul Delahaye. **Studies on Dynamics in the Classical Iterated Prisoner's Dilemma with Few Strategies: Is There Any Chaos in the Pure Dilemma ?**. Proceedings of the 4th european conference on Artificial Evolution (AE'99), 1999, Dunkerque, France, France. pp.177--190
- Bruno Beaufils, Jean-Paul Delahaye, Philippe Mathieu. **Complete Classes of Strategies for the Classical Iterated Prisoner's Dilemma**. Evolutionnary Programming VII (EP'7), 1998, undef, France. pp.33--41
- Bruno Beaufils, Jean-Paul Delahaye, Philippe Mathieu. **Our Meeting with Gradual, A Good Strategy for the Iterated Prisoner's Dilemma**. Proceedings of the Fifth International Workshop on the Synthesis and Simulation of Living Systems (ALIFE'5), 1996, Cambridge, MA, USA, France. pp.202--209

# License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see http://www.gnu.org/licenses/.
