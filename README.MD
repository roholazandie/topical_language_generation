## Topical Langauge generation (TLG)
![alt text](img/gui1.png)

### Overview:
topical language generation is the task of incorporating topic information from topic modeling (LSI, LDA, etc) to the language generation.
In this repository we have TLG with transformers. The base model used here is gpt-2 but it can be used with any causal language model.


## Installation
- Python 3.6+ 
- Install the requirements
```
pip install -r requirements
```

- You need to install graphviz to see the full GUI

```
 sudo apt-get install graphviz libgraphviz-dev pkg-config
```

## Samples:

for LDA with different prompts (bold face texts are prompts):

---
**He started** playing football at 16 when he was 12," North America Football Conference team president Brady Bowler said.
, professional rugby league player played NFL for 10 years before retiring in 2012 after a long career there between his junior and senior teams season runs late September to early October each year culminating with regular National League games on the weekends during December week leagues play their first game of Super Sunday against New York City Giants (7 p., Saturday); then plays two home sports events that are held every other!
---

---
**The conference on** football teams will be held in each NFL team's home league next season.
, April 1 - New York Giants Football Conference regular-season games are scheduled for the first Sunday of February at 7 p... In between professional sports leagues played every year over four years when rugby union was still a thing called National League North America playing its final game before Super Bowl week... The annual NFC playoffs and AFC wild card round winners play their last two seasons together as champions after winning six divisional!

---

