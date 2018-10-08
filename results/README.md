# Questions in french on word analogies

question format: 
Man is to woman what king is to ? (queen)
king - man + woman = queen

3 rules for acceptance:

## EXCEPTION:
result is valid if the closest word excluding the added word is the target word.
e.g. in "king - man + woman = queen", woman is discarded .

model 2018 (CBOW model / corpus CC + Wiki) outperform the others

## TOP 1
result is valid if the closest word is the target word.

Not so relevent. Mostly fails. But 2018 model outperforms the others.

## TOP5
resulst is valid if the closest word is in the top5 closest words

Model 2018 mostly outperforms the others, but the aggregated model is
sometimes better.

# Global remarks
Generally, if one model completely fails the task, the aggregated model
will also fail.

Skipgram happens to be better for 2 specific tasks
(adj -> adv, p.pres -> p.pass)