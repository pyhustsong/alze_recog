# alze_recog
This is a repo which contains files used to recognize alzheimer.
## Mentality ##
RecogNet contains two parts:
- encoder
A CNN used to encode imgs.
- decoder
A RNN used to combain infomations of all imgs and classify instant.
## Requirements ##
- Python >= 3.7
- Pytorch >= 1.1.0
## Train ##
```
python3 solver.py
```
## Test ##
```
python3 test.py
```
