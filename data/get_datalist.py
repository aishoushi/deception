import os
import random


data = open('pandakill_all.txt').readlines()

random.shuffle(data)

train_list = open('train_pandakill_5.txt', 'w')
test_list = open('test_pandakill_5.txt', 'w')
print(len(data))
for i in range(160):
    train_list.write(data[i])
for i in range(160, 200):
    test_list.write(data[i])
