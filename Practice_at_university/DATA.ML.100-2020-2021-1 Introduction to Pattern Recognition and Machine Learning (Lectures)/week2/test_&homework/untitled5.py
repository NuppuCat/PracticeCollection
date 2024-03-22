# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:35:44 2020

@author: onepiece
"""


from multiprocessing import Process as process

def mul_process():
    for i in range(0,10):
        return i

if __name__ == "__main__":
    core_1 = process(target = mul_process).start()
   
    core_2 = process(target = mul_process).start()
    core_3 = process(target = mul_process).start()