"""
Author: Rishikesh Dewalkar
Date: 30 September 2024
"""
from logic_gate import LogicGate
gate = LogicGate()

list = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x in list:
    y= gate.and_gate(x[0], x[1])
    print('{} AND {} = {}'.format(x[0], x[1], y))

for x in list:
    y = gate.nand_gate(x[0], x[1])
    print('{} NAND {} = {}'.format(x[0], x[1], y))

for x in list:
    y = gate.nor_gate(x[0], x[1])
    print('{} NOR {} = {}'. format(x[0], x[1], y))

