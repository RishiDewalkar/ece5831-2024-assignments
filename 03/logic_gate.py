import numpy as np

class LogicGate:
    def __init__(self):
        pass

    def and_gate(self, x1, x2):
        w = np.array([x1, x2])
        x = np.array([0.5, 0.5])
        b = -0.7

        y = np.sum(x*w)+b
        if y>0:
            return 1
        else:
            return 0
        
    def nand_gate(self, x1, x2):
        if self.and_gate(x1, x2)==0:
            return 1
        else:
            return 0
        
    def or_gate(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2

        y = np.sum(x*w) + b

        if y>0:
            return 1
        else:
            return 0
        
    def nor_gate(self, x1, x2):
        if(self.or_gate(x1, x2))==0:
            return 1
        else:
            return 0
        
    def xor_gate(self, x1, x2):
        y1 = self.or_gate(x1,x2)
        y2 = self.nand_gate(x1, x2)
        y = self.and_gate(y1, y2)

        return y

#%% 
    
if __name__ == "__main__":
    print('LogicGate: ')
    print("and_gate(x1, x2) results AND logic gate output.")
    print("For instance, and_gate(1, 0)-->")
    gate = LogicGate()
    y = gate.and_gate(1, 0)
    print(y)

    print("or_gate(x1, x2) results OR logic gate output")
    print("For instance, or_gate(1, 0)-->")
    gate = LogicGate()
    y = gate.or_gate(1, 0)
    print(y)

    print("nor_gate(x1, x2) results NOR logic gate output")
    print("For instance, nor_gate(1, 0) -->")
    gate = LogicGate()
    y = gate.nor_gate(1, 0)
    print(y)

    print("nand_gate(x1, x2) give NAND logic gate output")
    print("for instance, nand_gate(1, 0)-->")
    gate = LogicGate()
    y = gate.nand_gate(1, 0)
    print(y)

    print("xor_gate(x1, x2) gives XOR logic gate output")
    print("For instance, xor_gate(1, 0)-->")
    gate = LogicGate()
    y = gate.xor_gate(1, 0)
    print(y)