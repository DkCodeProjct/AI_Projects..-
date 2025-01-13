import hashlib 

# DKCodeCoing -> DC
class DkCodeBlock:
    def __init__(self, prevBlockHash, transX_list):
        self.prevBlockHash = prevBlockHash
        self.transX_list = transX_list
        
        #construct a data string
        # - cat transaction list and prev blockhash
        self.blockData = "-".join(transX_list) + "-" + prevBlockHash

        # hasing data
        self.blockHash = hashlib.sha256(self.blockData.encode()).hexdigest()

t1 = "mike sends 2 DC to bob "
t2 = "jon sends 1 DC to ksi "
t3 = "alex sends 5 DC to jj "
t4 = "dick sends 3 DC to ball "
t5 = "mike sends 1.02 DC to lex "
t6 = "mike sends 3 DC to alex "

initBlock = DkCodeBlock("init string", [t1, t2])
print(initBlock.blockData)
print(initBlock.blockHash)

initBlock2 = DkCodeBlock(initBlock.blockHash, [t3, t4])
print(initBlock2.blockData)
print(initBlock2.blockHash)

initBlock3 = DkCodeBlock(initBlock2.blockHash, [t4, t5])
print(initBlock3.blockData)
print(initBlock3.blockHash)

initBlock4 = DkCodeBlock(initBlock3.blockHash, [t5, t6])
print(initBlock4.blockData)
print(initBlock4.blockHash)




