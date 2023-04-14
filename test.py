class SS:
    demo = {'v' : 2}

    def check(self):
        self.demo = {}
        self.demo['a'] = 1
        
ss = SS()
ss.check()
print(ss.demo)