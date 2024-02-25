import torch
import torch.nn.functional as F
from rocsolidxgemm import rocb_create_extension,rocb_mm
from hipbsolidxgemm import hipb_create_extension,hipb_mm
import os
import yaml
import pandas as pd

class TunedGemm:
    def __init__(self):
        #rocb_create_extension()
        #hipb_create_extension()
        self.extensions_created = False
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
    def load_best_sols(self):
        tune_file = os.environ.get('VLLM_PERF_CSV')
        if tune_file is not None:
            self.bestsols = pd.read_csv(tune_file,index_col=[0])
            print(self.bestsols)
    def create_ds(self):
        df = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            soltype = 0
            key = (ds['M'],ds['N'],ds['K'])
            if ds['libtype']=='hipblaslt': soltype = 1
            elif ds['libtype']=='rocblas': soltype = 2
            solds[key] = (soltype,int(ds['solidx']))
        self.solids = solds
        #print('>>>',solds)
    def query_sol(self,m,n,k):
        return self.solids.get((m,n,k),(0,0))
    
    def mm(self,inp,weights):
        b=1
        dims = inp.dim()
        if dims==3:
            b = inp.shape[0]
            inp = inp.view(b*inp.shape[1], inp.shape[2])
        if self.extensions_created == False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        # print(inp.shape, weights.shape)
        # print(weights.shape[0],inp.shape[0],inp.shape[1])
        soltype,solidx = self.query_sol(m=weights.shape[0],n=inp.shape[0],k=inp.shape[1])
        # print(soltype, solidx)
        if soltype==1:
            out = hipb_mm(inp,weights.t(),solidx)
        elif soltype==2:
            out = rocb_mm(inp,weights.t(),solidx)
        else:
            #print('>>>Tgemm Default',inp.shape,weights.shape,soltype,solidx)
            out = F.linear(inp,weights)
        if dims==3:
            out = out.view(b, out.shape[0]//b, out.shape[1])
        return out

tgemm = TunedGemm()
