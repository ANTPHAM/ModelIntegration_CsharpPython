# DEBUG mode accepted values
# 0 - No debugger
# 1 - mixed-mode cross-language debugger
# 2 - attach Python debugger
# 3 - ptvsd remote debugging
DEBUG=0

import clr
#import apr2_predictionMax
#testing= 'testing'
import sys

def attachDebug():
    if DEBUG==2:
        import sys
        import time
        while not sys.gettrace():
            time.sleep(0.1)

    elif DEBUG==3:
        import ptvsd
        ptvsd.enable_attach('clr')
        ptvsd.wait_for_attach()

def pyfunc(args=None):
    if args:
        return args
    else:
        try:
            import scipy
            import numpy as np
            ##return ( np.random.randint(3))
            #import pandas as pd
            import sklearn as sk
            #boston = sk.datasets.load_boston()
            #return ('testing %d' %np.random.randint(3), boston.shape)
        except :
            import sys
            print (sys.exc_info())
            attachDebug()
            print ("Error")

               

#pyfunc()