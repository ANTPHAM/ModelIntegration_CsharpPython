
DEBUG=2 # 1 - mixed-mode cross-language debugger
        # 2 - attach Python debugger
        # 3 - ptvsd remote debugging

import clr

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
        return "testing"