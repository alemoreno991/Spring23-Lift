import argparse

# IDENTIFICATION STACK 
import identification_stack.identificationStack as idstack

if __name__ == '__main__':

    stack_opts = 2 # INTEL REALSENSE
    id_stack   = idstack.identificationStack( stack_opts )
    id_stack.runContinuousIdentifiactionTestStream( 10 )