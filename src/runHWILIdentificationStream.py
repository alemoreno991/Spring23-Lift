import argparse

# IDENTIFICATION STACK 
import identification_stack.identificationStack as idstack

if __name__ == '__main__':

    stack_opts = 1
    id_stack   = idstack.identificationStack()
    id_stack.runContinuousIdentifiactionStream(100)

