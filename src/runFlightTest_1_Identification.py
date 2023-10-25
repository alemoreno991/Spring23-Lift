# IDENTIFICATION STACK 
import identification_stack.identificationStack as idstack

if __name__ == '__main__':

    use_aruco       = True
    collect_iters   = 10000

    stack_opts      = None
    if use_aruco:
        stack_opts = r'FTA' # INTEL REALSENSE
    else:
        stack_opts = r'FT1' # INTEL REALSENSE
        
    id_stack        = idstack.identificationStack( stack_opts )
    id_stack.runContinuousIdentifiactionTestStream( collect_iters )