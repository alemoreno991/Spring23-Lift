# IDENTIFICATION STACK 
import identification_stack.identificationStack as idstack

if __name__ == '__main__':

    use_aruco       = True

    stack_opts      = None
    if use_aruco:
        stack_opts = r'FTA0' # INTEL REALSENSE
    else:
        stack_opts = r'FT0' # INTEL REALSENSE
        
    id_stack        = idstack.identificationStack( stack_opts )
    id_stack.runContinuousIdentifiactionStream_woData()