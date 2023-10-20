import argparse

# IDENIFICATION CONSTS
import idenificationConst as idc

# IDENTIFICATION STACK 
import identificationStack as idstack


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='runIdentification.py',
        description='''UT-Lift Script for Aerial Vehicles Identifying and Locating Markered Ground Crates'''
    )

    pipleline                   = idc.PIPELINE 
    cap_opt                     = idc.CAP_OPT 
    dnn_weights                 = idc.DNN_WEIGHTS
    ds_sub_address_string       = idc.SUB_ADDR
    logger_file_string          = idc.LOG_FILE

    myoptions                   = idstack.ids_opt(pipleline, cap_opt, dnn_weights, ds_sub_address_string, logger_file_string)
    id_stack                    = idstack.identificationStack()
    
    id_stack.runContinuousIdentifiactionStream(100)

