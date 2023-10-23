import argparse

# DETECTION NNETWORK
import detection_nnetwork.detectc as dtctnn

# CLASSIFICATION FILTER
import classification_filter.decodeImage as cfilter

# ESTIMATION 
import marker_localization.process_measurements as mrklcl

# MANAGEMENT AND LOGGING
import management_and_logging.management_and_logging as mnl

if __name__ == "__main__":
        import utils.identificationOptions as ido
else:
        from .utils import identificationOptions as ido


class identificationStack():
    
    DS_Manager      = None 
    CS_Manager      = None
    logger          = None 
    detectNN        = None

    def __init__( self, options_TYPE ):
        options = ido.ids_opt( options_TYPE )
        self.DS_Manager      = mnl.dataStreamManager( options.pipeline, options.cap_opt ) 
        self.CS_Manager      = mnl.captureStreamManager( options.ds_sub_address_string  ) 
        self.logger          = mnl.dataLogger( options.logger_file_string  )
        print("[INFO] PREPARING DETECTION NETWORK MODEL ...")
        self.detectNN        = dtctnn.detectionNetwork( options.dnn_weights )
        print("[INFO] DETECTION NETWORK MODEL PREPARED ...")


    def runIdentification( self, imgi ):
         # DETECT MARKERS IN IMAGE
        bbcrnrs_list = self.detectNN.imageDetection(imgi)

        # CLASSIFY MARKERS IN IMAGE
        for bbcrnr in bbcrnrs_list:
            
            read_code = []
            crate_center_pxlpt = []
            try:
                read_code, sctn_cntr_pxlpnt = cfilter.decodeImageSection(imgi,bbcrnr,0,True)
                x1, y1 = int(bbcrnr[0]), int(bbcrnr[1])
                
                if len(read_code) > 0:
                    if len( sctn_cntr_pxlpnt ) > 0:
                        crate_center_pxlpt = np.transpose( np.array( [ x1, y1 ] ) + sctn_cntr_pxlpnt )
                    return read_code, crate_center_pxlpt
                else:
                    continue

            except:
                # TODO: FIND POSSIBILITY FOR INDEXING ERROR IN CLASSIFICATION FILTER
                print("[ERROR]: AN UNEXPECTED ERROR OCCURED DURING IMAGE DECODING. SKIPPING DETECTED REGION ...")
                continue
        
        return [],[]
    

    def runIdentificationStream_wData( self ):
        try:
            # GRAB IMAGE AND TELEMETRY DATA
            data = self.DS_Manager.get_dataPackage()
            imgi = self.CS_Manager.get_Image()
            read_code, _ = self.runIdentification( imgi )

            if len(read_code) > 0:
                # SAVE TELEMETRYi and IMAGEi FOR ESTIMATION TESTS
                self.logger.save_data(data,imgi)

        except:
            return


    def runContinuousIdentifiactionStream( self, stop_indx = float('inf') ):            
        print("[INFO]: STARTING PRIMARY LOOP")
        ii = 0
        while ii < stop_indx:
            try:
                self.runIdentificationStream_wData()
                ii += 1
            except:
                continue
