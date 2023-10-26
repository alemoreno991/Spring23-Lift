import numpy as np
import cv2 as cv

if __name__ == "__main__":
        import utils.identificationOptions as ido
        import marker_identification.marker_identifier as mrklid
        import marker_localization.process_measurements as mrklcl
        import management_and_logging.management_and_logging as mnl
else:
        from .utils import identificationOptions as ido
        from .marker_identification import marker_identifier as mrklid
        from .marker_localization import process_measurements as mrklcl
        from .management_and_logging import management_and_logging as mnl


class identificationStack():
    
    DS_Manager      = None 
    CS_Manager      = None
    logger          = None 
    identifier      = None

    def __init__( self, options_TYPE ):
        options = ido.ids_opt( options_TYPE )
        self.DS_Manager      = mnl.dataStreamManager( options.ds_sub_address_string  ) 
        self.CS_Manager      = mnl.captureStreamManager( options.cap_opt, options.pipeline ) 
        self.logger          = mnl.dataLogger( options.logger_file_string  )
        self.identifier      = mrklid.marker_identifier(  options.classification_type, options.dnn_weights )
    
    def runSIIdentification( self, imgi ):
        return self.identifier.runIdentification( imgi )

    def runIdentificationStream_wData( self ):
        try:
            # GRAB IMAGE AND TELEMETRY DATA
            data = self.DS_Manager.get_dataPackage()
            imgi = self.CS_Manager.get_Image()
            read_code, _ = self.identifier.runIdentification( imgi )

            if len(read_code) > 0:
                # SAVE TELEMETRYi and IMAGEi FOR ESTIMATION TESTS
                self.logger.save_data(data,imgi)
                return True
            
        except:
            return False
        
        return False
    
    def runIdentificationStream_woData( self ):
        try:
            imgi = self.CS_Manager.get_Image()
            read_code, _ = self.identifier.runIdentification( imgi )

            if len(read_code) > 0:
                # SAVE TELEMETRYi and IMAGEi FOR ESTIMATION TESTS
                self.logger.save_img( imgi )
                return True
            
        except:
            return False
        
        return False


    def runIdentificationTestStream( self ):
        try:
            imgi = self.CS_Manager.get_Image()
            read_code, cntr = self.identifier.runIdentification( imgi )
            if len(read_code) > 0 and len(cntr) > 0:
                imgi = cv.circle( imgi.copy(), (int(cntr[0][0]),int(cntr[0][1])), 20, (int(255), int(0), int(0)), 3)
            # cv.imshow( 'rgb', imgi )

            if len(read_code) > 0:
                self.logger.save_img( imgi )
                return True
            
        except:
            return False
        
        return False


    def runContinuousIdentifiactionStream( self, stop_indx = float('inf') ):            
        print("[INFO]: STARTING PRIMARY LOOP")
        ii = 0
        try:
            while ii < stop_indx:
                try:
                    found = self.runIdentificationStream_wData()
                    if found:
                        ii += 1
                except:
                    continue
        except KeyboardInterrupt:
            print("[INFO] STOP COMMAND RECIEVED ... CLOSING COLLECTION. DATA IS IN ./data")


    def runContinuousIdentifiactionStream_woData( self, stop_indx = float('inf') ):            
        print("[INFO]: STARTING PRIMARY LOOP")
        ii = 0
        try: 
            while ii < stop_indx:
                try:
                    found = self.runIdentificationStream_woData()
                    if found:
                        ii += 1
                except:
                    continue
        except KeyboardInterrupt:
            print("[INFO] STOP COMMAND RECIEVED ... CLOSING COLLECTION. DATA IS IN ./data")


    def runContinuousIdentifiactionTestStream( self, stop_indx = float('inf') ):            
        print("[INFO]: STARTING PRIMARY LOOP")
        ii = 0
        while ii < stop_indx:
            try:
                found = self.runIdentificationTestStream()
                if found:
                    ii += 1
            except:
                continue 
