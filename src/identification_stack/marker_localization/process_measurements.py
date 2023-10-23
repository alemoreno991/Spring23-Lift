if __name__ == "__main__":
        import utils.vectorTools as vt 
        import utils.transformationTools as trnt
        import utils.cameraConst as cmrc
else:
        from .utils import vectorTools as vt 
        from .utils import transformationTools as trnt
        from .utils import cameraConst as cmrc

import numpy as np

# cocvf := camera opencv frame

class HPS_Estimators:
        # HP-Stage Errors from Altimeter
        def determineHPSErrors_Altimeter( self, bodyposek, cntrptk_cocvf, z_meas ):
                vk_hat_C = vt.getCameraRayUV_C( cntrptk_cocvf )
                vk_hat_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) \
                        * cmrc.CAM_RBC * vk_hat_C 
                lk_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) * cmrc.CAM_POS_B
                deltak = ( (z_meas + lk_I[2] ) /vk_hat_I[2] ) * vk_hat_I
                return deltak[0], deltak[1]

        # LP-Stage Errors from Range
        def determineHPSErrors_Range( self, bodyposek, cntrptk_cocvf, range_meas ):
                vk_hat_C = vt.getCameraRayUV_C( cntrptk_cocvf )
                vk_hat_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) \
                        * cmrc.CAM_RBC * vk_hat_C 
                deltak =  range_meas * vk_hat_I
                return deltak[0], deltak[1]
        

class LPS_Estimators:

        eps = 1E-3
        running_wmeas_sum = np.zeros( (3,1) )
        running_w_sum = 0
        updated_once = False

        scaleafter = 1000 # after 100000 measurements, rescale the sum
        scale = 1
        count = 0

        def weight_function( self, altitude ):
                return 1/( altitude + self.eps )

        def check_for_rescale( self ):
                if self.count >= self.scaleafter:
                        self.count = 0 
                        self.scale = self.scale * self.running_w_sum
                        self.running_wmeas_sum = self.running_wmeas_sum / self.running_w_sum
                        self.running_w_sum = self.running_w_sum / self.running_w_sum
                return        

        
        def update_WLSEstimate_Altimeter( self, bodyposek, cntrptk_cocvf, z_meas ):
                vk_hat_C = vt.getCameraRayUV_C( cntrptk_cocvf )
                vk_hat_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) \
                        * cmrc.CAM_RBC * vk_hat_C
                lk_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) * cmrc.CAM_POS_B
                range_meas = ( (z_meas + lk_I[2] ) /vk_hat_I[2] )
                vk_I = range_meas * vk_hat_I
                rBI_I = trnt.llaWGS84_to_lenu( bodyposek.position )
                zk_I = rBI_I + lk_I + vk_I
                alphak = self.weight_function( range_meas ) / self.scale

                self.running_wmeas_sum = self.running_wmeas_sum + alphak * zk_I
                self.running_w_sum = self.running_wmeas_sum + alphak
                
                if not self.updated_once:
                        self.updated_once = True

                self.count = self.count + 1
                self.check_for_rescle()
                return
        
        def update_WLSEstimate_Range( self, bodyposek, cntrptk_cocvf, range_meas ):
                vk_hat_C = vt.getCameraRayUV_C( cntrptk_cocvf )
                vk_hat_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) \
                        * cmrc.CAM_RBC * vk_hat_C
                vk_I = range_meas * vk_hat_I
                lk_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) * cmrc.CAM_POS_B
                rBI_I = trnt.llaWGS84_to_lenu( bodyposek.position )
                zk_I = rBI_I + lk_I + vk_I
                alphak = self.weight_function( range_meas ) / self.scale

                self.running_wmeas_sum = self.running_wmeas_sum + alphak * zk_I
                self.running_w_sum = self.running_wmeas_sum + alphak

                if not self.updated_once:
                        self.updated_once = True

                self.count = self.count + 1
                self.check_for_rescle()
                return
                        
        def getcurrent_WLSEstimate( self ):
                if self.updated_once:
                        return self.running_wmeas_sum/self.running_w_sum
                else:
                        return None