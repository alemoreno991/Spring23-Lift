if __name__ == "__main__":
        import utils.vectorTools as vt 
        import utils.truthTools as trtt
        import utils.transformationTools as trnt
        import utils.cameraConst as cmrc
else:
        from .utils import vectorTools as vt 
        from .utils import truthTools as trtt
        from .utils import transformationTools as trnt
        from .utils import cameraConst as cmc

import numpy as np

# cocvf := camera opencv frame
def determineLPSErrors( bodyposek, cntrptk_cocvf, z_meas ):
        vk_hat_C = vt.getCameraRayUV_C( cntrptk_cocvf )
        vk_hat_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) \
                * cmc.CAM_RBC * vk_hat_C 
        deltak = ( z_meas/vk_hat_I[2] ) * vk_hat_I