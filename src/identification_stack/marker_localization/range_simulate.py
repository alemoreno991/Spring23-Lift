if __name__ == "__main__":
        import utils.truthTools as trtt
        import utils.transformationTools as trnt
        import utils.cameraConst as cmrc
else:
        from .utils import truthTools as trtt
        from .utils import transformationTools as trnt
        from .utils import cameraConst as cmrc

import numpy as np

class range_meas_simulator:
        crate_origin_I = trtt.getCrateOrigin_LENU()

        def simulate( self, bodyposek ):
            lk_I = np.transpose( trnt.rpyDCM( bodyposek.attitude.roll, bodyposek.attitude.pitch, bodyposek.attitude.yaw  ) ) * cmrc.CAM_POS_B
            rCI_I = bodyposek.position + lk_I
            return np.linalg.norm( rCI_I - self.crate_origin_I )