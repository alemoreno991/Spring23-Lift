import numpy as np
from . import truthConst as tc
from . import transformationTools as trnsfm

def getCrateOrigin_LENU():
    return trnsfm.llaWGS84_to_lenu( np.array( [ [tc.CRATE_ORIGIN_TRUE_LAT],[tc.CRATE_ORIGIN_TRUE_LON],[tc.CRATE_ORIGIN_TRUE_ALT] ] ) )
