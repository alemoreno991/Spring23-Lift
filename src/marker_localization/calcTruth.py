from . import trueConst as tc
from . import transformations as trnsfm
import numpy as np

def getCrateOrigin_LENU():
    return trnsfm.llaWGS84_to_lenu( np.array( [ [tc.CRATE_ORIGIN_TRUE_LAT],[tc.CRATE_ORIGIN_TRUE_LON],[tc.CRATE_ORIGIN_TRUE_ALT] ] ) )
