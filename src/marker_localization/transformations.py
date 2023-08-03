import numpy as np
import math

WGS84_SEMI_MAJOR_A = 6378137. # METERS
WGS84_SEMI_MINOR_A = 6356752.314 # METERS
WGS84_ECCNTRCTY = math.sqrt( 1 - math.pow( WGS84_SEMI_MINOR_A/WGS84_SEMI_MAJOR_A ,2 ) ) # UNITLESS

ENUREF_WGS84_LAT = 444 # DEGRESS
ENUREF_WGS84_LON = 444 # DEGREES
ENUREF_WGS84_ALT = 444 # METERS

ENUREF_WGS84_LAT_RAD = math.radians( ENUREF_WGS84_LAT ) # RADIANS
ENUREF_WGS84_LON_RAD = math.radians( ENUREF_WGS84_LON ) # RADIANS
ECEF2LOCALENU_DCM = np.array(  [  [             -math.sin(ENUREF_WGS84_LON_RAD)               ,                   math.cos(ENUREF_WGS84_LON_RAD)          ,            0               ], \
                                  [ -math.sin(ENUREF_WGS84_LAT_RAD)*math.cos(ENUREF_WGS84_LON_RAD), -math.sin(ENUREF_WGS84_LAT_RAD)*math.sin(ENUREF_WGS84_LON_RAD), math.cos(ENUREF_WGS84_LAT_RAD) ], \
                                  [  math.cos(ENUREF_WGS84_LAT_RAD)*math.cos(ENUREF_WGS84_LON_RAD),  math.cos(ENUREF_WGS84_LAT_RAD)*math.sin(ENUREF_WGS84_LON_RAD), math.sin(ENUREF_WGS84_LAT_RAD) ]  \
                                ] )


def llaWGS84_to_ecef( lla ):
    # TRANSFOR LLAWGS84 TO ECEF
    lat = lla[0]
    lon = lla[1]
    lat_rad = math.radians( lat ) # TO RADIANS
    lon_rad = math.radians( lon ) # TO RADIANS
    alt = lla[2]

    pvroc = WGS84_SEMI_MAJOR_A/math.sqrt( 1 - math.pow( WGS84_ECCNTRCTY*math.sin( lat_rad ), 2 ) )

    x_ecef = ( pvroc + alt ) * math.cos( lat_rad ) * math.cos( lon_rad )
    y_ecef = ( pvroc + alt ) * math.cos( lat_rad ) * math.sin( lon_rad )
    z_ecef = ( ( 1 - math.pow( WGS84_ECCNTRCTY, 2 ) )*pvroc + alt ) * math.sin( lat_rad )

    return np.array( [ [x_ecef], [y_ecef], [z_ecef] ] )


ENUREF_ECEF = llaWGS84_to_ecef( np.array( [ [ENUREF_WGS84_LAT],[ENUREF_WGS84_LON],[ENUREF_WGS84_ALT] ] ) )
def ecef_to_localenu( pt_ecef ):
    # DETERMINE POINT IN LOCAL FIXED ENU FRAME
    xyz_localenu = ECEF2LOCALENU_DCM * ( pt_ecef - ENUREF_ECEF )
    return xyz_localenu
