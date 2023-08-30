import numpy as np
import math
from . import calibrationConst as clbc

def zDCM(rotz):
    return np.array(    [\
                            [ math.cos(rotz) , -math.sin(rotz) , 0 ],\
                            [ math.sin(rotz) ,  math.cos(rotz) , 0 ],\
                            [       0        ,         0       , 1 ]\
                        ] )

def yDCM(roty):
    return np.array(    [\
                            [ math.cos(roty)  ,   0   ,  math.sin(roty)  ],\
                            [      0          ,   1   ,        0         ],\
                            [ -math.sin(roty) ,   0   ,  math.cos(roty)  ]\
                        ] )


def xDCM(rotx):
    return np.array(    [\
                            [ 1 ,        0        ,        0        ],\
                            [ 0 , math.cos(rotx)  , -math.sin(rotx) ],\
                            [ 0 , math.sin(rotx)  ,  math.cos(rotx) ]\
                        ] )


def rpyDCM( roll, pitch, yaw ):
    return zDCM(yaw) * yDCM(pitch) * xDCM(roll)


def llaWGS84_to_ecef( lla ):
    # TRANSFOR LLAWGS84 TO ECEF
    lat = lla[0]
    lon = lla[1]
    lat_rad = math.radians( lat ) # TO RADIANS
    lon_rad = math.radians( lon ) # TO RADIANS
    alt = lla[2]

    pvroc = clbc.WGS84_SEMI_MAJOR_A/math.sqrt( 1 - math.pow( clbc.WGS84_ECCNTRCTY*math.sin( lat_rad ), 2 ) )

    x_ecef = ( pvroc + alt ) * math.cos( lat_rad ) * math.cos( lon_rad )
    y_ecef = ( pvroc + alt ) * math.cos( lat_rad ) * math.sin( lon_rad )
    z_ecef = ( ( 1 - math.pow( clbc.WGS84_ECCNTRCTY, 2 ) )*pvroc + alt ) * math.sin( lat_rad )

    return np.array( [ [x_ecef], [y_ecef], [z_ecef] ] )


ECEFREF_WGS84_LAT_RAD = math.radians( clbc.ECEFREF_WGS84_LAT ) # RADIANS
ECEFREF_WGS84_LON_RAD = math.radians( clbc.ECEFREF_WGS84_LON ) # RADIANS
ECEF2LOCALENU_DCM = np.array(  [  [             -math.sin(ECEFREF_WGS84_LON_RAD)               ,                   math.cos(ECEFREF_WGS84_LON_RAD)          ,            0               ], \
                                  [ -math.sin(ECEFREF_WGS84_LAT_RAD)*math.cos(ECEFREF_WGS84_LON_RAD), -math.sin(ECEFREF_WGS84_LAT_RAD)*math.sin(ECEFREF_WGS84_LON_RAD), math.cos(ECEFREF_WGS84_LAT_RAD) ], \
                                  [  math.cos(ECEFREF_WGS84_LAT_RAD)*math.cos(ECEFREF_WGS84_LON_RAD),  math.cos(ECEFREF_WGS84_LAT_RAD)*math.sin(ECEFREF_WGS84_LON_RAD), math.sin(ECEFREF_WGS84_LAT_RAD) ]  \
                                ] )
ECEFREF_ECEF = llaWGS84_to_ecef( np.array( [ [clbc.ECEFREF_WGS84_LAT],[clbc.ECEFREF_WGS84_LON],[clbc.ECEFREF_WGS84_ALT] ] ) )

def ecef_to_lenu( pt_ecef ):
    # DETERMINE POINT IN LOCAL FIXED ENU FRAME
    xyz_localenu = ECEF2LOCALENU_DCM * ( pt_ecef - ECEFREF_ECEF )
    return xyz_localenu


def llaWGS84_to_lenu( lla ):
    return ecef_to_lenu( llaWGS84_to_ecef( lla ) )