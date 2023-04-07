import math

def euclideanDistance(a,b):
    return pow( pow(a[0] - b[0],2) + pow(a[1] - b[1],2) , 0.5 )

def angleBetweenPoints(a,b):
    angle = math.atan2( b[1] - a[1], b[0] - a[0] )
    if angle < 0:
        angle = angle + 2 * math.pi 
    return angle

def determineAngleQuadrant( angle ):
    if angle < math.pi/2:
        return 0
    elif angle < math.pi:
        return 1
    elif angle < 3*math.pi/2:
        return 2
    else:
        return 3
