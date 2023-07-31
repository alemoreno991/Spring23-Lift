import deploymentConst as dplyC
import serial.tools.list_ports
import time
import struct

def main():
    ports = serial.tools.list_ports.comports()

    portsList = []
    print("[DEBUG]: DISPLAYING PORT OPTIONS")
    for port in ports:
        portsList.append(str(port))
        print(str(port))

    port_name = dplyC.SERIAL_PORT_NAME
    if port_name is None:
        port_select = input("\nSELECT PORT FROM LIST: COM")
        for ii in range(0,len(portsList)):
            if portsList[ii].startswith( "COM" + str(port_select) ):
                port_name = "COM" + str(port_select)
                break

    if not (port_name is None):
        print("[DEBUG]: SELECTED PORT IS {}".format(port_name))
    else:
        print("[ERROR]: SELECTED PORT WAS NOT FOUND ... EXITING")
        exit(0)

    # FORMAT SERIAL CONNECTION
    srl_obj = serial.Serial()
    srl_obj.port = port_name
    srl_obj.baudrate = dplyC.SERIAL_BAUDRATE
    
    srl_obj.open()
    if srl_obj.is_open:
        print("[DEBUG]: SERIAL CONNECTION OPENNED. SAMPLING IN 2 SECOND INTERVALS")
    else:
        print("[DEBUG]: SERIAL CONNECTION DID NOT OPEN .. EXITING")
        exit(0)

    for ii in range(0,6):
        srl_obj.flushInput()
        packet = srl_obj.readline()
        # packet_b = bytes.fromhex( packet )
        packet_b = bytes(packet)
        stamp_t, hash_t, phase_t, time_t, lon_t, lat_t, wgs_t, heading_t, vel_t, pitch_t, roll_t, yaw_t, p_t, q_t, r_t, xacc_t, yacc_t, zacc_t = struct.unpack( '>fIHfIIIfffffffffff', packet_b )
        print("######################################################################")
        print("[DEBUG]: TIMESTAMP - {}".format(stamp_t))
        print("[DEBUG]: HASH - {}".format(hash_t))
        print("[DEBUG]: PHASE - {}".format(phase_t))
        print("[DEBUG]: TIME - {}".format(time_t))
        print("[DEBUG]: LONGITUDE - {}".format(lon_t))
        print("[DEBUG]: LATITUDE - {}".format(lat_t))
        print("[DEBUG]: WGS - {}".format(wgs_t))
        print("[DEBUG]: HEADING - {}".format(heading_t))
        print("[DEBUG]: VELOCITY - {}".format(vel_t))
        print("[DEBUG]: PITCH - {}".format(pitch_t))
        print("[DEBUG]: ROLL - {}".format(roll_t))
        print("[DEBUG]: YAW - {}".format(yaw_t))
        print("[DEBUG]: P - {}".format(p_t))
        print("[DEBUG]: Q - {}".format(q_t))
        print("[DEBUG]: R - {}".format(r_t))
        print("[DEBUG]: X ACCELERATION - {}".format(xacc_t))
        print("[DEBUG]: Y ACCELERATION - {}".format(yacc_t))
        print("[DEBUG]: Z ACCELERATION - {}".format(zacc_t))
        print("######################################################################")
        time.sleep(2)

    srl_obj.close()
    print("[RESULT]: TEST CONCLUDED ... EXITING")
    

if __name__ == '__main__':
    main()