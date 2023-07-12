import deploymentConst as dplyC
import serial.tools.list_ports
import time

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
    srl_obj.bytesize = dplyC.SERIAL_BYTESIZE
    # srl_obj.parity
    # srl_obj.stopbits
    srl_obj.timeout = dplyC.SERIAL_TIMEOUT

    
    srl_obj.open()
    if srl_obj.is_open:
        print("[DEBUG]: SERIAL CONNECTION OPENNED")
    else:
        print("[DEBUG]: SERIAL CONNECTION DID NOT OPEN .. EXITING")
        exit(0)

    for ii in range(0,6):
        packet_t = srl_obj.readline()
        # TODO: PARSE THE PACKET WRT INCOMING DATA
        time.sleep(5)

    srl_obj.close()
    print("[RESULT]: TEST CONCLUDED ... EXITING")
    

if __name__ == '__main__':
    main()