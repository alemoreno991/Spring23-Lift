#! /usr/bin/env python3

import zmq
import struct
import time

TEST_INIT = False
UI16_PACKET_SIZE = 0
UI32_PACKET_SIZE = 0 
F32_PACKET_SIZE = 20

topic = "test_topic".encode('ascii')
print("Reading messages with topic: {}".format(topic))

with zmq.Context() as context:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic)

    # GET ONLY MOST RECENT MESSAGE
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect("tcp://127.0.0.1:5555")

    i = 0

    try:
        while True:
            binary_topic, data_buffer = socket.recv().split(b' ', 1)

            topic = binary_topic.decode(encoding = 'ascii')

            print("Message {:d}:".format(i))
            print("\ttopic: '{}'".format(topic))

            struct_format = ''

            if TEST_INIT:
                packet_size = len(data_buffer) // struct.calcsize("h")
                print("\tpacket size: {:d}".format(packet_size))
                struct_format = "{:d}h".format(packet_size)
            else:
                struct_format = "20f"

            data = struct.unpack(struct_format, data_buffer)
            print("\tdata: {}".format(data))
            time.sleep(2)
            i += 1

    except KeyboardInterrupt:
        socket.close()
    except Exception as error:
        print("ERROR: {}".format(error))
        socket.close()