#include <udp.h>
#include <zmq.hpp>

// Veronte includes
// #include <Vpkt.h>
// #include <Vpktparser.h>
// #include <Vpktproducer.h>
// #include <Cfg.h>
// #include <Irxid.h>
// #include <Irx.h>

// Tunable includes
// #include <Velcmd.h>
// #include <Attcmd.h>
// #include <Poscmd.h>

//#include <Serial.h>
#include <stdio.h>
//#include <Commgr.h>
//#include <Telemetryrx.h>
//#include <PosLLH.h>
//#include <Comm_hlp.h>

//std includes
#include <queue>
#include <any>
#include <vector>
#include <iostream>
//#include <adamStore.h>



// #include <iostream>
// #include <zmq.hpp>
#include <string>
// #include <stdio.h>
#include <stdlib.h> 
// SLEEP FUNCTION
#ifndef _WIN32
    #include <unistd.h>
#else
    #include <windows.h>
    #define sleep(n)    Sleep(n)
#endif


Lift::simCase mySimCase;



namespace utlift_zmq_cnnct{
    class publisher_zmq_cpp{
        private:
            // ZMQ CONTEXT AND SOCKET
            zmq::context_t ctx;
            zmq::socket_t skt;
            const char* tcp_address_string;

            // MESSAGING FORMATS
            const char* TOPIC; 
            const size_t topic_size;
            const size_t data_size;
            const size_t envelope_size;

        public:
            publisher_zmq_cpp( const char* addressi, const char* TOPICi, size_t datasi ) : 
                ctx(1),
                skt( ctx, ZMQ_PUB  ) ,
                tcp_address_string(addressi), 
                TOPIC(TOPICi), 
                topic_size(std::strlen(TOPIC)), 
                data_size(data_size), 
                envelope_size( std::strlen(TOPIC) + 1 + datasi)
            {
                printf("Topic: %s; topic size: %zu; Envelope size: %zu\n", this -> TOPIC, this -> topic_size, this -> envelope_size);
                this -> skt.bind( this -> tcp_address_string);
            };

            ~publisher_zmq_cpp(){
                this -> skt.close();
                // this -> ctx.close();
            }


            bool publish_test_messages( int16_t buffer[], const unsigned int PACKET_SIZE ){
                

                // printf("Read %u data values\n", PACKET_SIZE);
                zmq::message_t envelope( this -> envelope_size );

                // TOPIC
                memcpy( envelope.data(), this -> TOPIC, this -> topic_size );
                // PADDING
                memcpy((void*)((char*)envelope.data() + this -> topic_size), " ", 1);
                // DATA 
                memcpy((void*)((char*)envelope.data() + 1 + this -> topic_size), buffer, PACKET_SIZE * sizeof(int16_t));

                bool send_pass = this -> skt.send(envelope);
                if( !send_pass ){
                    printf("ERROR: ZeroMQ error occurred during zmq_msg_send(): %s\n", zmq_strerror(errno));
                }
                return send_pass;
            }


            bool publish_f32_messages( _Float32 buffer[], const unsigned int PACKET_SIZE ){
                

                // printf("Read %u data values\n", PACKET_SIZE);
                zmq::message_t envelope( this -> envelope_size );

                // TOPIC
                memcpy( envelope.data(), this -> TOPIC, this -> topic_size );
                // PADDING
                memcpy((void*)((char*)envelope.data() + this -> topic_size), " ", 1);
                // DATA 
                memcpy((void*)((char*)envelope.data() + 1 + this -> topic_size), buffer, PACKET_SIZE * sizeof(_Float32));

                bool send_pass = this -> skt.send(envelope);
                if( !send_pass ){
                    printf("ERROR: ZeroMQ error occurred during zmq_msg_send(): %s\n", zmq_strerror(errno));
                }
                return send_pass;
            }


            bool publish_sectioned_UTParse_Message( uint16_t ui16_buffer[], const unsigned int UI16_PACKET_SIZE, 
                                                    uint32_t ui32_buffer[], const unsigned int UI32_PACKET_SIZE,
                                                    _Float32 f32_buffer[], const unsigned int F32_PACKET_SIZE ){
                
                printf("Read %u data values\n", UI16_PACKET_SIZE + UI32_PACKET_SIZE + F32_PACKET_SIZE );
                zmq::message_t envelope( this -> envelope_size );

                // TOPIC
                memcpy( envelope.data(), this -> TOPIC, this -> topic_size );
                // PADDING
                memcpy((void*)((char*)envelope.data() + this -> topic_size), " ", 1);
                // DATA 
                memcpy((void*)((char*)envelope.data() + this -> topic_size + 1 ), 
                    ui16_buffer, UI16_PACKET_SIZE * sizeof(uint16_t));
                memcpy((void*)((char*)envelope.data() + this -> topic_size + 1 + (UI16_PACKET_SIZE * sizeof(uint16_t)) ), 
                    ui32_buffer, UI32_PACKET_SIZE * sizeof(uint32_t));
                memcpy((void*)((char*)envelope.data() + this -> topic_size + 1 + (UI16_PACKET_SIZE * sizeof(uint16_t)) + (UI32_PACKET_SIZE * sizeof(uint32_t)) ), 
                    f32_buffer, F32_PACKET_SIZE * sizeof(_Float32));

                bool send_pass = this -> skt.send(envelope);
                if( !send_pass ){
                    printf("ERROR: ZeroMQ error occurred during zmq_msg_send(): %s\n", zmq_strerror(errno));
                }
                return send_pass;
            }

    };
}



int main()
{
    const char* tcp_address = "tcp://*:5555";
    const char* tcp_topic = "veronte_tele";
    const unsigned int F32_PACKET_SIZE = 20;
    size_t data_size = F32_PACKET_SIZE * sizeof(_Float32);
    // const unsigned int REPETITIONS = 10;
    utlift_zmq_cnnct::publisher_zmq_cpp test_zmq_publisher( tcp_address, tcp_topic, data_size );
 
    // char c = 0;
    // while (c != 'X')
    unsigned int i = 0;
    while(true)
    {
        
        // // READ DATA INTO BUFFER
        // //size_t bytesRead = boost::asio::read(socket,recv_buf);
        // size_t bytesRead = socket.receive_from(boost::asio::buffer(recv_buf, BUF_SIZE_BYTES), remoteEndpoint);

        Lift::simData datain = mySimCase.simRead();
        _Float32 data_buffer[F32_PACKET_SIZE];
        data_buffer[0] = datain.time;
        data_buffer[1] = datain.radAlt;
        data_buffer[2] = datain.yaw;
        data_buffer[3] = datain.pitch;
        data_buffer[4] = datain.roll;
        data_buffer[5] = datain.lon;
        data_buffer[6] = datain.lat;
        data_buffer[7] = datain.ecef;
        data_buffer[8] = datain.vx;
        data_buffer[9] = datain.vy;
        data_buffer[10] = datain.vz;
        data_buffer[11] = datain.p;
        data_buffer[12] = datain.q;
        data_buffer[13] = datain.r;
        data_buffer[14] = datain.Xacc;
        data_buffer[15] = datain.Yacc;
        data_buffer[16] = datain.Zacc;
        data_buffer[17] = datain.ias;
        data_buffer[18] = datain.agl;
        data_buffer[19] = datain.msl;


        bool send_pass = test_zmq_publisher.publish_f32_messages( data_buffer, F32_PACKET_SIZE );
        if( !send_pass ){
            printf("ERROR: ZeroMQ error occurred during zmq_msg_send(): %s\n", zmq_strerror(errno));
            break;
        }

        // printf("Message sent; i: %u, topic: %s\n", i, tcp_topic);
        // i += 1;
        // c = getchar();
        // c = toupper(c);
        // putchar(c);
        // putchar('>');
    }

    // socket.shutdown();

    return 0;
}

