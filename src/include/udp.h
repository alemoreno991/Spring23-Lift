#include <iostream>
#include <queue>
#include <array>
#include <string>
//#include <find>
#include <boost/asio.hpp>
#include <boost/array.hpp>

namespace Lift
{
    struct simData
    {
        float time;
        float radAlt;
        float yaw;
        float pitch;
        float roll;
        float lon;
        float lat;
        float ecef;     // height (Earth centered, Earth fixed)
        float vx;       // velocity x
        float vy;
        float vz;
        float p;        // Angular velocity pitch (rad/s)
        float q;        // Angular velocity roll 
        float r;        // Angular velocity yaw
        float Xacc;     // Foward
        float Yacc;     // Right
        float Zacc;     // Back 
        float ias;      // Indicated air speed
        float agl;      // AGL altitude
        float msl;
    };

    class simCase
    {
        public:
            simCase() {}

            std::queue<simData> pktQueue;
            int a;
            int b;
            int c;
            int d;
            int m3 = 0x3;
            int m13 = 0xD;
            int m16 = 0x10;
            int m17 = 0x11;
            int m20 = 0x14;
            int m21 = 0x15;


            simData simRead() 
            {  
                
                boost::asio::io_context ioContext;
                // boost::asio::ip::address_v4 targetIP;
                // targetIP = boost::asio::ip::address_v4::from_string("192.168.0.8"); 

                // Create a UDP socket
                boost::asio::ip::udp::socket socket(ioContext);
                socket.open(boost::asio::ip::udp::v4());
                // Bind the socket to a specific port
                boost::asio::ip::udp::endpoint localEndpoint(boost::asio::ip::address_v4::any(), 49005);  // Replace with the desired port number
                socket.bind(localEndpoint);
                ioContext.run();
                
                // Receive data
                boost::asio::ip::udp::endpoint remoteEndpoint;

                // BUFFER
                const size_t BUF_SIZE_BYTES = 493;
                //std::unique_ptr<uint8_t[]> buf(new uint8_t[BUF_SIZE_BYTES]);
                
                // boost::asio::mutable_buffers_1 recv_buf = boost::asio::buffer(static_cast<void*>(buf.get()),
                //                                                  BUF_SIZE_BYTES);

                std::array<uint8_t,BUF_SIZE_BYTES> recv_buf;

                // READ DATA INTO BUFFER
                //size_t bytesRead = boost::asio::read(socket,recv_buf);
                size_t bytesRead = socket.receive_from(boost::asio::buffer(recv_buf, BUF_SIZE_BYTES), remoteEndpoint);


                // DO STUFF
                // std::cout << "Buffer length = " << std::dec << recv_buf.size() << std::endl;
                // std::array<uint8_t,493> G;
                //std::copy(std::begin(recv_buf),std::end(recv_buf),std::begin(G));
                // int S = G.size();

                // std::cout << recv_buf.data() << std::endl;

                // for(int i = 0;i<18;i++)
                // {
                //     //char please = reverseEndianness(recv_buf[i]); //c4 00 c0 79
                //     //int l = toBytes(recv_buf[i]);
                //     //std::cout << l << std::endl;
                //     std::cout << std::dec << i << ": "<< std::hex << static_cast<int>(recv_buf[i]) << std::endl;  //79 c0 00 c4            
                // }

                uint8_t nineByte[4];
                float nines;
                nineByte[0] = recv_buf[13];
                nineByte[1] = recv_buf[14];
                nineByte[2] = recv_buf[15];
                nineByte[3] = recv_buf[16];
                memcpy(&nines,nineByte,4);

                //std::cout << "is it dead? " << nines << std::endl;

                return simParse(recv_buf);

                // uint8_t* recv_buf_ptr = (uint8_t*)&(recv_buf[myIt]);

                // float FloatValue;
                // memcpy(&FloatValue,(uint8_t*)recv_buf_ptr,4);

                
                
            }


            

            simData simParse(std::array<uint8_t,493>& recv_buf){

                simData data;
                int myIt = 13;
                int a;
                int b;
                int c;
                int d;
                uint8_t m3 = 0x3;
                uint8_t m13 = 0xD;
                uint8_t m16 = 0x10;
                uint8_t m17 = 17;
                uint8_t m20 = 0x14;
                uint8_t m21 = 0x15;

                std::vector<int> it3;
                std::vector<int> it13;
                std::vector<int> it16;
                std::vector<int> it17;
                std::vector<int> it20;
                std::vector<int> it21;


                for(int i=myIt; i<480; i++){
                    //std::cout << std::dec << i << ": "<< std::hex << static_cast<int>(H[i]) << std::endl;std::cout << "it3 entry: " << i << std::endl;
                    // if(recv_buf[i]==m3){
                    //     it3.insert(it3.end(),i);
                    //     std::cout << "IAS entry: " << i << std::endl;
                    // }
                    // if(recv_buf[i]==m13){
                    //     it13.insert(it13.end(),i);
                    //     std::cout << "Rad and time entry: " << i << std::endl;
                    // }
                    if(recv_buf[i]==m16){
                        it16.push_back(i);
                        //std::cout << "Rates entry: " << i << std::endl;
                    }
                    if(recv_buf[i]==m17){
                        it17.insert(it17.end(),i);
                        //std::cout << "Attitude entry: " << i << std::endl;
                    }
                    if(recv_buf[i]==m20){
                        it20.insert(it20.end(),i);
                        //std::cout << "Position entry: " << i << std::endl;
                    }
                    if(recv_buf[i]==m21){
                        it21.insert(it21.end(),i);
                        //std::cout << "Velocity entry: " << i << std::endl;
                    }
                }

            //     // uint8_t* recv_buf_ptr = (uint8_t*)&(recv_buf[myIt]);

            //     // float FloatValue;
            //     // memcpy(&FloatValue,(uint8_t*)recv_buf_ptr,4);

            // FOR MEASSAGE 16 (p: data1[], q: data2[], r: data3[])

                float pFloat;
                float qFloat;
                float rFloat;
                uint8_t pByte[4];
                uint8_t qByte[4];
                uint8_t rByte[4];
                for(int i=0; i<it16.size();i++){    
                    //std::cout << "it16 size: " << it16[i] << std::endl;
                }

                //std::cout << "Printing finished" << std::endl;

                
                for(int i=0; i<it16.size(); i++){
                    //std::cout << "In loop" << std::endl;
                    int iter16 = it16[i];
                    // std::cout << "Iter16 = " << iter16 << std::endl; 
                    if(recv_buf[iter16+1]==0 & recv_buf[iter16+2]==0 & recv_buf[iter16+3]==0){
                        // std::cout << "In header check loop" << std::endl;
                        pByte[0] = recv_buf[iter16+4];
                        //std::cout << "pByte[0] = " << pByte[0] << std::endl; 
                        //printf("pByte[0] = %02x \n",pByte[0]);
                        pByte[1] = recv_buf[iter16+5];
                        pByte[2] = recv_buf[iter16+6];
                        pByte[3] = recv_buf[iter16+7];
                        // for(int i =0; i<4;i++){
                        //     std::cout << "pByte " << i << " : " << std::hex << (uint8_t)pByte[i] << std::endl;
                        // }
                        memcpy(&pFloat,pByte,4);
                        qByte[0] = recv_buf[iter16+8];
                        qByte[1] = recv_buf[iter16+9];
                        qByte[2] = recv_buf[iter16+10];
                        qByte[3] = recv_buf[iter16+11];
                        memcpy(&qFloat,qByte,4);
                        rByte[0] = recv_buf[iter16+12];
                        rByte[1] = recv_buf[iter16+13];
                        rByte[2] = recv_buf[iter16+14];
                        rByte[3] = recv_buf[iter16+15];
                        memcpy(&rFloat,rByte,4);
                    }
                }
                
                data.p = pFloat;
                data.q = qFloat;
                data.r = rFloat;
                //std::cout << "p = " << data.p << std::endl;
                //std::cout << "q = " << data.q << std::endl;
                //std::cout << "r = " << data.r << std::endl;


                // FOR MEASSAGE 17 (pitch: data1[], roll: data2[], yaw: data3[])

                float pitchFloat;
                float rollFloat;
                float yawFloat;
                uint8_t pitchByte[4];
                uint8_t rollByte[4];
                uint8_t yawByte[4];
                // std::cout << "Iter17 size = " << it17.size() << std::endl;
                for(int i=0; i<it17.size(); i++){
                    //std::cout << "In loop" << std::endl;
                    int iter17 = it17[i];
                    // std::cout << "Iter17 = " << iter17 << std::endl; 
                    if(recv_buf[iter17+1]==0 & recv_buf[iter17+2]==0 & recv_buf[iter17+3]==0){
                        //std::cout << "In header check loop" << std::endl;
                        pitchByte[0] = recv_buf[iter17+4];
                        //std::cout << "pByte[0] = " << pByte[0] << std::endl; 
                        //printf("pitchByte[0] = %02x \n",pByte[0]);
                        pitchByte[1] = recv_buf[iter17+5];
                        pitchByte[2] = recv_buf[iter17+6];
                        pitchByte[3] = recv_buf[iter17+7];
                        // for(int i =0; i<4;i++){
                        //     std::cout << "pByte " << i << " : " << std::hex << (uint8_t)pByte[i] << std::endl;
                        // }
                        memcpy(&pitchFloat,pitchByte,4);
                        rollByte[0] = recv_buf[iter17+8];
                        rollByte[1] = recv_buf[iter17+9];
                        rollByte[2] = recv_buf[iter17+10];
                        rollByte[3] = recv_buf[iter17+11];
                        memcpy(&rollFloat,rollByte,4);
                        yawByte[0] = recv_buf[iter17+12];
                        yawByte[1] = recv_buf[iter17+13];
                        yawByte[2] = recv_buf[iter17+14];
                        yawByte[3] = recv_buf[iter17+15];
                        memcpy(&yawFloat,yawByte,4);
                    }
                }
                //}
                data.pitch = pitchFloat;
                data.roll = rollFloat;
                data.yaw = yawFloat;
                //std::cout << "pitch = " << data.pitch << std::endl;
                //std::cout << "roll = " << data.roll << std::endl;
                //std::cout << "yaw = " << data.yaw << std::endl;


                // FOR MEASSAGE 20 (lat: data1[], lon: data2[], msl(ft): data3[], agl(ft) data4[])

                float latFloat;
                float lonFloat;
                float mslFloat;
                float aglFloat;
                uint8_t latByte[4];
                uint8_t lonByte[4];
                uint8_t mslByte[4];
                uint8_t aglByte[4];
                // std::cout << "Iter20 size = " << it20.size() << std::endl;
                for(int i=0; i<it20.size(); i++){
                    //std::cout << "In loop" << std::endl;
                    int iter20 = it20[i];
                    // std::cout << "Iter20 = " << iter20 << std::endl; 
                    if(recv_buf[iter20+1]==0 & recv_buf[iter20+2]==0 & recv_buf[iter20+3]==0){
                        //std::cout << "In header check loop" << std::endl;
                        latByte[0] = recv_buf[iter20+4];
                        //std::cout << "pByte[0] = " << pByte[0] << std::endl; 
                        //printf("latByte[0] = %02x \n",latByte[0]);
                        latByte[1] = recv_buf[iter20+5];
                        latByte[2] = recv_buf[iter20+6];
                        latByte[3] = recv_buf[iter20+7];
                        memcpy(&latFloat,latByte,4);
                        lonByte[0] = recv_buf[iter20+8];
                        lonByte[1] = recv_buf[iter20+9];
                        lonByte[2] = recv_buf[iter20+10];
                        lonByte[3] = recv_buf[iter20+11];
                        memcpy(&lonFloat,lonByte,4);
                        mslByte[0] = recv_buf[iter20+12];
                        mslByte[1] = recv_buf[iter20+13];
                        mslByte[2] = recv_buf[iter20+14];
                        mslByte[3] = recv_buf[iter20+15];
                        memcpy(&mslFloat,mslByte,4);
                        aglByte[0] = recv_buf[iter20+16];
                        aglByte[1] = recv_buf[iter20+17];
                        aglByte[2] = recv_buf[iter20+18];
                        aglByte[3] = recv_buf[iter20+19];
                        memcpy(&aglFloat,aglByte,4);
                    }
                }
                //}
                data.lat = latFloat;
                data.lon = lonFloat;
                data.msl = mslFloat;
                data.agl = aglFloat;
                //std::cout << "lat = " << data.lat << std::endl;
                //std::cout << "lon = " << data.lon << std::endl;
                //std::cout << "msl(ft) = " << data.msl << std::endl;
                //std::cout << "agl(ft) = " << data.agl << std::endl;


                // FOR MEASSAGE 21 (p: data4[], q: data5[], r: data6[])

                float vxFloat;
                float vyFloat;
                float vzFloat;
                uint8_t vxByte[4];
                uint8_t vyByte[4];
                uint8_t vzByte[4];
                for(int i=0; i<it21.size();i++){    
                    // std::cout << "it21 size: " << it21.size() << std::endl;
                }

                                
                for(int i=0; i<it21.size(); i++){
                    //std::cout << "In loop" << std::endl;
                    int iter21 = it21[i];
                    // std::cout << "Iter21 = " << iter21 << std::endl; 
                    if(recv_buf[iter21+1]==0 & recv_buf[iter21+2]==0 & recv_buf[iter21+3]==0){
                        //std::cout << "In header check loop" << std::endl;
                        vyByte[0] = recv_buf[iter21+16];
                        //std::cout << "pByte[0] = " << pByte[0] << std::endl; 
                        //printf("vxByte[0] = %02x \n",vxByte[0]);
                        vyByte[1] = recv_buf[iter21+17];
                        vyByte[2] = recv_buf[iter21+18];
                        vyByte[3] = recv_buf[iter21+19];
                        memcpy(&vyFloat,vyByte,4);
                        vzByte[0] = recv_buf[iter21+20];
                        vzByte[1] = recv_buf[iter21+21];
                        vzByte[2] = recv_buf[iter21+22];
                        vzByte[3] = recv_buf[iter21+23];
                        memcpy(&vzFloat,vzByte,4);
                        vxByte[0] = recv_buf[iter21+24];
                        vxByte[1] = recv_buf[iter21+25];
                        vxByte[2] = recv_buf[iter21+26];
                        vxByte[3] = recv_buf[iter21+27];
                        memcpy(&vxFloat,vxByte,4);
                    }
                }
                
                data.vx = -vxFloat;
                data.vy = vyFloat;
                data.vz = vzFloat;
                //std::cout << "vx = " << data.vx << std::endl;
                //std::cout << "vy = " << data.vy << std::endl;
                //std::cout << "vz = " << data.vz << std::endl;
                
                return data;

            //     //std::cout << std::dec << "Value = " << FloatValue << std::endl;

            }
    };
}