import pyrealsense2 as rs
import numpy as np

################################################################################
# This class handles all the camera related stuff
################################################################################
class RealsenseCamera:
    #---------------------------------------------------------------------------
    # Constructor: 
    #   - Loads and configures the camera
    #   - Stores the camera intrinsic parameters
    #---------------------------------------------------------------------------
    def __init__(self):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming 
        profile = self.pipeline.start(config)

        # Align depth information to BGR frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Get camera intrinsics
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    #---------------------------------------------------------------------------
    # get_frame_stream: 
    #   - Gets the color (BGR) and depth frames
    #   - Applies a filter to the depth frame to fill holes
    #---------------------------------------------------------------------------
    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Check camera status
        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print("Error, impossible to get the frame, make sure that the \
                    Intel Realsense camera is correctly connected")
            return False, None, None
        
        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)
        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        return True, color_frame, depth_frame


    def get_optical_center(self):
        return self.intr.ppx, self.intr.ppy

    def get_focal_center(self):
        return self.intr.fx, self.intr.fy

    def release(self):
        self.pipeline.stop()
