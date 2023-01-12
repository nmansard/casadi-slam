import apriltag
import cv2
import  numpy as np

# AprilTag definitions
tag_family  = 'tag36h11'
tag_size    = 0.168
tag_corners = tag_size / 2 * np.array([[-1,1],[1,1],[1,-1],[-1,-1]])
tag_corners_3d = tag_size / 2 * np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])

