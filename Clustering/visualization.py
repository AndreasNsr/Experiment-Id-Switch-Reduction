# vim: expandtab:ts=4:sw=4

import numpy as np
import pandas as pd
import colorsys
from .image_viewer import ImageViewer
from .live_keypoints_cam import MediaPipePose

import inspect
import cv2
import os
import sys




def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """


    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        
        
        # DF = pd.DataFrame(arr) 
        # save the dataframe as a csv file 
        # DF.to_csv("data1.csv")

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """
    
    landmarks_coordinates = np.empty([1, 5])


    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]


    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        
        
        print ('caller name:', inspect.stack()[1][3])



        # NOT ARRIVING        
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        

        # Get Super function that is calling this function: frame_callback func on deep_sort_app
        #print ('caller name 2:', inspect.stack()[1][3])
        
        #print ('caller name 2:', detections[0])

        
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)
            #self.viewer.is_labeled(*detection.tlwh)
           
    def is_labeled(x, y, w, h, track_id= None):
        """Check if this detection includes a label -> Replaced by calling only confirmed tracks from deep_sort_app

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.
        """
               
        #print ('caller name image view:', inspect.stack()[1][3], "\n")
        #print("XXXXXXX", type(x))
        
        return track_id
            
    # ADDED FUNCTION BY ME THAT CROPS THE IMAGE OF LABELED (ID) BOUNDING BOX:
    def crop_image_by_bounding_box(self, image, frame_idx, x, y, w, h, track_id):
        
        
        #print("INDSIDE crop_image_by_bounding_box: \n", type(x), type(y), type(w), type(h) )
        #image = cv2.imread(path)
        
        print("XYWH", int(y), int(h), int(x), int(w))

        cropped_image = None
        
        if (x < 0 or y < 0 or w < 0 or h < 0): # For avoiding cut bounding box outside the frame
            
            raise ValueError(f'The {frame_idx} image/detected track is None => Check bbox values')

        #cropped_image = image[ y:  y + h,  x :  x +  w ]
        cropped_image = image[int(y): int(y)+int(h), int(x) : int(x) + int(w)]
        
        cv2.imwrite('C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/single_detections/' + 'Fr. id' + str(frame_idx) + ' Label' + str(int(track_id)) + '.jpg', cropped_image)
    
        return cropped_image 

            
    # ADDED FUNCTION BY ME
    def draw_mp_pose_landmarks(self, original_image, bbox, track_id, seq_info, frame_idx):
        
        
        #path = 'C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/MOT16/img1/' + str(frame_idx) + ".jpg"
        #dir_list = os.listdir(path)
        
        """IN CASE OF GETTING ALL DTECTIONS IN FRAME (INCLUDING THE UNCONFIRMED ONES)"""
        """
        for i, detection in enumerate(detections):
            label = self.is_labeled(*detection.tlwh)
            print("LBEL", label)
            if label != "":
        """
    
        #print("NOT none - inside loop...", type(*detection.tlwh))
        #x,y,w,h = *detection.tlwh
        

        try:
            
            image = self.crop_image_by_bounding_box(original_image, frame_idx, bbox[0], bbox[1], bbox[2], bbox[3], track_id)
    
    
        except ValueError as err:
    
            print(err.args) #Frame 304/305 is problematic (bbox outside figure)
            return


        
        #self.crop_image_by_bounding_box(image, seq_info, frame_idx, *detection.tlwh)
        # Get Super function that is calling this function: frame_callback func on deep_sort_app
        #print ('caller name:', inspect.stack()[1][3])

        
        self.viewer.thickness = 1
        self.viewer.color = 0, 255, 0
        
        index = 1


        #print("DETECTIONS: ", detections, "\n")
        
        #for i, detection in enumerate(detections):
        #min_x, min_y, max_x, max_y = detection.to_tlbr() # top left, bottom right
        
        #image = cv2.imread(seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR) #Replaced by running pose est. to bbox and not whole
        
        #path = seq_info["sequence_name"][frame_idx]
        
        
        
        "NOTED -> ATEEMPT TO OVERCOME FRAME IDX OBSTACLE:"
        #frame_id= "%06d" % frame_idx
        #path = 'C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/MOT16/img1/' + frame_id + ".jpg"
        #dir_list = os.listdir(path)
        
        #print("\n\n PATHHH...", path)
        
        path = 'C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/MOT16/img1/' + str(frame_idx) + ".jpg"

        
        media_pipe_model = MediaPipePose(image, path, frame_idx)
        
        
               
        

        list_result = media_pipe_model.run(track_id)            
        
        if list_result is not None:

            #annotated_image, pose_landmarks, POSE_CONNECTIONS, landmark_drawing_spec = media_pipe_model.run()
                            
            #self.viewer.draw_landmarks(list_result[0], list_result[1], list_result[2], list_result[3])
            
            
            
            print("/n////////////////////////////////////////")

            
            for data_point in list_result[1].landmark:
                
                x = data_point.x*640 #Suitable  for MOT16 images
                y = data_point.y*480 #Suitable  for MOT16 images
                z = data_point.z*640 #Suitable  for MOT16 images
                visability = data_point.visibility
                
                self.viewer.circle(x, y, 3)
                


                

                #landmarks_coordinates = np.vstack([self.landmarks_coordinates, [x,y,z,visability, frame_idx]])
                #landmarks_coordinates_df = pd.DataFrame(self.landmarks_coordinates)
            
            #landmarks_coordinates_df.to_csv("x_y_landmarks.csv")
            return list_result[1]
                
            #print("list_result: \n", landmarks_list)

            #self.viewer.circle(landmarks_list[:,0], landmarks_list[:,1],1)
            
            
            index += 1
            #return drawings
            
            #self.viewer.rectangle(*detection.to_tlbr)
    
    
    
            
###################################################################################################        
            
            
            
        
    """
    #Original Code:
    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
    """
    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
           
           
#
