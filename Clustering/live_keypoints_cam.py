# vim: expandtab:ts=4:sw=4

"""
This module contains an image pose landmarks extraction

Created on Thu Apr  6 16:26:33 2023

@author: andre
"""
"""
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
"""

import matplotlib.pyplot as plt
import pandas as pd
import cv2, sys, os
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


landmarks_names_list = [
                 
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
          
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",  "RIGHT_EAR", "MOUTH_LEFT",
          
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX",
             
    "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
         
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    
]

all_left_landmarks_index_list = [ 
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "LEFT_EAR", "MOUTH_LEFT",
      
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_PINKY", "LEFT_INDEX", 
             
    "LEFT_THUMB",  "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX" 
]


all_right_landmarks_index_list = [ 
    
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "RIGHT_EAR", "MOUTH_RIGHT",
          
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_PINKY", "RIGHT_INDEX", 
             
    "RIGHT_THUMB",  "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"  

]




left_landmarks_short_names_list = [
             
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",  "LEFT_SHOULDER", "LEFT_HIP", 
    
    "LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX"

]

  
right_landmarks_short_names_list = [
             
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_SHOULDER", "RIGHT_HIP", 
    
    "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX"

]




#path = 'my/per_dat/'
#dir_list = os.listdir(path)


class MediaPipePose:
    
    
    length_results_left = np.empty([0, 8])    
    length_results_right =  np.empty([0,8]) #np.empty([len(dir_list), 7])
    
    left_landmarks_indices = []
    right_landmarks_indices = []
    
    hand_length_length = []
    dir_list = ""
    image_width = 0
    image_height = 0
    results = []
    
    length_results_left_df = pd.DataFrame(length_results_left, columns = ['Sho to Elb (L)','Elb to Wrist (L)',' Sho to Hip (L)', 'Hip to Knee (L)', 'Knee to Ankle (L)', 'Ankle to Heel (L)', ' Heel to Foot (L)', 'Track ID'] )
    length_results_right_df = pd.DataFrame(length_results_right, columns = ['Sho to Elb (R)','Elb to Wrist (R)',' Sho to Hip (R)', 'Hip to Knee (R)', 'Knee to Ankle (R)', 'Ankle to Heel (R)', ' Heel to Foot (R)', 'Track ID'])
    
    
    
    length_results_left_df.to_csv("C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/length_results_left.csv")
    length_results_right_df.to_csv("C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/length_results_right.csv")
   

    
    def __init__ (self, image, path_1, frame_idx, dirList = None):
        
        
        #print("PATHHHHH", path_1)
        
        self.image = image
        self.path = path_1
        self.dir_list = dirList
        #self.image_files =  [self.path + x for x in self.dir_list] # Add Path Prefix
        self.idx = frame_idx
        #results = []

        self.depth = np.zeros((1,33)) # 1 row, 33 landmarks depth columns
        self.depth = np.delete(self.depth, (0), axis=0)
        
        
        self.visibility_array_left_index_19 = []     
        left_landmarks_depths = np.zeros(8)
        right_landmarks_depths = np.zeros(8)
        self.temp_img_left_landmarks = []
        self.temp_img_right_landmarks = []
        
        

        
        self.exists = False
        self.mode = "w"
    
    
    visibility = np.zeros((1,33)) # 1 row, 33 landmarks depth columns
    visibility = np.delete(visibility, (0), axis=0)
    #image_files = [path + "00000096.jpg"]
    
    sns.set(font_scale=1)
    
    def euclidean_distance_landmarks(self, landmark1, landmark2, image_width, image_height):
            
        """
        for i in range(len(landmark11)):
            
            landmark11[i] = round(landmark11[i], 3)
            
        for i in range(len(landmark12)):
            
            landmark12[i] = round(landmark12[i], 3)
            
        """
        #pixel to cm = 0.0264583333
        diff_point1 = ((landmark1[0] - landmark2[0]))  ** 2
        diff_point2 = ((landmark1[1] - landmark2[1]))  ** 2
        diff_point3 = ((landmark1[2] - landmark2[2]))  ** 2
        
        return ((diff_point1 + diff_point2 + diff_point3) ** 0.5 ) 
    
        """
        landmark11 = np.array(landmark11) 
        landmark12 = np.array(landmark12) 
        
        euclidean_dist = np.sum((landmark11-landmark12)**2) 
        euclidean_dist = euclidean_dist **0.5
        #euclidean_dist = euclidean_dist * 0.0264583333  
        #print("euclidean_dist", euclidean_dist)
    
        return euclidean_dist
        """
    
        
        """ Euclidean distance between two points point1, point2 """
        
        
      
    def run(self, track_id):
    
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
    
        
        diff_array = []
        
       
        
        #BG_COLOR = (192, 192, 192) # gray
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence= 0.7) as pose:
          
          #for idx, file in tqdm(enumerate(self.image_files)):

            "Original Code - Iterating Over Folder of Images:"
            #for idx, file in tqdm(enumerate(self.image_files)):

              
              
            #image = cv2.imread(self.image) # Commented since image is already cv2.imread obj
            self.image_height, self.image_width, _ = self.image.shape
 
            self.image_height, self.image_width, self.image_depth = np.shape(self.image)
        
            #1920 X 1080    
            
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            
            if results.pose_landmarks is None: # If Pose was NOT detected
                return None
                #print("\n\n\n results......   \n", pose.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)).pose_landmarks, "\n\n")
            
            else:
            
                """
                # BELONGS TO THE UP MENTIONED COMMENTED LOOP (for idx, file) TOGETHER:
                    
                if not results.pose_landmarks:
                  continue
                """
                
                """
                print(
                    f'Nose coordinates: ('
                    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                    f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
                )
                print("HELLO\n")
                
                
                """
                annotated_image = self.image.copy()
                
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                #condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                #bg_image = np.zeros(image.shape, dtype=np.uint8)
                #bg_image[:] = BG_COLOR
                #annotated_image = np.where(condition, annotated_image, bg_image)
            
                # Draw pose landmarks on the image.
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
                # Plot pose world landmarks.
                #mp_drawing.plot_landmarks(
                #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            
                #print(results.pose_world_landmarks)
                
                self.visibility_array_left_index_19.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].visibility)
                cv2.imwrite('C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/pose_results/annotated_image' + str(self.idx) + '.png', annotated_image)
                
                self.calc_landmarks_lengths(results.pose_landmarks, mp_pose, track_id) # Adding the landmarks to the csv length later on
                
                
                #print("results: ", results, "\n")
                lst_result = []
                
                lst_result.append(annotated_image)
                lst_result.append(results.pose_landmarks)
                lst_result.append(mp_pose.POSE_CONNECTIONS)
                lst_result.append(mp_drawing_styles.get_default_pose_landmarks_style())
                
                return annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing_styles.get_default_pose_landmarks_style()
                
                """
                return [annotated_image, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()]
                """
            
            
            
                
    def calc_landmarks_lengths(self, pose_landmarks, mp_pose, track_id):
            
        iteration = 0
        
        """
        for data_point in results.pose_landmarks.landmark:
            
            print(iteration,'x is', data_point.x*image_width, 'y is', data_point.y*image_height, 'z is', data_point.z*image_width,
                'visibility is', data_point.visibility)
            
            iteration+=1
        """
        
        #hello how are ypu? is this working properly?
                
        #def annotate_image_given_landmarks(landmarks):
            
        all_landmarks = arr = np.empty((33,3))
        
        dic = list(pose_landmarks.landmark)
        
        matrix = []
        
        for row in dic:
            
            temp_row = []
            temp_row.append(row.x)
            temp_row.append(row.y) 
            temp_row.append(row.z)
            temp_row.append(row.visibility)
            
            matrix.append(temp_row)
            
        matrix = np.array(matrix)
    
        """
        scaler = MinMaxScaler()
        scaler.fit(all_landmarks)
        scaled_all_landmarks = scaler.transform(all_landmarks)
            
        left_shoulder  = [matrix[11,0]  , 
                          matrix[11,1]  ,
                          matrix[11,2]  ]   
        
        
        print("LEFT\n", left_shoulder)
        
        left_elbow =  [matrix[13,0]  , 
                       matrix[13,1]  ,
                       matrix[13,2]  ]
        
        #########################################################
        
        left_shoulder  = [matrix[11,0] * image_width , 
                          matrix[11,1] * image_height  ,
                          matrix[11,2] * image_width ]   
        
        
        
        print("LEFT\n", left_shoulder)
        
        left_elbow =  [matrix[13,0] * image_width , 
                       matrix[13,1] * image_height ,
                       matrix[13,2] * image_width]
        
        print("RIGHT\n", left_elbow)
    
        """
    
        
        
        #cv2.imwrite('C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/results/annotated_image' + str(self.idx) + '.png', annotated_image)
    
        # Open an Image
        img = Image.open(r'C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/pose_results/annotated_image' + str(self.idx) + '.png')
        #img = Image.open(r'C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/results/annotated_image' + str(self.idx) + '.png')

    
        # Call draw Method to add 2D graphics in an image
        I1 = ImageDraw.Draw(img)
         
        # Custom font style and font size
        #myFont = ImageFont.truetype('FreeMono.ttf', 65)
        font = ImageFont.truetype("arial.ttf", 30)
        
        
        #LEFT:
        # Add Text to an image
        I1.text((10, 10), "Median:", font = ImageFont.truetype("arial.ttf", 30), fill =(255, 0, 0)) 
    

    
        for i in range(len(left_landmarks_short_names_list)-1):
            
            if(i == 2): continue # Avoiding calculating the length between wrist to shoulder (meaning less)
        
            
            landmark_name_1 = left_landmarks_short_names_list[i]
            landmark_name_2 = left_landmarks_short_names_list[i+1]
    
            #print()        
            #print("Landmark1" , landmark_name_1)
            #print("Landmark2" , landmark_name_2)
    
            
            landmark_index_1 = landmarks_names_list.index(landmark_name_1)
            landmark_index_2 = landmarks_names_list.index(landmark_name_2)
            
            
            first_landmark = [matrix[landmark_index_1,0] * self.image_width , 
                              matrix[landmark_index_1,1] * self.image_height ,
                              matrix[landmark_index_1,2] * self.image_width ]   
            
                    
            second_landmark = [matrix[landmark_index_2,0] * self.image_width , 
                               matrix[landmark_index_2,1] * self.image_height,
                               matrix[landmark_index_2,2] * self.image_width] 
    
            #left_landmarks_depths[i] = (matrix[landmark_index_1,2] + matrix[landmark_index_2,2]) / 2
            if(matrix[landmark_index_1,3] >= 0.5 and matrix[landmark_index_2,3] >= 0.5):
                res1 = self.euclidean_distance_landmarks(first_landmark, second_landmark, self.image_width, self.image_height)
            
            else:
                res1 = 0
            
            distance_x_coordinates = ((matrix[landmark_index_1,0] * self.image_width) + (matrix[landmark_index_2,0] * self.image_width)) / 2
            distance_y_coordinates = ((matrix[landmark_index_1,1] * self.image_height) + (matrix[landmark_index_2,1] * self.image_height)) / 2
            
            I1.text((distance_x_coordinates, distance_y_coordinates), str(round(res1/7.8577,3)), font = font, fill =(255, 0, 0)) # Shoulder to elbow
    
        
            self.temp_img_left_landmarks.append(round(res1,3))
        
    
    
        #### Draw Lengths On Images: ####################################################################################
        
        """   
        I1.text((first_landmark, 260), str(temp_img_left_landmarks[0]), font = font, fill =(255, 0, 0)) # Shoulder to elbow
        I1.text((501, 321), str(temp_img_left_landmarks[1]), font = font, fill =(255, 0, 0)) # Elbow to wrist
        I1.text((414, 479), str(temp_img_left_landmarks[2]), font = font, fill =(255, 0, 0)) # Hip to knee
        I1.text((412, 622), str(temp_img_left_landmarks[3]), font = font, fill =(255, 0, 0)) # knee to ankle
        I1.text((340, 692), str(temp_img_left_landmarks[4]), font = font, fill =(255, 0, 0)) # ankle to heel
        I1.text((380, 724), str(temp_img_left_landmarks[5]), font = font, fill =(255, 0, 0)) # Heel to foot
        """
    
        #################################################################################################################
    
        
        
        sns.set(font_scale=8)
        
        #print("IMAGGGGE", type(self.temp_img_left_landmarks))
        
        self.temp_img_left_landmarks.append(track_id)
        self.temp_img_left_landmarks = np.asarray(self.temp_img_left_landmarks)
        
        #print("temp_img_left_landmarks\n", self.temp_img_left_landmarks)
        #length_results_left = np.asarray(length_results_left)
        
        
        self.length_results_left = np.vstack([self.length_results_left, self.temp_img_left_landmarks])
        
        #print("temp_img_left_landmarks\n", self.length_results_left)

        
        #self.length_results_left = self.length_results_left # / 7.8577
        np.delete(self.length_results_left,2)

        
        
        temp_img_right_landmarks = []
        
        for i in range(len(right_landmarks_short_names_list)-1):
            
            if(i == 2): continue #Avoiding calculating the length between wrist to shoulder (meaning less)
    
    
            landmark_name_1 = right_landmarks_short_names_list[i]
            landmark_name_2 = right_landmarks_short_names_list[i+1]
            
            landmark_index_1 = landmarks_names_list.index(landmark_name_1)
            landmark_index_2 = landmarks_names_list.index(landmark_name_2)
       
            first_landmark = [matrix[landmark_index_1,0] * self.image_width , 
                              matrix[landmark_index_1,1] * self.image_height ,
                              matrix[landmark_index_1,2] * self.image_width ]   
            
                    
            second_landmark = [matrix[landmark_index_2,0] * self.image_width , 
                               matrix[landmark_index_2,1] * self.image_height,
                               matrix[landmark_index_2,2] * self.image_width] 
       
        
            #left_landmarks_depths[i] = (matrix[landmark_index_1,2] + matrix[landmark_index_2,2]) / 2
            if(matrix[landmark_index_1,3] >= 0.5 and matrix[landmark_index_2,3] >= 0.5 ):
                res1 = self.euclidean_distance_landmarks(first_landmark, second_landmark, self.image_width, self.image_height)
            
            else:
                res1 = 0
            
            distance_x_coordinates = ((matrix[landmark_index_1,0]  * self.image_width) + (matrix[landmark_index_2,0]  * self.image_width)) / 2
            distance_y_coordinates = ((matrix[landmark_index_1,1]  * self.image_height) + (matrix[landmark_index_2,1]  * self.image_height)) / 2
            
            I1.text((distance_x_coordinates, distance_y_coordinates), str(round(res1/7.8577,3)), font = font, fill =(255, 0, 0)) # Shoulder to elbow
    
        
            self.temp_img_right_landmarks.append(round(res1,3))
        
        
        
        #### Draw Lengths On Images: ####################################################################################
    
        """    
        #RIGHT:
        # Add Text to an image
        I1.text((170, 260), str(temp_img_right_landmarks[0]), font = font, fill =(255, 0, 0)) # Shoulder to elbow
        I1.text((120, 321), str(temp_img_right_landmarks[1]), font = font, fill =(255, 0, 0)) # Elbow to wrist
        I1.text((190, 479), str(temp_img_right_landmarks[2]), font = font, fill =(255, 0, 0)) # Hip to knee
        I1.text((195, 622), str(temp_img_right_landmarks[3]), font = font, fill =(255, 0, 0)) # knee to ankle
        I1.text((280, 692), str(temp_img_right_landmarks[4]), font = font, fill =(255, 0, 0)) # ankle to heel
        I1.text((230, 724), str(temp_img_right_landmarks[5]), font = font, fill =(255, 0, 0)) # Heel to foot
        """
        
        # Display edited image
        #img.show()
         
        # Save the edited image
        img.save('C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/annotated_image' + str(self.idx) + '.png')
        #################################################################################################################
        
        
        single_current_image_landmarks_depth_row = abs(matrix[:,2])
        single_current_image_landmarks_visibility_row = abs(matrix[:,3])
    
        self.depth = np.vstack([self.depth, single_current_image_landmarks_depth_row])
        self.visibility = np.vstack([self.visibility, single_current_image_landmarks_visibility_row])
           
        self.temp_img_right_landmarks.append(track_id)
        self.temp_img_right_landmarks = np.asarray(self.temp_img_right_landmarks)

        
        
        #print("length_results_right: \n", length_results_right)

        
        
        self.length_results_right = np.vstack([self.length_results_right, self.temp_img_right_landmarks])
       
        np.delete(self.length_results_right,2)
        
        #print("length_results_right: \n", length_results_right)
    
    
    
    
    
    
    
    
    
    
    
    
    
        """
        print("LEFT\n",left_shoulder)
        print("RIGHT\n",left_elbow)
    
    
         left_shoulder  = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width  , 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height ,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z  ]
         #print("RIGHT\n", right_shoulder)
         
         left_elbow =  [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width ,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].z ]
        """    
        #print("LEFT\n", right_elbow)
    
        
        left_elbow  = [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x* self.image_width, 
                          pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y* self.image_height ]
        
        left_wrist =  [pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x* self.image_width,
                           pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y* self.image_height ]
            
                           #results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].z* image_width ]
    
        #res2 = euclidean_distance_landmarks(left_elbow, left_wrist)
        
        #res3 = res1+res2
        
        #diff_array.append(res1 / res3)
    
        self.hand_length_length.append(res1)
        
        """
        try:
            landmarks = results.pose_landmarks.landmark
            print(landmarks)
        except:
            pass
        
        print("Median:", np.median(diff_array))
        print("Mean:", np.mean(diff_array))
        print("diff_array", diff_array)
    
    
        """
        
        #self.length_results_left[:-1] = self.length_results_left[:-1] / 7.8577
        self.length_results_left = self.length_results_left.round(3)
        
        #print("temp_img_left_landmarks2222\n", self.length_results_left)

        
        
        #self.length_results_right[:-1] = self.length_results_right[:-1] / 7.8577
        self.length_results_right = self.length_results_right.round(3)

        
        
        #print("length_results_right...: \n", self.length_results_right)


        self.length_results_left_df = pd.DataFrame(self.length_results_left, columns = ['Sho to Elb (L)','Elb to Wrist (L)',' Sho to Hip (L)', 'Hip to Knee (L)', 'Knee to Ankle (L)', 'Ankle to Heel (L)', ' Heel to Foot (L)', 'Track ID'] )
        self.length_results_right_df = pd.DataFrame(self.length_results_right, columns = ['Sho to Elb (R)','Elb to Wrist (R)',' Sho to Hip (R)', 'Hip to Knee (R)', 'Knee to Ankle (R)', 'Ankle to Heel (R)', ' Heel to Foot (R)', 'Track ID'])
        #print("length_results_left_d\n", self.length_results_left_df.head())

        
        """
        from pathlib import Path
        my_file_left = Path("C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/length_results_left.csv")
        my_file_right = Path("C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/length_results_right.csv")
        """
    
        self.length_results_left_df.to_csv("C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/length_results_left.csv", mode = "a", index = False, header= False)
        self.length_results_right_df.to_csv("C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/my/multi person/Multiple Object Tracking/application_util/calc_length/length_results_right.csv", mode = "a", index = False, header= False)
       
            
        #print("length_results_left_df: \n", self.length_results_left_df, "\n")
        #print("length_results_right_df: \n", self.length_results_right_df, "\n")

        

         
        
        
########################################################################################
    
        
    def get_left_right_landmarks_indices(self):
       
        # Getting indices of left landmarks:
        
       
        
        #for left side
        for i in range(len(all_left_landmarks_index_list)):
        
            landmark_name = all_left_landmarks_index_list[i]    
            landmark_index = landmarks_names_list.index(landmark_name)
            self.left_landmarks_indices.append(landmark_index)
            
        #for right side
        for i in range(len(all_right_landmarks_index_list)):
        
            landmark_name = all_right_landmarks_index_list[i]    
            landmark_index = landmarks_names_list.index(landmark_name)
            self.right_landmarks_indices.append(landmark_index)
        
    def create_mean_median_summary_images(self): 
        
        left_depth = np.take(self.depth, self.left_landmarks_indices, 1)
        right_depth = np.take(self.depth, self.right_landmarks_indices, 1 )
        
        
        print("left_depth", self.depth)
        print("all_left_landmarks_index_list", all_left_landmarks_index_list)
        
        
        depth_of_left_landmarks = pd.DataFrame(data = left_depth, columns = all_left_landmarks_index_list) 
        
        depth_of_right_landmarks = pd.DataFrame(data = right_depth, columns = all_right_landmarks_index_list) 
        
        #### Statistics: ###################################################################################
        
        
        
        print("LEFT\n", depth_of_left_landmarks.describe())
        print("RIGHT\n", depth_of_right_landmarks.describe())
        print("\n\n\n\n\n")
        print("LEFT MEAN", depth_of_left_landmarks.mean().mean())
        print("RIGHT MEAN", depth_of_right_landmarks.mean().mean())
        
        
        self.visibility = pd.DataFrame(self.visibility)
        #visibility.to_csv("96_images_visibility.csv")
        
        self.depth = pd.DataFrame(self.depth)
        #depth.to_csv("96_images_depth.csv")
        
        """
        #plt.rc('font', size=20)          # controls default text sizes
        plt.title("Depths Of Left Landmarks Plot111")
        plt.xticks(np.arange(0, 100,20))
        plt.xlabel("Number Of Image")
        plt.ylabel("Left Depths")
        """
        
        plt.subplots(figsize=(90, 50))
        
        gfg = sns.lineplot(data=depth_of_left_landmarks.iloc[:,:8], lw = 10)
        sns.move_legend(gfg, "upper left", bbox_to_anchor=(1, 1))
        plt.setp(gfg.get_legend().get_texts(), fontsize='50') 
        
        # get the legend object
        leg = gfg.legend()
        
        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(12.0)
        
        plt.title("Line Plot Representing The Depths of Left Side Landmarks")
        plt.xticks(np.arange(0, len(self.dir_list),2)) # Dir_list: number of images for x axis
        plt.yticks(np.arange(0, 1.3,0.2))
        gfg.set_xlabel("Image Number",fontsize=30)
        gfg.set_ylabel("Depths",fontsize=20)
        
        
        plt.show()
        
        
        #plt.subplots(figsize=(90, 50))
        #sns.set(font_scale=1)
        
        plt.subplots(figsize=(90, 50))
        sns.set(font_scale=6)
        
        gfg = sns.lineplot(data=depth_of_right_landmarks.iloc[:,:8], lw = 10)
        sns.move_legend(gfg, "upper left", bbox_to_anchor=(1, 1))
        plt.setp(gfg.get_legend().get_texts(), fontsize='50') 
        
        
        # get the legend object
        leg = gfg.legend()
        
        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(12.0)
        
        plt.title("Line Plot Representing The Depths of Right Side Landmarks")
        plt.xticks(np.arange(0, len(self.dir_list),2)) # Dir_list: number of images for x axis
        plt.yticks(np.arange(0, 1.3,0.2))
        plt.xlabel("Number Of Image")
        plt.ylabel("Image Number")
        plt.show()
        
        #gfg = sns.lineplot(data=depth_of_left_landmarks.iloc[:,:8])
        #plt.setp(gfg.get_legend().get_texts(), fontsize='3')  
        #gfg.legend(bbox_to_anchor= (1,1))
        #fig, ax = plt.subplots(figsize=(120, 50))
        #plt.show()
        
        """
        gfg = sns.lineplot(data=depth_of_right_landmarks.iloc[:,:8])
        plt.setp(gfg.get_legend().get_texts(), fontsize='30')  
        gfg.legend(bbox_to_anchor= (1.4,1))
        fig, ax = plt.subplots(figsize=(70, 10))
        
        plt.title("Depths Of Right Landmarks Plot")
        plt.xticks(np.arange(0, 160,30))
        plt.xlabel("Number Of Image")
        plt.ylabel("Right Depths")
        plt.show()
        
        median_depth_33_landmarks = np.median(depth_of_right_landmarks, axis = 0)
        median_depth_33_landmarks = list(median_depth_33_landmarks.reshape((16, 1)))
        
        """
        
        ########################################################################################
        
        hand_length_length = sorted(self.hand_length_length)
        
        hand_length_length = np.array(hand_length_length) 
        print(hand_length_length)
        print("HAND LENGTH MEANS", np.mean(hand_length_length ))
        
        min_range = np.min(hand_length_length/7.8577)
        max_range = np.max(hand_length_length/7.8577)
        median = np.median(hand_length_length/7.8577)
        #37.7952755906 
        
        print("median", median)
        print("depth mean", np.mean(self.depth))
        
        median = round(median, 3)
        
        mean = np.mean(np.array(self.depth))
        
        print("mean: \n", self.length_results_left[len(self.dir_list):,:], "\n\n\n\n")
        median_left_landmarks_short_names_list = np.around( np.median(self.length_results_left[len(self.dir_list):,:]/7.8577, axis = 0), decimals=3)
        median_right_landmarks_short_names_list =  np.around( np.median(self.length_results_right[len(self.dir_list):,:]/7.8577, axis = 0), decimals=3)
        
        
        mean_left_landmarks_short_names_list =  np.around( np.mean(self.length_results_left[len(self.dir_list):,:]/7.8577, axis = 0), decimals=3)
        mean_right_landmarks_short_names_list =  np.around (np.mean(self.length_results_right[len(self.dir_list):,:]/7.8577, axis = 0), decimals=3)
        
        
        ### Writing Annotation on image ##########################################################
        
        # Importing the PIL library
        from PIL import Image
        from PIL import ImageDraw
        from PIL import ImageFont
         
        # Open an Image
        img = Image.open('./general_landmarks2.png')
         
        # Call draw Method to add 2D graphics in an image
        I1 = ImageDraw.Draw(img)
         
        # Custom font style and font size
        #myFont = ImageFont.truetype('FreeMono.ttf', 65)
        font = ImageFont.truetype("arial.ttf", 15)
        
        
        #LEFT:
        # Add Text to an image
        
        I1.text((10, 10), "Mean:", font = ImageFont.truetype("arial.ttf", 40), fill =(255, 0, 0)) 
        
        I1.text((440, 260), str(mean_left_landmarks_short_names_list[0]), font = font, fill =(255, 0, 0)) # Shoulder to elbow
        I1.text((501, 321), str(mean_left_landmarks_short_names_list[1]), font = font, fill =(255, 0, 0)) # Elbow to wrist
        I1.text((414, 479), str(mean_left_landmarks_short_names_list[2]), font = font, fill =(255, 0, 0)) # Hip to knee
        I1.text((412, 622), str(mean_left_landmarks_short_names_list[3]), font = font, fill =(255, 0, 0)) # knee to ankle
        I1.text((340, 692), str(mean_left_landmarks_short_names_list[4]), font = font, fill =(255, 0, 0)) # ankle to heel
        I1.text((380, 724), str(mean_left_landmarks_short_names_list[5]), font = font, fill =(255, 0, 0)) # Heel to foot
        
        
        #RIGHT:
        # Add Text to an image
        I1.text((170, 260), str(mean_right_landmarks_short_names_list[0]), font = font, fill =(255, 0, 0)) # Shoulder to elbow
        I1.text((120, 321), str(mean_right_landmarks_short_names_list[1]), font = font, fill =(255, 0, 0)) # Elbow to wrist
        I1.text((190, 479), str(mean_right_landmarks_short_names_list[2]), font = font, fill =(255, 0, 0)) # Hip to knee
        I1.text((195, 622), str(mean_right_landmarks_short_names_list[3]), font = font, fill =(255, 0, 0)) # knee to ankle
        I1.text((280, 692), str(mean_right_landmarks_short_names_list[4]), font = font, fill =(255, 0, 0)) # ankle to heel
        I1.text((230, 724), str(mean_right_landmarks_short_names_list[5]), font = font, fill =(255, 0, 0)) # Heel to foot
        
        
        # Display edited image
        img.show()
         
        # Save the edited image
        img.save("mean.png")
        
        
        #### Mean: ##########################################################
        
        # Open an Image
        img = Image.open('./general_landmarks2.png')
        
        # Call draw Method to add 2D graphics in an image
        I1 = ImageDraw.Draw(img)
         
        # Custom font style and font size
        #myFont = ImageFont.truetype('FreeMono.ttf', 65)
        font = ImageFont.truetype("arial.ttf", 15)
        
        
        #LEFT:
        # Add Text to an image
        I1.text((10, 10), "Median:", font = ImageFont.truetype("arial.ttf", 30), fill =(255, 0, 0)) 
        
        I1.text((440, 260), str(median_left_landmarks_short_names_list[0]), font = font, fill =(255, 0, 0)) # Shoulder to elbow
        I1.text((501, 321), str(median_left_landmarks_short_names_list[1]), font = font, fill =(255, 0, 0)) # Elbow to wrist
        I1.text((225, 343), str(median_left_landmarks_short_names_list[3]), font = font, fill =(255, 0, 0)) # Hip to knee
        I1.text((412, 622), str(median_left_landmarks_short_names_list[4]), font = font, fill =(255, 0, 0)) # knee to ankle
        I1.text((340, 692), str(median_left_landmarks_short_names_list[5]), font = font, fill =(255, 0, 0)) # ankle to heel
        I1.text((380, 724), str(median_left_landmarks_short_names_list[6]), font = font, fill =(255, 0, 0)) # Heel to foot
        
        
        
        #RIGHT:
        # Add Text to an image
        I1.text((170, 260), str(median_right_landmarks_short_names_list[0]), font = font, fill =(255, 0, 0)) # Shoulder to elbow
        I1.text((120, 321), str(median_right_landmarks_short_names_list[1]), font = font, fill =(255, 0, 0)) # Elbow to wrist
        I1.text((418, 350), str(median_right_landmarks_short_names_list[3]), font = font, fill =(255, 0, 0)) # Hip to knee
        I1.text((195, 622), str(median_right_landmarks_short_names_list[4]), font = font, fill =(255, 0, 0)) # knee to ankle
        I1.text((280, 692), str(median_right_landmarks_short_names_list[5]), font = font, fill =(255, 0, 0)) # ankle to heel
        I1.text((230, 724), str(median_right_landmarks_short_names_list[6]), font = font, fill =(255, 0, 0)) # Heel to foot
        
        """
        from colour import Color
        yellow = Color("yellow")
        colors = list(yellow.range_to(Color("blue"),4))
        """
        
        
        depth_max = np.max(self.depth, axis = 0)
        
        depth_median = np.median(self.depth, axis = 0)
        
        
        values = np.linspace(0,1,30)
        print(values)
        colors = [f"rgb({round(v*255)}, {round(v*255)}, 255)" for v in values][::-1]
        
        depth_median = depth_median.T
        relevant_depths = []
        relevant_depths.append(depth_median[11])
        relevant_depths.append(depth_median[12])
        relevant_depths.append(depth_median[13])
        relevant_depths.append(depth_median[14])
        
        relevant_depths.append(depth_median[15])
        relevant_depths.append(depth_median[16])
        relevant_depths.append(depth_median[17])
        relevant_depths.append(depth_median[18])
        
        relevant_depths.append(depth_median[19])
        relevant_depths.append(depth_median[20])
        relevant_depths.append(depth_median[21])
        
        relevant_depths.append(depth_median[22])
        relevant_depths.append(depth_median[23])
        relevant_depths.append(depth_median[24])
        relevant_depths.append(depth_median[25])
        relevant_depths.append(depth_median[26])
        relevant_depths.append(depth_median[27])
        relevant_depths.append(depth_median[28])
        
        relevant_depths.append(depth_median[29])
        relevant_depths.append(depth_median[30])
        relevant_depths.append(depth_median[31])
        relevant_depths.append(depth_median[32])
        
        
        relevant_depths = np.array(relevant_depths)
        relevant_depths = relevant_depths.argsort()[::-1]
        
        
        #I1.ellipse((336,132,347,140), fill="blue", outline = "blue")
        I1.ellipse((407,252,419,261), fill= colors[relevant_depths[0]]) # 11
        I1.ellipse((227,252,239,261), fill= colors[relevant_depths[1]]) # 12
        I1.ellipse((467,313,478,321), fill= colors[relevant_depths[2]]) # 13 
        I1.ellipse((165,309,176,321), fill= colors[relevant_depths[3]]) # 14 
        I1.ellipse((525,286,536,294), fill= colors[relevant_depths[4]]) # 15 e
        I1.ellipse((108,283,119,291), fill= colors[relevant_depths[5]]) # 16 e
        I1.ellipse((573,284,584,292), fill= colors[relevant_depths[6]]) # 17 e
        I1.ellipse((60,286,71,294), fill= colors[relevant_depths[7]]) # 18 e
        I1.ellipse((558,243,569,251), fill= colors[relevant_depths[8]]) # 19 e
        I1.ellipse((76,243,87,253), fill= colors[relevant_depths[9]]) # 20 e
        I1.ellipse((524,256,535,264), fill= colors[relevant_depths[10]]) # 21 e
        I1.ellipse((112,256,123,265), fill= colors[relevant_depths[11]]) # 22 e
        
        I1.ellipse((378,414,390,423), fill= colors[relevant_depths[12]]) # 23
        I1.ellipse((253,414,265,423), fill= colors[relevant_depths[13]]) # 24
        
        I1.ellipse((407,544,419,552), fill= colors[relevant_depths[14]]) # 25
        I1.ellipse((222,544,232,552), fill= colors[relevant_depths[15]]) # 26
        I1.ellipse((375,673,383,681), fill= colors[relevant_depths[16]]) # 27
        I1.ellipse((252,674,264,682), fill= colors[relevant_depths[17]]) # 28
        
        
        I1.ellipse((356,710,367,718), fill= colors[relevant_depths[18]]) # 29
        I1.ellipse((272,709,283,719), fill= colors[relevant_depths[19]]) # 30
        I1.ellipse((421,715,432,724), fill= colors[relevant_depths[20]]) # 31
        I1.ellipse((204,715,215,726), fill= colors[relevant_depths[21]]) # 32
        
        
        
        
        
        
        
        # Display edited image
        img.show()
        
         
        # Save the edited image
        img.save("median.png")
        
        """
        import cv2
        image = cv2.imread('median.png') 
        image = cv2.circle(img, (755,171), 30, (255,133,233), -1) 
        cv2.imshow("Median", image)  
        
        image.save("median3.png")
        """
        
    
   
    
    
    
    def length_correlation(self):
    
        #### Correlation: ##########################################################
        
        df = pd.DataFrame(self.length_results_left[3:,:], columns = ['Sho to Elb','Elb to Wrist',' Wrist to Hip', 'Hip to Knee', 'Knee to Ankle', 'Ankle to Heel', ' Heel to Foot'])
        
        corr = df.corr()
        
        
        # plot the heatmap
        sns.set(font_scale=1)
        sns.heatmap(corr, cmap="crest", annot=True)
        
        plt.title("Distance Correlations")
        plt.show()
        
    
    
    def create_plots(self):
    
        #sns.set(font_scale=2)
        
        plt.plot(range(len(self.visibility_array_left_index_19)), self.visibility_array_left_index_19)
        plt.xlabel ("Image Number")
        plt.title("LEFT HAND LANDMARK 19 visibility")
        plt.ylabel("Visibility")
        plt.xticks(np.arange(0, len(self.dir_list),5))
        plt.show()
        
        plt.plot(range(len(self.visibility_array_left_index_19)), self.visibility_array_left_index_19)
        plt.xlabel ("Image Number")
        plt.title("LEFT HAND LANDMARK 19 visibility")
        plt.ylabel("Visibility")
        plt.xticks(np.arange(0, len(self.dir_list),5))
        
        plt.show()
    
    
        for val in range(len(self.visibility_array_left_index_19)):
            
            self.visibility_array_left_index_19[val] = 1 - self.visibility_array_left_index_19[val]
            
        
        plt.plot(range(len(self.visibility_array_left_index_19)), self.visibility_array_left_index_19)
        plt.xlabel ("Image Number")
        plt.title("LEFT HAND LANDMARK 19 visibility")
        plt.ylabel("Visibility")
        plt.xticks(np.arange(0, len(self.dir_list),5))
        
        plt.show()
        
        
        #### Bones Length: ##########################################################
        
        
        
        #print("lllllllll\n", length_results_left[:len(dir_list),:])
        length_results_left = self.length_results_left / 7.8577
        length_results_left = self.length_results_left.round(3)
        
        length_results_right = self.length_results_right / 7.8577
        length_results_right = self.length_results_right.round(3)
        
        length_results_left_df = pd.DataFrame(length_results_left, columns = ['Sho to Elb (L)','Elb to Wrist (L)',' Sho to Hip (L)', 'Hip to Knee (L)', 'Knee to Ankle (L)', 'Ankle to Heel (L)', ' Heel to Foot (L)'] )
        length_results_right_df = pd.DataFrame(length_results_right, columns = ['Sho to Elb (R)','Elb to Wrist (R)',' Sho to Hip (R)', 'Hip to Knee (R)', 'Knee to Ankle (R)', 'Ankle to Heel (R)', ' Heel to Foot (R)'])
        
        length_results_left_df.to_csv("C:/Users/andre/Desktop/Haifa Uni/Thesis/length_results_left.csv")
        length_results_right_df.to_csv("C:/Users/andre/Desktop/Haifa Uni/Thesis/length_results_right.csv")
        
        
        
        for i in range(7):
            
            sns.set(font_scale=1)
            
            # plotting a histogram
            ax = sns.histplot([length_results_left_df.iloc[:,i], length_results_right_df.iloc[:,i]] ,
                              bins=50,
                              kde=True,
                              color='blue'
                              ) #hist_kws={"linewidth": 15,'alpha':1})
            ax.set(xlabel='Normal Distribution', ylabel='Frequency')
            
            
            kdeline = ax.lines[0]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()
            mode_idx = np.argmax(ys)
            ax.vlines(xs[mode_idx], 0, ys[mode_idx], color='red', ls='--', lw=2)
            kde_max_point_left = round(xs[mode_idx], 2)
        
            
            kdeline = ax.lines[1]
            xs = kdeline.get_xdata()
            ys = kdeline.get_ydata()
            mode_idx = np.argmax(ys)
            ax.vlines(xs[mode_idx], 0, ys[mode_idx], color='red', ls='--', lw=2)
            kde_max_point_right = round(xs[mode_idx], 2)
            
            plt.title(length_results_left_df.columns[i][:-4])
            ax.legend(loc='best', labels=["Left - Max: " + str(kde_max_point_left) , "Right - Max: " + str(kde_max_point_right) ])
        
           
            
            plt.show()
            
        
        
        #print(df.head(10))
        
        
        #print(df.tail(10))
        
        
        
        
        
        
        
        
        # import packages
        import scipy.stats as stats
        import seaborn as sns
        import matplotlib.pyplot as plt
          
        """
        sns.set(font_scale=2)
        
        # plotting a histogram
        ax = sns.histplot(hand_length_length / 7.8577,
                          bins=50,
                          kde=True,
                          color='blue',
                          ) # hist_kws={"linewidth": 15,'alpha':1}
        ax.set(xlabel='Normal Distribution', ylabel='Frequency')
         
        plt.show()
        """
        
        
        for i in range(len(self.hand_length_length)):
            
            self.hand_length_length[i] = round(self.hand_length_length[i], 2)
        
        """
        # plotting a histogram
        ax = sns.distplot(hand_length_length,
                          bins=30,
                          kde=True,
                          hist_kws={"linewidth": 15,'alpha':1})
        ax.set(xlabel='Normal Distribution', ylabel='Frequency')
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        mode_idx = np.argmax(ys)
        ax.vlines(xs[mode_idx], 0, ys[mode_idx], color='red', ls='--', lw=2)
        plt.show()
        """
        
        
        ax = sns.histplot(self.hand_length_length / 7.8577, bins=30, kde=True, color='blue')
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        mode_idx = np.argmax(ys)
        ax.vlines(xs[mode_idx], 0, ys[mode_idx], color='red', ls='--', lw=2)
        
        kde_max_point = round(xs[mode_idx], 3)
        plt.title("KDE Plot - Max at: " + str(kde_max_point))
        plt.xlabel("Distribution")
        plt.show()
        
        
        sns.set(font_scale=1)
    
    




min_range = 0

max_range = 0
    
"""
 
# objective function
def objective(x, true_value):
 return (x - true_value)**2.0
 
# derivative of objective function
def derivative(x, true_value):
    return  2 * (x - true_value) 
 
# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size, true_value):
    # track all solutions
    solutions, scores = list(), list()
    # generate an initial point
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # run the gradient descent
    for i in range(n_iter):
        # calculate gradient
        gradient = derivative(solution, true_value)
        # take a step
        solution = solution - step_size * gradient
        # evaluate candidate point
        solution_eval = objective(solution, true_value)
        # store solution
        solutions.append(solution)
        scores.append(solution_eval)
    # report progress
    print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
    return [solutions, scores]

# define range for input
bounds = np.asarray([[min_range, max_range]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1

init_state = np.random.randint(min_range, max_range)

# perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size, init_state)
# sample input range uniformly at 0.1 increments
inputs = np.arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs, init_state)
# create a line plot of input vs result
plt.plot(inputs, results)
# plot the solutions found
plt.plot(solutions, scores, '.-', color='red')


# show the plot
plt.show()




print("PART 2")


# define range for input
bounds = np.asarray([[min_range, max_range]])
# define the total iterations
n_iter = 30
# define the step size
step_size = 0.1

init_state = np.random.randint(min_range, max_range)
# perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size, init_state)
# sample input range uniformly at 0.1 increments
inputs = np.arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs, init_state)
# create a line plot of input vs result
plt.plot(inputs, results)
# plot the solutions found
plt.plot(solutions, scores, '.-', color='red') #'.-'
plt.title("Second graph")

# show the plot
plt.show()




#diff_array = np.array(diff_array)
#diff_array = diff_array.reshape(-1,1)    
#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(diff_array)
#print(kde.score_samples(diff_array))
#plt.show()



"""

"""
#Put the landmark on the corect position:


print("LEFT")
print("X", results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x)
print("Y", results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y) 
print("Z", results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].z)


print("width", image_width)
print("height", image_height)


# Get picture with the correct elbow kandmark:
results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x = 1074/image_width
results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y = 315/image_height





annotated_image = image.copy()

mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

cv2.imwrite('C:/Users/andre/Desktop/Haifa Uni/Thesis/MediaPipe - Landmark/results/annotated_imageELBOW' + str(idx) + '.png', annotated_image)



"""




  
#print("width", image_width, "\n", "height", image_height, "\n")


"""
print("LANDMARKS\n", results.pose_landmarks.landmark[13], "\n")
print("LANDMARKS\n", results.pose_landmarks.landmark[14], "\n")


    
#def annotate_image_given_landmarks(landmarks):
    
res1 = euclidean_distance_landmarks(left_shoulder, right_shoulder)


left_shoulder  = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x* image_width, 
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y* image_height, 
                  results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z* image_width ]


right_shoulder =  [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x* image_width,
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y* image_height, 
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].z* image_width ]
    

res2 = euclidean_distance_landmarks(left_shoulder, right_shoulder)
    
    
print("Diff:", res1 / res2)
    

sys.exit()

"""





"""
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    [
    print('x is', data_point.x, 'y is', data_point.y, 'z is', data_point.z,
          'visibility is', data_point.visibility)
    for data_point in results.pose_landmarks.landmark
    ]
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

print("HELLO2")

cap.release()
"""
