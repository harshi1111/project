# project
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.
## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.
## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.

--- 

## Program and Output:
## I)Perform ROI from an image
### Step 1: Import necessary packages
```
import cv2
import numpy as np
```
### Step 2: Read the image and convert the image into RGB
```
img = cv2.imread(r"C:\Users\admin\Downloads\digitalimage\moana.jpg")  # Replace with your image path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
### Step 3: Display the original image
```
cv2.imshow("Original Image", img)
```

![image](https://github.com/user-attachments/assets/6867fa0c-cbf4-47bf-ac64-325fa24a45b1)

### Step 4: Set the pixels to display the ROI
```
mask = np.zeros(img.shape[:2], dtype=np.uint8)
x, y, w, h = 100, 100, 200, 200  # Adjust ROI as needed
mask[y:y+h, x:x+w] = 255  # White rectangle on black mask
```
### Step 5: Perform bitwise conjunction of the two arrays using bitwise_and
```
segmented_roi = cv2.bitwise_and(img, img, mask=mask)
```
### Step 6: Display the segmented ROI from an image
```
cv2.imshow("Segmented ROI (Masked)", segmented_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![image](https://github.com/user-attachments/assets/4fcd6109-7c31-4776-a019-638d71a76ec9)


