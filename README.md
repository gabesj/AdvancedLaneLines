## Advanced Lane Finding
**Gabe Johnson**

### This is my completed Advanced Lane Finding project for Udacity's Self-Driving Car Engineer Nanodegree.  It is in the format of a Jupyter Notebook, titled AdvancedLaneLines.ipynb

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Save processing time by using a parallel pixel-finding function with a more targeted search area
* Use a Class data type to accumulate video frame data for use in validating/smoothing the outputs
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## **Setup**
---

This code is written in a Jupyter Notebook using Python.  In order to run it, you will need a python environment with the necessary libraries.  You can follow Udacity's setup instructions for that here: [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md)

[//]: # (Image References)

[image1]: ./process_steps_pictures/undistorted.jpg "Undistorted Image"
[image2]: ./process_steps_pictures/region_of_interest.jpg "Region of Interest"
[image3]: ./process_steps_pictures/perspective_transform.jpg "Perspective Transform"
[image4]: ./process_steps_pictures/HSV_Cylinder.png "HSV Cylinder"
[image5]: ./process_steps_pictures/binary_image.jpg "Binary Image"
[image6]: ./process_steps_pictures/polynomial_fit.jpg "Polynomial Fit"
[image7]: ./process_steps_pictures/curve_and_offset.jpg "Curve and Offset"
[image8]: ./test_images_output/highlighted_test4.jpg "Curve and Offset"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the section titled "Camera Calibration".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function.

**Here is an image before distortion correction:**

<img src="./camera_cal/calibration1.jpg" width="427" height="240" />

**Here is that same image after distortion correction**

<img src="./camera_cal_results/calibration1_undist.jpg" width="427" height="240" />

### Process (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the parameters determined in the Camera Calibration section, I used OpenCV's `.undistort()` function to get a distortion-corrected image such as this:
![alt text][image1]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The section titled "Determine the Region of Interest" is where the area in which to search for lane lines is defined.  The verticies of this trapezoid are defined as variables in terms of the image size

```python
area_start_point_1 = (int(img_size[0]*0),int(img_size[1]*0.95))
area_end_point_1 = (int(img_size[0]*0.408), int(img_size[1]*0.65))

area_start_point_2 = (int(img_size[0]*1),int(img_size[1]*0.95))
area_end_point_2 = (int(img_size[0]*0.59), int(img_size[1]*0.65))

```
I then drew a representation of this area on a sample image for convenience when setting up the variables mentioned above.  This image shows the left and right edges of the area which will be searched for lane lines:
![alt text][image2]

I then used OpenCV's `.getPerspectiveTransform()` to determine the tranformation matrix that will map the region of interest from the trapezoid defined above onto a rectangle the size of the image, representing a top-down view.  I used that matrix in OpenCV's `warpPerspective()` to perform the transformation.
I then verified that my perspective transform was working as expected by performing the transformation on an image of a straight road.  Since there is no curve in the lane lines, the transformed image should show the two parallel lane lines vertically in a birds-eye fashion, which it does:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The next step was to identify which pixels in the image are the lane lines.  Since the lane lines are either white or yellow, I extracted only these two colors using the function `pull_yellow_white()`.  First I converted a copy of the image into HSV color space.  Here is a representation of HSV color space courtesy of https://en.wikipedia.org/wiki/HSL_and_HSV (accessed 6/2/20):

![alt text][image4]

To extract the yellow pixels, I identified the pixels with values within these limits:
```python
yellow_lower = np.array([15, 70, 70])  #[Hue,Saturation,Value]
yellow_upper = np.array([40, 255, 255])
yellow = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
```
Then I identified the white pixels within these limits:
```python
white_lower = np.array([0,0,180])
white_upper = np.array([255,30,255])
white = cv2.inRange(hsv_image, white_lower, white_upper)
```
And I combined the two results into a binary image displaying only pixels representing the lane lines:
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used `find_lane_pixels()` to create a histogram of y values vs x values of the white pixels in the bottom half of the image.  This assumes that even if the road cuves, the lane lines in the bottom half of the image remain fairly straight.  So finding the two peaks of the histogram are a good approximation of the x coordinates of the starting points of the two lane lines.  Using these two x coordinates, I create a window around each at the bottom of the image and search for the local maximums, which I store as (x,y) points - one for each lane line.  Then I create another window just above each of the first two, centered on the x values.  Again, I use a histogram approach to find the local maximums and store a new (x,y) point for each lane line.  I continue this for the whole image and accumulate a set of (x,y) points representing each lane line.  Then I used the function `fit_polynomial()` to take those points and create a line of best fit using Numpy's `.polyfit()` function.  This gave me the coefficients of a 2nd order polynomial of the form `x=ay^2+by+c` for each lane line.  I used y as the independent variable to avoid situations of infinite slope when the lanes are straight.  Using these two equations, I plotted the lines of best-fit on the image as shown here:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature and lane position, I needed to convert pixels into a real-world unit.  I chose meters and used an approximate conversion factor.  I used the function `fit_polynomial_scaled()`, which takes the (x,y) coordinates of the two lane lines and again uses Numpy's `polyfit()` function to determine coefficients for a line of best-fit, but this time I used the conversion factor to scale the results to meters.  I used those coefficients and a scaled y value equal to the bottom of the image (road right in front of the car) to calculate the radius of curvature of a circle tangent to the lane line right in front of the car.  I then calculated the offset of the center of the car to the center of the road by comparing the midpoint of the camera image to the midpoint of the lane lines directly in front of the car.  Here is an image with the calculations displayed:

![alt text][image7]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To plot the results back down from bird's-eye perspective to a driver's perspective, I needed to again use OpenCV's `warpPerspective()`, except this time I used an inverse of the first matrix, which I obtained using Numpy's `.linalg.inv()` function.  Here is the result:

![alt text][image8]

The results of all the processed images are saved in the directory `/test_images_output`

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./test_videos_output/lanes_project_video.mp4)

Here's a [link to my result for the challenge video](./test_videos_output/lanes_challenge_video.mp4)

Both are saved in the directory `/test_videos_output`


#### 2. Use a Class data type to store frame results for validating/smoothing the results.

I created a Class data type called Lines() in which I stored results for frames where I successfully found lane lines.  This was a very helpful tool for debugging.  I also used it to compare the best-fit line coordinates of the current video frame to those preceding it.  Here is the class definition:
```python
class Line():
    def __init__(self):
        #choosing a number of iterations
        self.n = 20
        # was the line detected in the last iteration?
        self.detected = False
        # x values for the best-fit lines
        self.recent_xfitted = []
        #average x value of the best-fit lines over the last n iterations
        self.bestx = None
        #polynomial coefficients for the best-fit line
        self.recent_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = [0,0,0]#None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

    def add_recent_xfitted(self,new_recent_xfitted):
        #record the newest x points
        self.recent_xfitted.append(new_recent_xfitted)
        size = len(self.recent_xfitted)
        last_n_xfitted_means = []
        if ((size-self.n)<0): #account for there being less than n images so far
            num = size
        else:
            num = self.n
        #fill the array 'last_n_fitted_means' with the mean x value of the line in each image
        for fit in range((size-num),(size)):
            if self.recent_xfitted[fit][0] != 0:
                last_n_xfitted_means.append(int(statistics.mean(self.recent_xfitted[fit])))
        #find the median of the last n images' mean x values
        self.bestx = (int(statistics.median(last_n_xfitted_means)))


    def add_recent_fit(self,new_recent_fit):
        #record the newest polynomial coefficients
        self.recent_fit.append(new_recent_fit)
        size = len(self.recent_fit)
        coeffA = []
        coeffB = []
        coeffC = []
        if ((size-self.n)<0): #account for there being less than n images so far
            num = size
        else:
            num = self.n
        #fill arrays with the last n polynomial coefficients
        for fit in range((size-num),(size)):
            if ((self.recent_fit[fit][0]!=0) and (self.recent_fit[fit][1]!=0) and (self.recent_fit[fit][2]!=0)):
                coeffA.append(self.recent_fit[fit][0])
                coeffB.append(self.recent_fit[fit][1])
                coeffC.append(self.recent_fit[fit][2])
        #find the median of the last n polynomial coefficients
        self.best_fit[0]=statistics.median(coeffA)
        self.best_fit[1]=statistics.median(coeffB)
        self.best_fit[2]=statistics.median(coeffC)
        #find the difference between the newest polynomial coefficients and those of the previous image
        self.diffs[0] = abs(new_recent_fit[0]-self.recent_fit[size-2][0])
        self.diffs[1] = abs(new_recent_fit[1]-self.recent_fit[size-2][1])
        self.diffs[2] = abs(new_recent_fit[2]-self.recent_fit[size-2][2])
```
I used the stored data to smooth the results and discard outliers for each lane line.
If the coefficients for the current frame's line of best-fit were far from the median of those in the last n images, then I threw them out and replaced them with a compromise between the coefficients of the accumulated mean and the coefficients of the last good frame.  Then I marked that frame as bad and did not save the coefficients.
If the coefficients for the current frame's line of best-fit were close to the median of those in the last n image but still a little too far off, then I replaced them with a compromise between those coefficients and the median of those in the last n images.  This helped to smooth the results.
Here is a section of the `fit_polynomial()` function that handles the left lane line:

```python
if ((ftype == 'Video') and (frameNum > leftStats.n) and (leftStats.detected == True)):
        numFitL=len(leftStats.recent_fit)
        if ((abs(leftStats.recent_fit[numFitL-1][0]-left_fit[0])> (0.0005*4)) or (abs(leftStats.recent_fit[numFitL-1][1]-left_fit[1])> (0.5*4)) or (abs(leftStats.recent_fit[numFitL-1][2]-left_fit[2])>(40*4))):
            leftStats.detected = False
            left_fit=((leftStats.best_fit+leftStats.recent_fit[numFitL-1])/2)
        elif ((abs(leftStats.recent_fit[numFitL-1][0]-left_fit[0])> (0.0005)) or (abs(leftStats.recent_fit[numFitL-1][1]-left_fit[1])> (0.5)) or (abs(leftStats.recent_fit[numFitL-1][2]-left_fit[2])>(40))):
            leftStats.detected = True
            left_fit[0] = (leftStats.best_fit[0]*2+left_fit[0])/3 #if out of range, replace with average coefficients of recent frames
            left_fit[1] = (leftStats.best_fit[1]*2+left_fit[1])/3
            left_fit[2] = (leftStats.best_fit[2]*2+left_fit[2])/3
```
#### 3. Use a more targeted approach to finding lane pixels.

Initially, I used the `find_lane_pixels()` function described above to get the coordinates of pixels representing the lane lines.  This utilizes a fresh histogram search each time.
Once I determined that a video frame contains a good line of best-fit, I substituted the function `search_around_poly()` in the next frame, which quickly and accurately targets its search area around the location of the line of best-fit in the previous frame.  I use the Class data type to retain the best-fit coefficients of the previous frame for comparison here.  If I later determine that the line found in this frame is an outlier, then I revert back to the `find_lane_pixels()` function for the next video frame.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This code does a reasonable job of smoothing over or rejecting outliers, but if there is a long section of video with undetectable lane lines, and the road curves or car position changes greatly at the same time during that section, then the smoothing and rejecting comparison will not accurately reflect the true state of the road - rather it will retain its comparison to the data before the undetectable lines and before the change in curve/position. Then the code will improperly compare current frames to the true data, resulting in outputs that are smoothed more towards the data before the udetectable line and before the change in curve/position.  I could improve this by creating a second Class instance for each lane line and use the second one to constantly record data (not just for frames determined "good") and if I find consistency here during frames directly following smoothed/discarded data, then I can use it to more quickly return to the true state of the lane lines.

This code only works well when the curves are gradual enough that they stay inside the defined region of interest.  If needed, the region of interest could be redefined to accomodate sharper curves.

This code assumes the lanes follow a 2nd degree polynomial, which may not be the most accurate in all curves.

### Acknowledgement

I would like to especially thank all the contributors to:
- [docs.python.org](http://docs.python.org)
- [geeksforgeeks.org](https://www.geeksforgeeks.org)
- [stackoverflow.com](https://www.stackoverflow.com)

Without you guys, I would still be scratching my head in debug.
