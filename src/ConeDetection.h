#ifndef CONEDETECTION_H_
#define CONEDETECTION_H_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <chrono>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace std;
using namespace cuda;


/*------------MAIN FUNCTION FOR THE CONES DETECTION-------------*/
/*
It makes:
    - a color conversion from RGB to HSV to perform a color threshold that isolate 
        the objects, in the image, that has the same color of the cones;
    - a bitwise-and between the depth image and the thresholded one to consider only
        the objects in a renge of distance between 1.3 and 10 meters;
    - when it had isolate the objects, it find the contours of the shapes in the image
        and make them convex discarding the shapes with less than 3 vertices;
    - the detected cones is made of two blobs (divided by the white line). For this reason,
        the final step of the detection is join the contour. This joining is not arbitraty
        but is performed taking into account some shape relation that a cone must have.
        The detected cones are only the shapes that was joined.
*/
void ConeDetection(Mat *image, Mat *depth, vector<Rect> *bounding_rects, vector<float> *distances);




/*-------------------FUNCTION DECLARATIONS----------------------*/

// FUNCTION TO JOIN TWO PARTS OF A CONE DIVIDED BY THE WHITE LINE
/*
Performed controls of the cone shape:
    - the aspect ratio (h/w) of a joined shape must be in an imposed range;
    - the x-distance between two blobs must be less than half width of the bottom bolb.
*/
void coneJoin(vector<vector<Point>> *contours, vector<Rect> *bounding_rects, vector<bool> *join, int steps);


// ADAPT THE DEPTH FRAME TO THE RGB ONE

void maskFrame(Mat* depth);


// ESTIMATES THE DISTANCE OF THE CONE
/*
The depth computation is performed arround the centroid of the cones making 
the average in a window of some pixels area.
*/
float depth_computation(Rect* bounding_rect, Mat* depth_img);


// TAKES AS INPUT A FLOAT AND TRANSFORM IT IN A STRING (TRONCATING IT)
/*
This function is performed to round the value of the distance and show it in the final video
*/
string troncf2str(float num_in, int decimals);


// COMPUTE THE "inRange" FUNCTION USING GPU

void inRangeCuda(cv::cuda::GpuMat* InputArray, Scalar hsv_low, Scalar hsv_high, cv::cuda::GpuMat* OutputArray);

#endif