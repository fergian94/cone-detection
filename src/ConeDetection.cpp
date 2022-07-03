#include "./ConeDetection.h"

#define TestCUDA false

#define HULL_MIN_SIZE 3
#define JOIN_ITER_N 2
#define MIN_ASP_RATIO 0.5f
#define MAX_ASP_RATIO 1.8f
#define UPPER_SAT 10.0
#define LOWER_SAT 1.3
#define DEPTH_KER 5

Size imgSize = Size(1280,720);



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
void ConeDetection(Mat *image, Mat *depth, vector<Rect> *bounding_rects, vector<float> *distances)
{
    Mat image_1, image_2, image_3, image_4;
    cv::cuda::GpuMat GpuImage_1, GpuImage_2, GpuImage_3;
    vector<vector<Point>> contours, convex_hulls;
    int i, k, n;



    ////////////////////////////////////////////////////////////////
    // CONVERT THE IMAGE IN HSV

    if (TestCUDA) {

        GpuImage_1.upload(*image);
        cv::cuda::cvtColor(GpuImage_1, GpuImage_2, COLOR_BGR2HSV);
        GpuImage_2.download(image_1);

    } else {

        cv::cvtColor(*image, image_1, COLOR_BGR2HSV);

    }



    ////////////////////////////////////////////////////////////////
    // COMPUTE THE COLOR THRESHOLDS

    if (TestCUDA) {

        // RED CONES
        inRangeCuda(&GpuImage_2, Scalar(0,135,135)  , Scalar(15,255,255) , &GpuImage_1);
        inRangeCuda(&GpuImage_2, Scalar(159,135,135), Scalar(179,255,255), &GpuImage_3);
        // YELLOW CONES
        //inRangeCuda(&GpuImage_2, Scalar(21,60,60) , Scalar(30,255,255) , &GpuImage_1);
        // BLUE CONES
        //inRangeCuda(&GpuImage_2, Scalar(106,57,40), Scalar(121,255,255), &GpuImage_3);

        cv::cuda::bitwise_or(GpuImage_1, GpuImage_3, GpuImage_1);
        // GpuImage_1.download(image_2);

    } else {

        // RED CONES
        inRange(image_1, Scalar(0,135,135), Scalar(15,255,255), image_2);
        inRange(image_1, Scalar(159,135,135), Scalar(179,255,255), image_3);
        // YELLOW CONES
        //inRange(image_1, Scalar(21,60,60), Scalar(30,255,255), image_2);
        // BLUE CONES
        //inRange(image_1, Scalar(106,57,40), Scalar(121,255,255), image_3);

        cv::bitwise_or(image_2, image_3, image_2);

    }



    ////////////////////////////////////////////////////////////////
    // APPLY DEPTH MASK

    maskFrame(depth);

    if (TestCUDA) {

        GpuImage_3.upload(*depth);
        cv::cuda::threshold(GpuImage_3, GpuImage_3, 1, 255, cv::THRESH_BINARY);
        cv::cuda::bitwise_and(GpuImage_1, GpuImage_3, GpuImage_1);
        GpuImage_1.download(image_2);

    } else {

        cv::threshold(*depth,image_3, 1, 255, cv::THRESH_BINARY);
        cv::bitwise_and(image_2, image_3, image_2);

    }


   
    ////////////////////////////////////////////////////////////////
    // FIND THE EDGES OF THE FIGURE
    
    cv::findContours(image_2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);



    ////////////////////////////////////////////////////////////////
    // MAKE CONVEX HULLS

    k = 0;
    vector<Point> temp_vec;
    Rect temp_rec;
    n = contours.size();
    for (i=0; i<n; i++)
    {
        convexHull(contours[i], temp_vec);
        temp_rec = boundingRect(temp_vec);
        if ( temp_vec.size()>HULL_MIN_SIZE )
        {
            convex_hulls.push_back(temp_vec);
            (*bounding_rects).push_back(temp_rec);
            k++;
        }
    }



    ////////////////////////////////////////////////////////////////
    // JOIN THE CONES DIVIDED BY THE WHITE-LINE

    vector<bool> join(convex_hulls.size(), false);
    coneJoin(&convex_hulls, bounding_rects, &join, JOIN_ITER_N);



    ////////////////////////////////////////////////////////////////
    // CHECK AND SAVE

    n = convex_hulls.size();
    (*bounding_rects).clear();

    for(i=0; i<n; i++)
    {
        if( join[i] )
        {
            (*bounding_rects).push_back(boundingRect(convex_hulls[i]));
            (*distances).push_back(depth_computation(&(*bounding_rects).back(), depth));
        }
    }

}







/*-------------------FUNCTION DECLARATIONS----------------------*/

// FUNCTION TO JOIN TWO PARTS OF A CONE DIVIDED BY THE WHITE LINE
/*
Performed controls of the cone shape:
    - the aspect ratio (h/w) of a joined shape must be in an imposed range;
    - the x-distance between two blobs must be less than half width of the bottom bolb.
*/
void coneJoin(vector<vector<Point>> *contours, vector<Rect> *bounding_rects, vector<bool> *join, int steps)
{

    int i, j, k;
    Rect P1, P2;
    vector<Point> temp;
    float DY, centerDX;
    float ar_j;
    int B;
    
    for (i=0; i<steps; i++)
    {
        j = 0;
        while (j<(*contours).size())
        {
            k = j + 1;
            while (k<(*contours).size())
            {
                P1 = (*bounding_rects)[j];
                P2 = (*bounding_rects)[k];
                DY = abs(P1.y - P2.y);
                centerDX = abs(P1.x + P1.width / 2 - P2.x - P2.width / 2);
                
                if (P1.y > P2.y){
                    B = P1.width;
                    DY += P1.height;
                    ar_j = DY/B;
                } else {
                    B = P2.width;
                    DY += P2.height;
                    ar_j = DY/B;
                }
                
                if ((ar_j > MIN_ASP_RATIO) && (ar_j < MAX_ASP_RATIO) && (centerDX < B/2))
                {
                    (*contours)[j].insert((*contours)[j].end(), (*contours)[k].begin(), (*contours)[k].end());
                    convexHull((*contours)[j], temp);
                    (*contours)[j] = temp;
                    (*join)[j] = true;
                    (*bounding_rects)[j] = boundingRect((*contours)[j]);
                    (*contours).erase((*contours).begin() + k);
                    (*bounding_rects).erase((*bounding_rects).begin() + k);
                    (*join).erase((*join).begin() + k);
                }
                k++;
            }
            j++;
        }
    }

}



// ADAPT THE DEPTH FRAME TO THE RGB ONE

void maskFrame(Mat* depth)
{

    Rect crop_region(203, 103, 883, 496);
    *depth = (*depth)(crop_region);

    cv::resize(*depth, *depth, imgSize, 0.0, 0.0, cv::INTER_NEAREST);
    
}



// ESTIMATES THE DISTANCE OF THE CONE
/*
The depth computation is performed arround the centroid of the cones making 
the average in a window of some pixels area.
*/
float depth_computation(Rect* bounding_rect, Mat* depth_img)
{

    int x,y;
    float dist = 0;

    x = (*bounding_rect).x + (*bounding_rect).width / 2;
    y = (*bounding_rect).y + (*bounding_rect).height / 2;
    for (int j = 0; j < DEPTH_KER; j++)
        for (int k = 0; k < DEPTH_KER; k++)
            dist += (*depth_img).at<uchar>((y-DEPTH_KER/2)+k, (x-DEPTH_KER/2)+j);
        
    dist = dist/(DEPTH_KER * DEPTH_KER);

    dist = ( 255 - dist ) / 255.0f * (UPPER_SAT - LOWER_SAT) + LOWER_SAT;

    return dist;

}



// TAKES AS INPUT A FLOAT AND TRANSFORM IT IN A STRING (TRONCATING IT)
/*
This function is performed to round the value of the distance and show it in the final video
*/
string troncf2str(float num_in, int decimals)
{
    
    string num_str = to_string(num_in);
    string num_out = "";
    bool flag = false;
    int j = 0;

    for(int i = 0; i < num_str.length(); i++){
        if (num_str[i] == '.'){
            flag = true;
            num_out.push_back(num_str[i]);
        } else {
            if (flag){
                num_out.push_back(num_str[i]);
                j ++;
                if (j >= decimals)
                    break;
            } else {
                num_out.push_back(num_str[i]);
            }
        }
    }

    return num_out;

}



// COMPUTE THE "inRange" FUNCTION USING GPU

void inRangeCuda(cv::cuda::GpuMat* InputArray, Scalar hsv_low, Scalar hsv_high, cv::cuda::GpuMat* OutputArray)
{

    cv::cuda::GpuMat tmp;
    cv::cuda::GpuMat mat_parts[3];
    cv::cuda::GpuMat mat_parts_low[3];
    cv::cuda::GpuMat mat_parts_high[3];

    cv::cuda::split(*InputArray, mat_parts);

    cv::cuda::threshold(mat_parts[0], mat_parts_low[0], hsv_low[0], 255, cv::THRESH_BINARY);
    cv::cuda::threshold(mat_parts[0], mat_parts_high[0],  hsv_high[0], 255, cv::THRESH_BINARY_INV);
    cv::cuda::bitwise_and(mat_parts_high[0], mat_parts_low[0], mat_parts[0]);

    cv::cuda::threshold(mat_parts[1], mat_parts_low[1], hsv_low[1], 255, cv::THRESH_BINARY);
    cv::cuda::threshold(mat_parts[1], mat_parts_high[1],  hsv_high[1], 255, cv::THRESH_BINARY_INV);
    cv::cuda::bitwise_and(mat_parts_high[1], mat_parts_low[1], mat_parts[1]);

    cv::cuda::threshold(mat_parts[2], mat_parts_low[2], hsv_low[2], 255, cv::THRESH_BINARY);
    cv::cuda::threshold(mat_parts[2], mat_parts_high[2],  hsv_high[2], 255, cv::THRESH_BINARY_INV);
    cv::cuda::bitwise_and(mat_parts_high[2], mat_parts_low[2], mat_parts[2]);

    cv::cuda::bitwise_and(mat_parts[0], mat_parts[1], tmp);
    cv::cuda::bitwise_and(tmp, mat_parts[2], *OutputArray);

}