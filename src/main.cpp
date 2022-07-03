
// Import the settings from "ConeDetection.h" file
#include "./ConeDetection.h"


int main(int argc, char const *argv[]){

    Mat frame, frame_out, depth;
    vector<Rect> bounding_rects;
    vector<float> distances;
    string filename; 
    string val;
    float round_val;

    // Read the video sequence in RGB
    VideoCapture cap("../data/stream_hd.mp4");

    if (!cap.isOpened()){
        return 0;
    }

    // Resize the output window to see better the output video on the screen
    namedWindow("detected", WINDOW_KEEPRATIO);
    resizeWindow("detected", 960,540);

    for (int i=0; i<2208-308; i++)
    {
        cap >> frame;

        if (i>=1392)
        {
            // Read the depth frame from Frames folder (already transformed in JPG)
            filename = "../data/Frames/img_" + to_string(i+308) + ".jpg";
            depth = imread(filename,0);

            // Call of the "ConeDetection" function: the main function for the detection of the cones
            ConeDetection(&frame, &depth, &bounding_rects, &distances);
            
            // Output frame building drawing the rectangles 
            for (int j=0; j<bounding_rects.size(); j++){
                rectangle(frame,bounding_rects[j],Scalar(0,255,0),2,LINE_AA);
                
                val = troncf2str(distances[j],2);

                putText(frame, val, Point((bounding_rects[j].x), (bounding_rects[j].y)-10),  
                        FONT_HERSHEY_SIMPLEX, 0.6f, Scalar(0, 255, 0),1);
            }
        
            imshow("detected",frame);
            waitKey(5);
            if(i==1392)
                waitKey(0);

            bounding_rects.clear();
            
        }

    }

    return 0;
}
