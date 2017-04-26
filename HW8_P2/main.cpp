#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <fstream>
#include <string>
#include <iomanip>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat image;
    Mat image_color;
    Mat image_prev;
    string filename;
    string header;
    string tail;
    Size winsize = Size(30,30);
    vector<Point2f> corners, corners_prev, corners_undistort, corners_prev_undistort;
    vector<uchar> status;
    vector<float> err;
    Mat a;
    float tau;
    Mat tau_matrix(19,2,CV_32F);
    char number_str [6];
    Mat ransac_num;
    vector<int> remove_tracker;
    int reject_count = 0;
    vector<int> keep_tracker;
    int keep_count = 0;
    vector<int> condition_tracker;
    Mat H1, H2;
    Mat F;
    Mat t_scaled = (Mat_<double>(3,1) << 0, 0, 0);
    Mat C_k, C_k_prev, T_k, T_k_prev;
    C_k = (Mat_<double>(4,4) << 1, 0, 0, 0,
           0, 1, 0, 0,
           0, 0, 1, 0,
           0, 0, 0, 1);
    float minEigThreshold = 1e-3;

    ofstream fileout;
    fileout.open("/home/dallin/robotic_vision/HW8/VO_Practice_Sequence/vo_estimate_hall.txt");

    //    header = "/home/dallin/robotic_vision/HW8/VO_Practice_Sequence/VO_Practice_Sequence/";
    header = "/home/dallin/robotic_vision/HW8/BYU_Hallway_Sequence/BYU_Hallway_Sequence/";
    tail = ".png";


    for (int i = 0; i < 2241; i+=1)
    {
        sprintf(number_str, "%06d", i);

        filename = header + number_str + tail;
        if(image.empty())
        {
            image = Mat::zeros(480,640,CV_32F);
        }

        image_prev = image.clone();
        image = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
        image_color = imread(filename,CV_LOAD_IMAGE_COLOR);

        goodFeaturesToTrack(image_prev,corners_prev,500,.01,10,noArray(),3,false,.04);

        if(corners_prev.empty())
        {
            cout<< "No corners!" << endl;
        }
        else
        {
            calcOpticalFlowPyrLK(image_prev,image,corners_prev,corners,status,err,winsize,5,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,30,.01),OPTFLOW_LK_GET_MIN_EIGENVALS,minEigThreshold);

            for(int i = corners.size() - 1; i >= 0; i--)
            {
                if(!status[i] || err[i] < minEigThreshold)
                {
                    corners.erase(corners.begin() + i);
                    corners_prev.erase(corners_prev.begin() + i);
                }
            }

            if(corners_prev.size() > 20)
            {

                F = findFundamentalMat(corners_prev,corners,ransac_num,FM_RANSAC,.01,.999);

                for(int i = 0; i<corners.size(); i++)
                {
                    condition_tracker.push_back(1); //All points are marked initially as valid and current.
                }

                for(int i = 0; i<ransac_num.rows; i++)
                {
                    condition_tracker[i] = ransac_num.at<bool>(0,i);
                }

                for(int i = 0; i<condition_tracker.size(); i++)
                {
                    if (condition_tracker[i] == 0)
                    {
                        //Remove point from array
                        corners_prev.erase(corners_prev.begin() + i);
                        corners.erase(corners.begin() + i);
                        condition_tracker.erase(condition_tracker.begin() + i);
                        remove_tracker.push_back(i + reject_count); //Keep track of the points that were removed
                        reject_count = reject_count +1; //Since we're adjusting i, we need to add this in to get original location.
                        //                    cout << "edge case" << endl;
                        i = i-1;
                    }

                }


                //                stereoRectifyUncalibrated(corners_prev,corners,F,Size(640,480),H1,H2,0);

                Mat M1 = (Mat_<double>(3,3) <<  6.7741251774486568e+02, 0.0000000000000000e+00, 3.2312557438767283e+02,
                          0.0000000000000000e+00, 6.8073800850564749e+02, 2.2477413395670021e+02,
                          0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00);

                Mat M2 = M1;
                undistortPoints(corners_prev,corners_prev_undistort,M1,noArray(),noArray(),M1);
                undistortPoints(corners,corners_undistort,M1,noArray(),noArray(),M1);

                //            Mat R1 = M1.inv(DECOMP_LU)*H1*M1;
                //            Mat R2 = M2.inv(DECOMP_LU)*H2*M2;

                Mat M2_T;
                transpose(M2,M2_T);
                Mat E = M2_T*F*M1;

                Mat w,u,vt;
                SVD::compute(E,w,u,vt,0);
                w.at<double>(0) = 1;
                w.at<double>(1) = 1;
                w.at<double>(2) = 0;
                E = u*Mat::diag(w)*vt;

                Mat R, t;

                recoverPose(E,corners_prev_undistort,corners_undistort,R,t,1.0,Point2d(0,0),noArray());

                if(t.at<double>(2) > 0)
                {
                    t = -1*t;
                }


                t_scaled = t*.8;


                C_k_prev = C_k;
                T_k = (Mat_<double>(4,4) << R.at<double>(0), R.at<double>(1), R.at<double>(2), t_scaled.at<double>(0),
                       R.at<double>(3), R.at<double>(4), R.at<double>(5), t_scaled.at<double>(1),
                       R.at<double>(6), R.at<double>(7), R.at<double>(8), t_scaled.at<double>(2),
                       0, 0, 0, 1);

                C_k = C_k_prev*T_k.inv(DECOMP_LU);

                fileout << C_k.at<double>(0) << "," << C_k.at<double>(1) << "," << C_k.at<double>(2) << "," << C_k.at<double>(3)
                        << "," << C_k.at<double>(4) << ","<< C_k.at<double>(5) << ","<< C_k.at<double>(6) << ","
                        << C_k.at<double>(7) << "," << C_k.at<double>(8) << "," << C_k.at<double>(9) << "," << C_k.at<double>(10) << ","
                        << C_k.at<double>(11) << "," << endl;


                for(int i=0; i<corners_prev.size(); i++)
                {
                    circle(image_color,corners_prev[i],2,Scalar(0,255,0),2,LINE_8,0);
                }
                for(int i=0; i<corners_prev.size(); i++)
                {
                    line(image_color,corners[i],corners_prev[i],Scalar(0,0,255),2,LINE_8,0);
                }
            }
            else
            {
                C_k = C_k_prev;
                fileout << C_k.at<double>(0) << "," << C_k.at<double>(1) << "," << C_k.at<double>(2) << "," << C_k.at<double>(3)
                        << "," << C_k.at<double>(4) << ","<< C_k.at<double>(5) << ","<< C_k.at<double>(6) << ","
                        << C_k.at<double>(7) << "," << C_k.at<double>(8) << "," << C_k.at<double>(9) << "," << C_k.at<double>(10) << ","
                        << C_k.at<double>(11) << "," << endl;
            }

            condition_tracker.clear();
            remove_tracker.clear();
            reject_count = 0;
        }

        imshow("Image Color", image_color);
        waitKey(1);

    }
    fileout.close();
    waitKey(0);

    return 0;
}
