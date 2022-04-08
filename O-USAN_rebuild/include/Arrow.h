//
// Created by insppp on 3/28/22.
//

#ifndef DEMO_ARROW_H
#define DEMO_ARROW_H

#include "opencv2/opencv.hpp"

class Arrow {
public:
    double strength;
    cv::Point2i p;
    std::pair<double, double> dp;
    Arrow(double dx,double dy, double strength, int i, int j);
    void draw(cv::Mat src) const;
};

#endif //DEMO_ARROW_H
