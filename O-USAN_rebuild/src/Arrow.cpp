//
// Created by insppp on 3/28/22.
//
#include "Arrow.h"

Arrow::Arrow(double dx, double dy, double strength, int i, int j) {
    this->dp = std::make_pair(dx, dy);
    this->strength = strength;
    this->p = cv::Point2i (j, i);
}

void Arrow::draw(cv::Mat src) const {
    int x = p.x+5 * dp.first;
    if (x <5 || x>=src.cols-5) {
        return;
    }
    int y = p.y+5 * dp.second;
    if (y <5 || y>=src.rows-5) {
        return;
    }
    cv::Point2i p1 = cv::Point2i(p.x+5 * dp.first, p.y+5 * dp.second);
    cv::arrowedLine(src, p, p1, cv::Scalar(0, 0, 255));
}