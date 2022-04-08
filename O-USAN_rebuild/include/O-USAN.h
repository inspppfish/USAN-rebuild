//
// Created by insppp on 3/23/22.
//
#ifndef O_USAN_REBUILD_O_USAN_H
#define O_USAN_REBUILD_O_USAN_H

#include <iostream>
#include <cstdint>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "kernel_gen.h"
#include "Arrow.h"

//---------------config-----------------//

#define OUSAN_DETECT_NON_MAXIMUM_SUPPRESSION 0x0002
/* configure sample
    const uint32_t config = (
        KERNEL_STRUCTOR_PRINT_ENABLE |
        OUSAN_DETECT_NON_MAXIMUM_SUPPRESSION
    )
*/

//--------------#config#----------------//


//---------------core/kernel/OUSAN-----------------//
class Ousan_kernel : public Kernel{
public:
    using Kernel::Kernel;
    int direct_area; // area of each direction (no center)
    void kernelStructor() override; // builder implement
    int direct[2][9][100]; // pos of the pixel (related to the center of kernel) of each direction
};
//--------------#core/kernel/OUSAN#----------------//


//---------------core/detector-----------------//
class Ousan_detector {
private:
    Ousan_kernel * kernel;
    uint32_t config;
    int oSTU(cv::Mat roi);
    void binary (cv::Mat& roi, int t);
    std::pair<double, double> kernelJudge(cv::Mat roi, double boundFNT, double boundBGN,double boundD);
    static int getColorDistance(cv::Vec3b color1, cv::Vec3b color2);
    bool judgeByDirections(cv::Mat roi, int threshold_v, int og);
public:
    std::vector<Arrow> * detect(cv::Mat src, double boundFNT, double boundBGN,double boundD, int threshold_v, int og);
    Ousan_detector(Ousan_kernel * knl, uint32_t config);
};
//--------------#core/detector#----------------//





#endif //O_USAN_REBUILD_O_USAN_H
