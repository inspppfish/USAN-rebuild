#include "O-USAN.h"
#include <string>
using namespace cv;
using namespace std;

uint32_t config = (
        KERNEL_STRUCTOR_PRINT_ENABLE
//         | OUSAN_DETECT_NON_MAXIMUM_SUPPRESSION
);


Ousan_detector * ousan_run(int inputDiameter, uint32_t config) {
    auto ousanKernel = new Ousan_kernel(9, config);
    auto ousanDetector =  new Ousan_detector(ousanKernel, config);
    ousanKernel->kernelStructor();
    return ousanDetector;
}


int main (int argc,char *argv[]) {
    string fileName;
    if (argc == 2) {
        fileName = argv[1];
    }else {
        printf("input ONE param as filename");
        return 0;
    }
    Mat src = imread(fileName, IMREAD_COLOR);
    printf("rows : %d , cols : %d\n", src.rows, src.cols);
    double boundFNT = 0.20;
    double boundBGN = 0.20;
    double boundD = 0.4;
    auto arrows = ousan_run(9, config)->detect(src, boundFNT, boundBGN, boundD, 100, 0);

    for (auto & arrow : *arrows) {
        arrow.draw(src);
    }

    imshow("Detected", src);
    imwrite(fileName+".png", src);
    cvWaitKey(0);
    return 0;
}

