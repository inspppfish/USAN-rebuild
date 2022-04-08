//
// Created by insppp on 3/23/22.
//
#include "O-USAN.h"
#include "cstring"
#include "cmath"


//---------------core/kernel/OUSAN/builder-----------------//
void Ousan_kernel::kernelStructor() {

    // iterator of direction area pixels
    int dir_area_count[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = -1 * (this->diameter / 2); i <= this->diameter / 2; i++) {
        for (int j = -1 * (this->diameter / 2); j <= this->diameter / 2; j++) {

            // include the pixel by distance from center
            if ((i * i + j * j) * 4 < this->diameter * this->diameter && !(i == 0 && j == 0)/* no center*/) {

                if (config & KERNEL_STRUCTOR_PRINT_ENABLE) {
                    printf(" 0");
                }

                this->grp[0].push_back(i);
                this->grp[1].push_back(j);

                if (i >= 0 && j >= 0) {
                    if (i >= j) {
                        // in the area 1
                        this->direct[0][1][++dir_area_count[1]] = i;
                        this->direct[1][1][dir_area_count[1]] = j;
                    }
                    if (i <= j) {
                        // in the area 2
                        this->direct[0][2][++dir_area_count[2]] = i;
                        this->direct[1][2][dir_area_count[2]] = j;
                    }
                }
                if (i <= 0 && j >= 0) {
                    if (-1 * i <= j) {
                        // in the area 3
                        this->direct[0][3][++dir_area_count[3]] = i;
                        this->direct[1][3][dir_area_count[3]] = j;
                    }
                    if (-1 * i >= j) {
                        // in the area 4
                        this->direct[0][4][++dir_area_count[4]] = i;
                        this->direct[1][4][dir_area_count[4]] = j;
                    }
                }
                if (i <= 0 && j <= 0) {
                    if (-1 * i >= -1 * j) {
                        // in the area 5
                        this->direct[0][5][++dir_area_count[5]] = i;
                        this->direct[1][5][dir_area_count[5]] = j;
                    }
                    if (-1 * i <= -1 * j) {
                        // in the area 6
                        this->direct[0][6][++dir_area_count[6]] = i;
                        this->direct[1][6][dir_area_count[6]] = j;
                    }
                }
                if (i >= 0 && j <= 0) {
                    if (i <= -1 * j) {
                        // in the area 7
                        this->direct[0][7][++dir_area_count[7]] = i;
                        this->direct[1][7][dir_area_count[7]] = j;
                    }
                    if (i >= -1 * j) {
                        // in the area 8
                        this->direct[0][8][++dir_area_count[8]] = i;
                        this->direct[1][8][dir_area_count[8]] = j;
                    }
                }
            } else {
                if (config & KERNEL_STRUCTOR_PRINT_ENABLE)
                    printf("  ");
            }
        }
        if (config & KERNEL_STRUCTOR_PRINT_ENABLE)
            printf("\n");
    }
    this->area = (int) this->grp[0].size();
    this->direct_area = dir_area_count[1];
    if (config & KERNEL_STRUCTOR_PRINT_ENABLE) {
        printf("Area : %d (Non center) \n", area);
        printf("Direct_Area : %d\n", direct_area);
    }
}
//--------------#core/kernel/OUSAN/builder#----------------//


//---------------core/detector/init-----------------//
Ousan_detector::Ousan_detector(Ousan_kernel *knl, uint32_t inputConfig) {
    kernel = knl;
    config = inputConfig;
}
//--------------#core/detector/init#----------------//


//---------------core/detector/detect-----------------//
double matStrength[3000][3000][2];
std::vector<Arrow> *
Ousan_detector::detect(cv::Mat src, double boundFNT, double boundBGN, double boundD, int threshold_v = 0, int og = 0) {

    memset(matStrength, 0, sizeof(matStrength));
    cv::Mat proc;
    cvtColor(src, proc, cv::COLOR_Lab2BGR);

    int imax = src.rows - (kernel->diameter / 2);
    int jmax = src.cols - (kernel->diameter / 2);

    for (int i = kernel->diameter / 2; i < imax-2; i++) {
        for (int j = kernel->diameter / 2; j < jmax-2; j++) {
            int same = 0;

            cv::Rect rec(j - kernel->diameter / 2, i - kernel->diameter / 2, kernel->diameter, kernel->diameter);
            cv::Mat roi;
            proc(rec).copyTo(roi);
            //ch0 = distance from (i, j)

            //getDistance
            for (int k = 0; k < kernel->area; k++) {
                int ii, jj;
                ii = jj = kernel->diameter / 2;
                int ni = ii + kernel->grp[0][k];
                int nj = jj + kernel->grp[1][k];
                roi.at<cv::Vec3b>(ni, nj)[0] = getColorDistance
                        (src.at<cv::Vec3b>(i, j), src.at<cv::Vec3b>(i + kernel->grp[0][k], j + kernel->grp[1][k]));
            }

            //OSTU//
            int t = oSTU(roi);

            //Barbarization :: light the BACGround//
            binary(roi, t);

            //judge the kernel//
            std::pair<double, double> ans1 = kernelJudge(roi, boundFNT, boundBGN, boundD);
            bool flag2 = judgeByDirections(roi, threshold_v, og);

            //it's tested that judgeByDirections has few effect on results :-(

            // accepted
            if (!(abs(ans1.first) < 0.001 && abs(ans1.second) < 0.001)) {
                matStrength[i][j][0] = ans1.first;
                matStrength[i][j][1] = ans1.second;
            }
        }

    }

    auto ans = new std::vector<Arrow>();

    if (config & OUSAN_DETECT_NON_MAXIMUM_SUPPRESSION) {
        for (int i = kernel->diameter / 2; i < src.cols - (kernel->diameter / 2) - 1; i++) {
            for (int j = kernel->diameter / 2; j < src.rows - (kernel->diameter / 2) - 1; j++) {
                double maxD = 0;
                for (int k = 0; k<kernel->area; k++) {
                    int ni = i + kernel->grp[0][k];
                    int nj = j + kernel->grp[1][k];
                    double x = matStrength[ni][nj][0];
                    double y = matStrength[ni][nj][1];

                    maxD = maxD > x * x + y * y ? maxD : x * x + y * y;
                }
                double x = matStrength[i][j][0];
                double y = matStrength[i][j][1];
                if (x * x + y * y > maxD) {
                    ans->push_back(Arrow(x, y, x*x + y*y, i, j));
                }
            }
        }
    }
    else {
        for (int i = kernel->diameter / 2; i < src.cols - (kernel->diameter / 2) - 1; i++) {
            for (int j = kernel->diameter / 2; j < src.rows - (kernel->diameter / 2) - 1; j++) {
                double x = matStrength[i][j][0];
                double y = matStrength[i][j][1];
                if (!(abs(matStrength[i][j][0]) < 0.001 && abs(matStrength[i][j][1]) < 0.001)) {
                    ans->push_back(Arrow(x, y, x * x + y * y, i, j));
                }
            }
        }
    }
    return ans;
}
//--------------#core/detector/detect#----------------//


//--------------core/detector/OSTU--------------------//
int Ousan_detector::oSTU(cv::Mat roi) {
    //411 = 255*sqrt(3)
    int ans = 1;
    int maxG = -999999999;
    for (int t = 1; t <= 411; t++) {
        int w0 = 0;
        int u0 = 0;
        int w1 = 0;
        int u1 = 0;
        for (int k = 0; k < kernel->area; k++) {
            int ii, jj;
            ii = jj = kernel->diameter / 2;
            int ni = ii + kernel->grp[0][k];
            int nj = jj + kernel->grp[1][k];

            if (roi.at<cv::Vec3b>(ni, nj)[0] <= t) {
                w1 += 1;
                u1 += roi.at<cv::Vec3b>(ni, nj)[0];
            } else {
                w0 += 1;
                u0 += roi.at<cv::Vec3b>(ni, nj)[0];
            }
        }
        int g = w0 * w1 * (u1 - u0) * (u1 - u0);
        if (g > maxG) {
            maxG = g;
            ans = t;
        }
    }
    return ans;
}
//-------------#core/detector/OSTU--------------------//


//-------------core/detector/binary-------------------//
void Ousan_detector::binary(cv::Mat &roi, int t) {
    for (int k = 0; k < kernel->area; k++) {
        int ii, jj;
        ii = jj = kernel->diameter / 2;
        int ni = ii + kernel->grp[0][k];
        int nj = jj + kernel->grp[1][k];

        roi.at<cv::Vec3b>(ni, nj)[0] = roi.at<cv::Vec3b>(ni, nj)[0] <= t ? 0 : 1;
    }
}
//------------#core/detector/binary-------------------//


//------------core/detector/judge---------------------//
std::pair<double, double> Ousan_detector::kernelJudge(cv::Mat roi, double boundFNT, double boundBGN, double boundD) {
    double n = 0;
    double ci = 0;
    double cj = 0;
    double cii = 0;
    double cjj = 0;
    for (int k = 0; k < kernel->area; k++) {
        int ii, jj;
        ii = jj = kernel->diameter / 2;
        int di = kernel->grp[0][k];
        int dj = kernel->grp[1][k];
        int ni = ii + di;
        int nj = jj + dj;

        if (roi.at<cv::Vec3b>(ni, nj)[0]) {
            n += 1;
            ci += 1.0 * di;
            cj += 1.0 * dj;
        } else {
            cii += 1.0 * di;
            cjj += 1.0 * dj;
        }
    }

    ci = ci / n;
    cj = cj / n;
    cii = cii / (1.0 * (kernel->area - n));
    cjj = cjj / (1.0 * (kernel->area - n));
    n = n / (1.0 * kernel->area);

    if (
            n > boundBGN
            &&
            (1 - n) > boundFNT
            &&
            (
                    (
                            sqrt(
                                    pow(ci, 2) +
                                    pow(cj, 2)
                            ) > 1.0 * boundD * kernel->diameter / 2
                    )
            )
    ) {
        return std::make_pair(cj, ci);
    }
    return std::make_pair(0, 0);
}
//-----------#core/detector/judge---------------------//


//-----------core/detector/getColorDistance-----------//
int Ousan_detector::getColorDistance(cv::Vec3b color1, cv::Vec3b color2) {
    //input Lab color
    return (uchar) sqrt(pow(color1[0] - color2[0], 2)
    + pow(color1[1] - color2[1], 2)
    + pow(color1[2] - color2[2], 2)) *
           255 / 411;
}

//---------#core/detector/getColorDistance------------//


//---------core/detector/judgeByDirections------------//
bool Ousan_detector::judgeByDirections(cv::Mat roi, int threshold_v, int og) {
    int n_same[9];
    memset(n_same, 0, sizeof(n_same));
    for (int n = 1; n <= 8; n++) {
        for (int m = 1; m <= kernel->direct_area; m++) {
            int ii = kernel->diameter / 2;
            int jj = kernel->diameter / 2;
            int ni = ii + kernel->direct[0][n][m];
            int nj = jj + kernel->direct[1][n][m];
            // if(thisKernel.at<Vec3b>(ni, nj)[0]<= t) {
            if (abs(roi.at<cv::Vec3b>(ni, nj)[0] - roi.at<cv::Vec3b>(ii, jj)[0]) <= threshold_v) {
                n_same[n]++;
            }
        }
    }

    sort(n_same + 1, n_same + 8 + 1, std::greater<int>());

    if (n_same[1] - n_same[8] <= og) {
        return false;
    }
    int d_same[9][2];
    for (int n = 1; n <= 8; n++) {
        d_same[n][0] = n;
        if (n_same[n] == 0) {
            d_same[n][1] = 0;
            continue;
        }
        d_same[n][1] = 1.0 * (n_same[n] - n_same[(n + 1) % 8]) / (1.0 * n_same[n]);
    }
    bool mainDir[9];
    for (int k = 1; k <= 8; k++) {
        if (d_same[k][1] > 0.5) {
            break;
        }
        mainDir[k] = true;
    }
    int rightSide = 0;
    for (int n = 1; n <= 8; n++) {
        if (mainDir[n] == 1 && mainDir[(n + 1) % 8] == 0) {
            rightSide++;
        }
    }

    if (rightSide == 1) {
        return true;
    }
    return false;
}
//--------#core/detector/judgeByDirections------------//


