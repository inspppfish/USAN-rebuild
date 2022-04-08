//
// Created by insppp on 3/23/22.
//

#include "kernel_gen.h"
#include <cstring>

Kernel ::Kernel(int inputDiameter, uint32_t inputConfig ) {
    diameter = inputDiameter;
    config = inputConfig | KERNEL_STRUCTOR_ENABLE;
    area = 0;
}

Kernel ::Kernel (int inputDiameter, uint32_t inputConfig, std::vector<int> * inputGrp) {
    diameter = inputDiameter;
    config = inputConfig & ~KERNEL_STRUCTOR_ENABLE;
    area = (int)inputGrp[0].size();
    grp[0].insert(grp[0].end(), inputGrp[0].begin(), inputGrp[0].end());
    grp[1].insert(grp[1].end(), inputGrp[1].begin(), inputGrp[1].end());
}

////sample builder////
//void Kernel::kernelStructor() {
//    for (int i = -1 * this->diameter / 2; i <= this->diameter / 2; i++) {
//        for (int j = -1 * this->diameter / 2; j <= this->diameter / 2; j++) {
//            this->grp[0][++this->area] = i;
//            this->grp[1][this->area] = j;
//
//            if (config & KERNEL_STRUCTOR_PRINT_ENABLE)
//                printf(" 0");
//        }
//        if (config & KERNEL_STRUCTOR_PRINT_ENABLE)
//            printf("\n");
//    }
//}
///////////////////////
