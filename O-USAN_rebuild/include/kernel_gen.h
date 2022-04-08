//
// Created by insppp on 3/23/22.
//

#ifndef DEMO_KERNEL_GEN_H
#define DEMO_KERNEL_GEN_H

#include <cstdint>
#include <vector>

#define KERNEL_STRUCTOR_PRINT_ENABLE 0x0001
#define KERNEL_STRUCTOR_ENABLE       0x0004

class Kernel {
public:
    int area; // total area of the kernel (including center)
    int diameter; //diameter of the kernel
    uint32_t config; // config that can be passed by kernel
    Kernel (int inputDiameter, uint32_t inputConfig); // Init::kernelStructor is needed
    Kernel (int diameter, uint32_t inputConfig, std::vector<int> * inputGrp); // Init::struct form existing map
    virtual void kernelStructor() = 0; // builder
    std::vector<int> grp[2];  // pos of the pixel (related to the center of kernel)
};

#endif //DEMO_KERNEL_GEN_H
