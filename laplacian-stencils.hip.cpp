#include <iostream>

#include "bricksetup.h"
#include "multiarray.h"
#include "brickcompare.h"
#include <omp.h>
#include <cmath>
#include <cassert>

__host__ void runNaiveTest(bool verify, bElem* arr_a, bElem *expected);
__host__ void runLargeBrickTest(bool verify, bElem *arr_a, bElem *expected);
__host__ void runNaiveArrayTest(bool verify, bElem *arr_a, bElem *expected);

#include <brick-hip.h>
#include "vecscatter.h"
#include "brick.h"

#define N 512
#define OFF (GZ + PADDING)
#define STRIDE (N + 2 * (OFF))

// there should be exactly one brick of ghost-zone
#define GZ (TILE)
#define GB (GZ / (TILE))

#define BLOCK (N / TILE)

#define NAIVE_BSTRIDE ((N + 2 * GZ) / (TILE))

#define VSVEC "HIP"
#define FOLD 8,8

#define BRICK_SIZE TILE, TILE, TILE

#define BType Brick<Dim<BRICK_SIZE>, Dim<FOLD>>

#define hipSynchronizeAssert(e) assert(hipDeviceSynchronize() == hipSuccess)

#ifdef NO_VERIFY
#define VERIFY false
#else
#define VERIFY true
#endif

__constant__ bElem dev_coeff[10];

__global__ void no_prof_single_thread_xpt(bElem *in, bElem *out, size_t radius) {
    bElem(*out_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) out;
    bElem(*in_sized)[STRIDE][STRIDE] = (bElem (*)[STRIDE][STRIDE]) in;
    for (int k = OFF; k < N + OFF; k++) {
        for (int j = OFF; j < N + OFF; j++) {
            for (int i = OFF; i < N + OFF; i++) {
                out_sized[k][j][i] = dev_coeff[0] * in_sized[k][j][i];
                // #pragma unroll
                for (int a = 1; a <= radius; a++) {
                    out_sized[k][j][i] = dev_coeff[a] * (
                        in_sized[k][j][i + a] + in_sized[k][j + a][i] + in_sized[k + a][j][i] +
                        in_sized[k][j][i - a] + in_sized[k][j - a][i] + in_sized[k - a][j][i]
                    );
                }
            }
        }
    }
}

__device__ void naive_xpt_sum(bElem (*in)[STRIDE][STRIDE], bElem (*out)[STRIDE][STRIDE], const size_t radius) {
    unsigned i = OFF + (hipBlockIdx_x) * TILE + hipThreadIdx_x;
    unsigned j = OFF + (hipBlockIdx_y) * TILE + hipThreadIdx_y;
    unsigned k = OFF + (hipBlockIdx_z) * TILE + hipThreadIdx_z;

    out[k][j][i] = dev_coeff[0] * in[k][j][i];
    // #pragma unroll
    for (int a = 1; a <= radius; a++) {
        out[k][j][i] += dev_coeff[a] * (
            in[k][j][i + a] + in[k][j + a][i] + in[k + a][j][i] +
            in[k][j][i - a] + in[k][j - a][i] + in[k - a][j][i]);
    }
}

__global__ void naive_13pt_sum(bElem *in, bElem *out) {
    return naive_xpt_sum((bElem (*)[STRIDE][STRIDE]) in, (bElem (*)[STRIDE][STRIDE]) out, 2);
}

__global__ void naive_31pt_sum(bElem (*in)[STRIDE][STRIDE], bElem (*out)[STRIDE][STRIDE]) {
    return naive_xpt_sum(in, out, 5);
}

__global__ void naive_49pt_sum(bElem (*in)[STRIDE][STRIDE], bElem (*out)[STRIDE][STRIDE]) {
    return naive_xpt_sum(in, out, 8);
}

__device__ void naive_brick_xpt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType &bIn, BType &bOut, size_t radius) {
    unsigned b = grid[hipBlockIdx_z + GB][hipBlockIdx_y + GB][hipBlockIdx_x + GB];
    unsigned i = hipThreadIdx_x + (hipBlockIdx_x) * TILE;
    unsigned j = hipThreadIdx_y + (hipBlockIdx_y) * TILE;
    unsigned k = hipThreadIdx_z + (hipBlockIdx_z) * TILE;
    bOut[b][k][j][i] = dev_coeff[0] * bIn[b][k][j][i];
    
    // #pragma unroll
    for (int a = 1; a <= radius; a++) {
        bOut[b][k][j][i] += dev_coeff[a] * (
            bIn[b][k][j][i + a] + bIn[b][k][j + a][i] + bIn[b][k + a][j][i] +
            bIn[b][k][j][i - a] + bIn[b][k][j - a][i] + bIn[b][k - a][j][i]
        );
    }
}

__global__ void naive_brick_13pt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    naive_brick_xpt(grid, bIn, bOut, 2);
}

__global__ void naive_brick_31pt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    naive_brick_xpt(grid, bIn, bOut, 5);
}

__global__ void naive_brick_49pt(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    naive_brick_xpt(grid, bIn, bOut, 8);
}

__global__ void brick_gen(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[hipBlockIdx_z + GB][hipBlockIdx_y + GB][hipBlockIdx_x + GB];
    brick("13pt.py", VSVEC, (TILE, TILE, TILE), (FOLD), b);
}

__global__ void brick_gen31(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[hipBlockIdx_z + GB][hipBlockIdx_y + GB][hipBlockIdx_x + GB];
    brick("31pt.py", VSVEC, (TILE, TILE, TILE), (FOLD), b);
}

__global__ void brick_gen49(unsigned (*grid)[NAIVE_BSTRIDE][NAIVE_BSTRIDE], BType bIn, BType bOut) {
    unsigned b = grid[hipBlockIdx_z + GB][hipBlockIdx_y + GB][hipBlockIdx_x + GB];
    brick("49pt.py", VSVEC, (TILE, TILE, TILE), (FOLD), b);
}

#define THRESH 0.001
__host__ void check_gpu_answer(bElem (*expected)[STRIDE][STRIDE], bElem *dev_solution, const char *error_message) {
    auto solution = (bElem (*)[STRIDE][STRIDE]) malloc(STRIDE * STRIDE * STRIDE * sizeof(bElem));
    hipMemcpy(solution, dev_solution, STRIDE * STRIDE * STRIDE * sizeof(bElem), hipMemcpyDeviceToHost);

    for (int i = OFF; i < N + OFF; i++) {
        for (int j = OFF; j < N + OFF; j++) {
            for (int k = OFF; k < N + OFF; k++) {
                if (abs(solution[i][j][k] - expected[i][j][k]) > THRESH) {
                    fprintf(stderr, "Error encountered at %d %d %d. Solution: %f, Expected %f\n", i, j, k, solution[i][j][k], expected[i][j][k]);
                    fflush(stderr);
                    throw std::runtime_error(error_message);
                }
            }
        }
    }

    free(solution);
}

__host__ void check_device_brick(bElem (*expected)[STRIDE][STRIDE], BrickStorage device_bstorage, BrickInfo<3> *binfo, unsigned brick_size, unsigned *bgrid, const char *error_message) {
    auto brick_storage = movBrickStorage(device_bstorage, hipMemcpyDeviceToHost);
    BType bOut(binfo, brick_storage, brick_size);
    if (!compareBrick<3>({N, N, N}, {PADDING, PADDING, PADDING}, {GZ, GZ, GZ}, (bElem *) expected, bgrid, bOut)) {
        throw std::runtime_error(error_message);
    }
}

#define bIn(a, b, c) arr_in[c][b][a]
#define bOut(a, b, c) arr_out[c][b][a]

__global__ void codegen_tile(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE]) {
    long k = OFF + (hipBlockIdx_z * TILE);
    long j = OFF + (hipBlockIdx_y * TILE);
    long i = OFF + (hipBlockIdx_x * 64);
    tile("13pt.py", VSVEC, (TILE, TILE, 64), ("k", "j", "i"), (1, 1, 64));
}

__global__ void codegen_tile31(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE]) {
    long k = OFF + (hipBlockIdx_z * TILE);
    long j = OFF + (hipBlockIdx_y * TILE);
    long i = OFF + (hipBlockIdx_x * 64);
    tile("31pt.py", VSVEC, (TILE, TILE, 64), ("k", "j", "i"), (1, 1, 64));
}

__global__ void codegen_tile49(bElem (*arr_in)[STRIDE][STRIDE], bElem (*arr_out)[STRIDE][STRIDE]) {
    long k = OFF + (hipBlockIdx_z * TILE);
    long j = OFF + (hipBlockIdx_y * TILE);
    long i = OFF + (hipBlockIdx_x * 64);
    tile("49pt.py", VSVEC, (TILE, TILE, 64), ("k", "j", "i"), (1, 1, 64));
}

#undef bIn
#undef bOut

int main(void) {
    bElem coeff[] = {1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    hipMemcpyToSymbol((bElem *) dev_coeff, (bElem *) coeff, 10 * sizeof(bElem), hipMemcpyDeviceToHost);

    // ---- CREATING BASIC ARRAYS ----
    bElem *arr_a = randomArray({STRIDE, STRIDE, STRIDE});
    bElem *arr_b = zeroArray({STRIDE, STRIDE, STRIDE});
    bElem *dev_a;
    bElem *dev_b;
    {
        unsigned size = STRIDE * STRIDE * STRIDE * sizeof(bElem);
        hipMalloc(&dev_b, size);
        hipMalloc(&dev_a, size);
        hipMemcpy(dev_a, arr_a, size, hipMemcpyHostToDevice);
    }
    // ---- DONE WITH BASIC ARRAYS ----


    // ---- GENERATE KNOWN SOLUTION ----
    printf("Generating expected\n");
    if (VERIFY) {
        auto expected = (bElem (*)[STRIDE][STRIDE]) malloc(STRIDE * STRIDE * STRIDE * sizeof(bElem));
        {
            bElem *dev_gpu_b;
            hipMalloc(&dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem));
            hipLaunchKernelGGL(no_prof_single_thread_xpt, 1, 1, 0, 0, dev_a, dev_gpu_b, 2);
            hipDeviceSynchronize();

            hipMemcpy((bElem *) expected, dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem), hipMemcpyDeviceToHost);
            hipFree(dev_gpu_b);
        }

        printf("Generating expected49\n");
        auto expected49 = (bElem (*)[STRIDE][STRIDE]) malloc(STRIDE * STRIDE * STRIDE * sizeof(bElem));
        {
            bElem *dev_gpu_b;
            hipMalloc(&dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem));
            hipLaunchKernelGGL(no_prof_single_thread_xpt, 1, 1, 0, 0, dev_a, dev_gpu_b, 8);
            hipDeviceSynchronize();

            hipMemcpy((bElem *) expected, dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem), hipMemcpyDeviceToHost);
            hipFree(dev_gpu_b);
        }

        printf("Generating expected 31\n");
        auto expected31 = (bElem (*)[STRIDE][STRIDE]) malloc(STRIDE * STRIDE * STRIDE * sizeof(bElem));
        {
            bElem *dev_gpu_b;
            hipMalloc(&dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem));
            hipLaunchKernelGGL(no_prof_single_thread_xpt, 1, 1, 0, 0, dev_a, dev_gpu_b, 5);
            hipDeviceSynchronize();

            hipMemcpy((bElem *) expected, dev_gpu_b, STRIDE * STRIDE * STRIDE * sizeof(bElem), hipMemcpyDeviceToHost);
            hipFree(dev_gpu_b);
        }
    }
    // ---- KNOWN SOLUTION GENERATED ----


    // ---- BRICK SETUP ----
    unsigned *bgrid;
    auto binfo = init_grid<3>(bgrid, {NAIVE_BSTRIDE, NAIVE_BSTRIDE, NAIVE_BSTRIDE});
    unsigned *device_bgrid;
    {
        unsigned grid_size = (NAIVE_BSTRIDE * NAIVE_BSTRIDE * NAIVE_BSTRIDE) * sizeof(unsigned);
        hipMalloc(&device_bgrid, grid_size);
        hipMemcpy(device_bgrid, bgrid, grid_size, hipMemcpyHostToDevice);
    }
    
    BrickInfo<3> _binfo = movBrickInfo(binfo, hipMemcpyHostToDevice);
    BrickInfo<3> *device_binfo;
    {
        unsigned binfo_size = sizeof(BrickInfo<3>);
        hipMalloc(&device_binfo, binfo_size);
        hipMemcpy(device_binfo, &_binfo, binfo_size, hipMemcpyHostToDevice);
    }
    auto brick_size = cal_size<BRICK_SIZE>::value;
    // double number of bricks for a and b
    auto brick_storage = BrickStorage::allocate(binfo.nbricks, brick_size * 2);

    BType bIn(&binfo, brick_storage, 0);
    BType bOut(&binfo, brick_storage, brick_size);
    copyToBrick<3>({N + 2 * GZ, N + 2 * GZ, N + 2 * GZ}, {PADDING, PADDING, PADDING}, {0, 0, 0}, arr_a, bgrid, bIn);

    BrickStorage device_bstorage = movBrickStorage(brick_storage, hipMemcpyHostToDevice);
    bIn = BType(device_binfo, device_bstorage, 0);
    bOut = BType(device_binfo, device_bstorage, brick_size);
    // ---- DONE WITH BRICK SETUP ----

    // ---- RUN TESTS ----
    dim3 blocks(BLOCK, BLOCK, BLOCK);
    dim3 threads(TILE, TILE, TILE);
    
    printf("Naive Array 13pt\n");
    hipLaunchKernelGGL(naive_13pt_sum, blocks, threads, 0, 0, dev_a, dev_b);
    hipSynchronizeAssert();
    if (VERIFY) check_gpu_answer(expected, dev_b, "Naive array solution mismatch");

    printf("Naive Array 31pt\n");
    hipLaunchKernelGGL(naive_31pt_sum, blocks, threads, 0, 0, (bElem (*)[STRIDE][STRIDE]) dev_a, (bElem (*)[STRIDE][STRIDE]) dev_b);
    hipSynchronizeAssert();
    if (VERIFY) check_gpu_answer(expected, dev_b, "Naive array solution mismatch");

    printf("Naive Array 49pt\n");
    hipLaunchKernelGGL(naive_49pt_sum, blocks, threads, 0, 0, (bElem (*)[STRIDE][STRIDE]) dev_a, (bElem (*)[STRIDE][STRIDE]) dev_b);
    hipSynchronizeAssert();
    if (VERIFY) check_gpu_answer(expected49, dev_b, "Naive array 49pt solution mismatch");

    {
        dim3 block(BLOCK, BLOCK, N / 64);
        dim3 thread(64, 1, 1);
        printf("Codegen Tile 13pt\n");
        hipLaunchKernelGGL(codegen_tile, block, thread, 0, 0, (bElem (*)[STRIDE][STRIDE]) dev_a, (bElem (*)[STRIDE][STRIDE]) dev_b);
        hipSynchronizeAssert();
        if (VERIFY) check_gpu_answer(expected, dev_b, "Codegen tile solution mismatch");

        printf("Codegen Tile 31pt\n");
        hipLaunchKernelGGL(codegen_tile31, block, thread, 0, 0, (bElem (*)[STRIDE][STRIDE]) dev_a, (bElem (*)[STRIDE][STRIDE]) dev_b);
        hipSynchronizeAssert();
        if (VERIFY) check_gpu_answer(expected31, dev_b, "Codegen tile solution mismatch");

        printf("Codegen Tile 49pt\n");
        hipLaunchKernelGGL(codegen_tile49, block, thread, 0, 0, (bElem (*)[STRIDE][STRIDE]) dev_a, (bElem (*)[STRIDE][STRIDE]) dev_b);
        hipSynchronizeAssert();
        if (VERIFY) check_gpu_answer(expected49, dev_b, "Codegen tile solution mismatch");
    }

    printf("Naive Brick 13pt\n");    
    hipLaunchKernelGGL(naive_brick_13pt, blocks, threads, 0, 0, (unsigned (*)[NAIVE_BSTRIDE][NAIVE_BSTRIDE]) device_bgrid, bIn, bOut);
    hipSynchronizeAssert();
    if (VERIFY) check_device_brick(expected, device_bstorage, &binfo, brick_size, bgrid, "Naive brick solution mismatch");

    printf("Naive Brick 31pt\n");    
    hipLaunchKernelGGL(naive_brick_31pt, blocks, threads, 0, 0, (unsigned (*)[NAIVE_BSTRIDE][NAIVE_BSTRIDE]) device_bgrid, bIn, bOut);
    hipSynchronizeAssert();
    if (VERIFY) check_device_brick(expected31, device_bstorage, &binfo, brick_size, bgrid, "Naive brick solution mismatch");

    printf("Naive Brick 49pt\n");    
    hipLaunchKernelGGL(naive_brick_49pt, blocks, threads, 0, 0, (unsigned (*)[NAIVE_BSTRIDE][NAIVE_BSTRIDE]) device_bgrid, bIn, bOut);
    hipSynchronizeAssert();
    if (VERIFY) check_device_brick(expected49, device_bstorage, &binfo, brick_size, bgrid, "Naive brick solution mismatch");

    printf("Brick Gen\n");
    hipLaunchKernelGGL(brick_gen, blocks, 64, 0, 0, (unsigned (*)[NAIVE_BSTRIDE][NAIVE_BSTRIDE]) device_bgrid, bIn, bOut);
    hipSynchronizeAssert();
    if (VERIFY) check_device_brick(expected, device_bstorage, &binfo, brick_size, bgrid, "Brick gen solution mismatch");

    printf("Brick Gen\n");
    hipLaunchKernelGGL(brick_gen31, blocks, 64, 0, 0, (unsigned (*)[NAIVE_BSTRIDE][NAIVE_BSTRIDE]) device_bgrid, bIn, bOut);
    hipSynchronizeAssert();
    if (VERIFY) check_device_brick(expected31, device_bstorage, &binfo, brick_size, bgrid, "Brick gen solution mismatch");

    printf("Brick Gen 49\n");
    hipLaunchKernelGGL(brick_gen49, blocks, 64, 0, 0, (unsigned (*)[NAIVE_BSTRIDE][NAIVE_BSTRIDE]) device_bgrid, bIn, bOut);
    hipSynchronizeAssert();
    if (VERIFY) check_device_brick(expected49, device_bstorage, &binfo, brick_size, bgrid, "Brick gen solution mismatch");
    // ---- DONE RUNNING TESTS ----


    // ---- CLEANUP ----
    free(arr_a);
    free(arr_b);
    free(bgrid);
    free(binfo.adj);
    hipFree(device_binfo);
    hipFree(device_bgrid);
    hipFree(dev_a);
    hipFree(dev_b);
    return 0;
}