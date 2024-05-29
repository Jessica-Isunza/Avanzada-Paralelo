#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// Verifica errores de CUDA
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel para la escala de grises
__global__ void escalaGrisesKernel(uchar3* img_src, uchar3* img_dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        uchar3 pixel = img_src[idx];
        unsigned char gray = (pixel.x + pixel.y + pixel.z) / 3;
        img_dst[idx] = make_uchar3(gray, gray, gray);
    }
}

// Kernel para el filtro gaussiano 5x5
__global__ void filtroGauss5x5Kernel(uchar3* img_src, uchar3* img_dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int filtro[5][5] = { {0, 1, 2, 1, 0},
                         {1, 3, 5, 3, 1},
                         {2, 5, 9, 5, 2},
                         {1, 3, 5, 3, 1},
                         {0, 1, 2, 1, 0} };
    int divisor = 49; // Suma de los elementos del filtro

    if (x >= 2 && x < cols - 2 && y >= 2 && y < rows - 2) {
        double total = 0.0;
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                int idx = (y + i) * cols + (x + j);
                total += img_src[idx].x * filtro[i + 2][j + 2];
            }
        }
        total /= divisor;
        img_dst[y * cols + x] = make_uchar3(total, total, total);
    }
}

// Kernel para el filtro gaussiano 9x9
__global__ void filtroGauss9x9Kernel(uchar3* img_src, uchar3* img_dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int filtro[9][9] = { {0, 1, 2, 3, 4, 3, 2, 1, 0},
                         {1, 3, 5, 7, 9, 7, 5, 3, 1},
                         {2, 5, 8, 11, 14, 11, 8, 5, 2},
                         {3, 7, 11, 15, 19, 15, 11, 7, 3},
                         {4, 9, 14, 19, 24, 19, 14, 9, 4},
                         {3, 7, 11, 15, 19, 15, 11, 7, 3},
                         {2, 5, 8, 11, 14, 11, 8, 5, 2},
                         {1, 3, 5, 7, 9, 7, 5, 3, 1},
                         {0, 1, 2, 3, 4, 3, 2, 1, 0} };
    int divisor = 285; // Suma de los elementos del filtro

    if (x >= 4 && x < cols - 4 && y >= 4 && y < rows - 4) {
        double total = 0.0;
        for (int i = -4; i <= 4; i++) {
            for (int j = -4; j <= 4; j++) {
                int idx = (y + i) * cols + (x + j);
                total += img_src[idx].x * filtro[i + 4][j + 4];
            }
        }
        total /= divisor;
        img_dst[y * cols + x] = make_uchar3(total, total, total);
    }
}

// Kernel para el análisis de bordes
__global__ void bordesVideoKernel(uchar3* img_src, uchar3* img_dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Hx[3][3] = { {0, -1, 0},
                     {-1, 4, -1},
                     {0, -1, 0} };
    int Hy[3][3] = { {0, -1, 0},
                     {-1, 4, -1},
                     {0, -1, 0} };

    if (x > 0 && x < cols - 1 && y > 0 && y < rows - 1) {
        double grad_x = 0.0, grad_y = 0.0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int idx = (y + i) * cols + (x + j);
                grad_x += img_src[idx].x * Hx[i + 1][j + 1];
                grad_y += img_src[idx].x * Hy[i + 1][j + 1];
            }
        }

        double magnitude = sqrt(grad_x * grad_x + grad_y * grad_y);
        if (magnitude > 50) {
            img_dst[y * cols + x] = make_uchar3(255, 255, 255);
        }
        else {
            img_dst[y * cols + x] = make_uchar3(0, 0, 0);
        }
    }
}

// Kernel para dibujar los bordes
__global__ void dibujarBordesVideoKernel(uchar3* img_bordes, uchar3* img_color, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        if (img_bordes[idx].x != 0) {
            img_color[idx] = make_uchar3(0, 255, 0);
        }
    }
}

void escalaGrises(Mat img_src, Mat* img_dst) {
    int rows = img_src.rows;
    int cols = img_src.cols;
    uchar3* d_img_src;
    uchar3* d_img_dst;

    cudaCheckError(cudaMalloc(&d_img_src, rows * cols * sizeof(uchar3)));
    cudaCheckError(cudaMalloc(&d_img_dst, rows * cols * sizeof(uchar3)));

    cudaCheckError(cudaMemcpy(d_img_src, img_src.ptr(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    escalaGrisesKernel << <gridDim, blockDim >> > (d_img_src, d_img_dst, rows, cols);

    cudaCheckError(cudaMemcpy(img_dst->ptr(), d_img_dst, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_img_src));
    cudaCheckError(cudaFree(d_img_dst));
}

void filtroGauss5x5Video(Mat img_src, Mat* img_dst) {
    int rows = img_src.rows;
    int cols = img_src.cols;
    uchar3* d_img_src;
    uchar3* d_img_dst;

    cudaCheckError(cudaMalloc(&d_img_src, rows * cols * sizeof(uchar3)));
    cudaCheckError(cudaMalloc(&d_img_dst, rows * cols * sizeof(uchar3)));

    cudaCheckError(cudaMemcpy(d_img_src, img_src.ptr(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    filtroGauss5x5Kernel << <gridDim, blockDim >> > (d_img_src, d_img_dst, rows, cols);

    cudaCheckError(cudaMemcpy(img_dst->ptr(), d_img_dst, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_img_src));
    cudaCheckError(cudaFree(d_img_dst));
}

void filtroGauss9x9Video(Mat img_src, Mat* img_dst) {
    int rows = img_src.rows;
    int cols = img_src.cols;
    uchar3* d_img_src;
    uchar3* d_img_dst;

    cudaCheckError(cudaMalloc(&d_img_src, rows * cols * sizeof(uchar3)));
    cudaCheckError(cudaMalloc(&d_img_dst, rows * cols * sizeof(uchar3)));

    cudaCheckError(cudaMemcpy(d_img_src, img_src.ptr(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    filtroGauss9x9Kernel << <gridDim, blockDim >> > (d_img_src, d_img_dst, rows, cols);

    cudaCheckError(cudaMemcpy(img_dst->ptr(), d_img_dst, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_img_src));
    cudaCheckError(cudaFree(d_img_dst));
}

void bordesVideo(Mat img_src, Mat* img_dst) {
    int rows = img_src.rows;
    int cols = img_src.cols;
    uchar3* d_img_src;
    uchar3* d_img_dst;

    cudaCheckError(cudaMalloc(&d_img_src, rows * cols * sizeof(uchar3)));
    cudaCheckError(cudaMalloc(&d_img_dst, rows * cols * sizeof(uchar3)));

    cudaCheckError(cudaMemcpy(d_img_src, img_src.ptr(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    bordesVideoKernel << <gridDim, blockDim >> > (d_img_src, d_img_dst, rows, cols);

    cudaCheckError(cudaMemcpy(img_dst->ptr(), d_img_dst, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_img_src));
    cudaCheckError(cudaFree(d_img_dst));
}

void dibujarBordesVideo(Mat img_bordes, Mat* img_color) {
    int rows = img_bordes.rows;
    int cols = img_bordes.cols;
    uchar3* d_img_bordes;
    uchar3* d_img_color;

    cudaCheckError(cudaMalloc(&d_img_bordes, rows * cols * sizeof(uchar3)));
    cudaCheckError(cudaMalloc(&d_img_color, rows * cols * sizeof(uchar3)));

    cudaCheckError(cudaMemcpy(d_img_bordes, img_bordes.ptr(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_img_color, img_color->ptr(), rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    dibujarBordesVideoKernel << <gridDim, blockDim >> > (d_img_bordes, d_img_color, rows, cols);

    cudaCheckError(cudaMemcpy(img_color->ptr(), d_img_color, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_img_bordes));
    cudaCheckError(cudaFree(d_img_color));
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat frame;
    Mat gray_frame;
    Mat gauss5_frame;
    Mat gauss9_frame;
    Mat edges_frame;
    Mat final_frame;

    while (1) {
        cap >> frame;
        if (frame.empty())
            break;

        // Inicializar las matrices para cada etapa
        gray_frame = Mat(frame.size(), frame.type());
        gauss5_frame = Mat(frame.size(), frame.type());
        gauss9_frame = Mat(frame.size(), frame.type());
        edges_frame = Mat(frame.size(), frame.type());
        final_frame = frame.clone();

        // Escala de Grises
        escalaGrises(frame, &gray_frame);

        // Filtro Gaussiano 5x5
        filtroGauss5x5Video(gray_frame, &gauss5_frame);

        // Filtro Gaussiano 9x9
        filtroGauss9x9Video(gauss5_frame, &gauss9_frame);

        // Análisis de Bordes
        bordesVideo(gauss9_frame, &edges_frame);

        // Dibujo de Bordes en el Color Original
        dibujarBordesVideo(edges_frame, &final_frame);

        // Mostrar cada etapa en una ventana separada
        imshow("Original Video", frame);
        imshow("Grayscale Video", gray_frame);
        imshow("Gaussian 5x5 Video", gauss5_frame);
        imshow("Gaussian 9x9 Video", gauss9_frame);
        imshow("Edges Video", edges_frame);
        imshow("Final Processed Video", final_frame);

        if (waitKey(10) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
