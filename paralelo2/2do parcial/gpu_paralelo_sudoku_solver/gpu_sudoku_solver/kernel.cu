#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

#define SIZE 9

__device__ bool is_valid_gpu(char board[SIZE][SIZE], int row, int col, char num) {
    //row
    for (int i = 0; i < SIZE; i++) {
        if (board[row][i] == num) {
            return false;
        }
    }
    //col
    for (int i = 0; i < SIZE; i++) {
        if (board[i][col] == num) {
            return false;
        }
    }
    //subgrid
    int start_row = 3 * (row / 3);
    int start_col = 3 * (col / 3);
    for (int i = start_row; i < start_row + 3; i++) {
        for (int j = start_col; j < start_col + 3; j++) {
            if (board[i][j] == num) {
                return false;
            }
        }
    }
    return true;
}

__device__ bool solve_sudoku_recursive_gpu(char board[SIZE][SIZE], int row, int col) {
    if (row == SIZE) {
        return true;
    }

    //si no esta vacia a siguiente celda
    if (board[row][col] != '.') {
        int next_row = row + (col + 1) / SIZE;
        int next_col = (col + 1) % SIZE;
        return solve_sudoku_recursive_gpu(board, next_row, next_col);
    }

    for (char num = '1'; num <= '9'; num++) {
        if (is_valid_gpu(board, row, col, num)) {
            board[row][col] = num;

            //recursion
            int next_row = row + (col + 1) / SIZE;
            int next_col = (col + 1) % SIZE;
            if (solve_sudoku_recursive_gpu(board, next_row, next_col)) {
                return true; 
            }

            //quitar numero para resolver otra vez
            board[row][col] = '.';
        }
    }
    return false;
}

__global__ void solve_sudoku_kernel(char* dev_board, bool* dev_results) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Cada hilo resuelve una instancia diferente del sudoku
    char board[SIZE][SIZE];
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            board[i][j] = dev_board[index * SIZE * SIZE + i * SIZE + j];
        }
    }

    // Resuelve el sudoku para la instancia asignada al hilo
    dev_results[index] = solve_sudoku_recursive_gpu(board, 0, 0);
}

bool solve_sudoku_gpu(char board[SIZE][SIZE]) {
    std::vector<char> board_vec;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            board_vec.push_back(board[i][j]);
        }
    }

    char* dev_board;
    bool* dev_results;
    cudaMalloc((void**)&dev_board, SIZE * SIZE * sizeof(char));
    cudaMalloc((void**)&dev_results, SIZE * sizeof(bool));

    cudaMemcpy(dev_board, board_vec.data(), SIZE * SIZE * sizeof(char), cudaMemcpyHostToDevice);

    solve_sudoku_kernel << <5, SIZE >> > (dev_board, dev_results);

    bool results[SIZE];
    cudaMemcpy(results, dev_results, SIZE * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(dev_board);
    cudaFree(dev_results);

    // Verifica si alguna de las instancias del sudoku se resolvió con éxito
    for (int i = 0; i < SIZE; i++) {
        if (results[i]) {
            return true;
        }
    }
    return false;
}

int main() {
    // Tablero de ejemplo
    char board[SIZE][SIZE] = {
        {'5', '3', '.', '.', '7', '.', '.', '.', '.'},
        {'6', '.', '.', '1', '9', '5', '.', '.', '.'},
        {'.', '9', '8', '.', '.', '.', '.', '6', '.'},
        {'8', '.', '.', '.', '6', '.', '.', '.', '3'},
        {'4', '.', '.', '8', '.', '3', '.', '.', '1'},
        {'7', '.', '.', '.', '2', '.', '.', '.', '6'},
        {'.', '6', '.', '.', '.', '.', '2', '8', '.'},
        {'.', '.', '.', '4', '1', '9', '.', '.', '5'},
        {'.', '.', '.', '.', '8', '.', '.', '7', '9'}
    };

    std::cout << "tablero original:" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << board[i][j] << " ";
        }
        std::cout << std::endl;
    }

    if (solve_sudoku_gpu(board)) {
        std::cout << "\nsolucion:" << std::endl;
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                std::cout << board[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        std::cout << "\nno se pudo resolver el sudoku :c" << std::endl;
    }

    return 0;
}
//no lo resuelve :cc lo deja igual pero si corre el kernel y si actualiza en solve_sudoku_recursive_gpu
