#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>

#define SIZE 9
#define BLOCK_SIZE 512

__device__ bool valid_numb(char* board, int row, int col, char num) {
    // row
    for (int i = 0; i < SIZE; i++) {
        if (board[row * SIZE + i] == num) {
            return false;
        }
    }

    // col
    for (int i = 0; i < SIZE; i++) {
        if (board[i * SIZE + col] == num) {
            return false;
        }
    }

    // subgrid
    int start_row = 3 * (row / 3);
    int start_col = 3 * (col / 3);
    for (int i = start_row; i < start_row + 3; i++) {
        for (int j = start_col; j < start_col + 3; j++) {
            if (board[i * SIZE + j] == num) {
                return false;
            }
        }
    }

    return true;
}

__device__ bool sudoku_recursive(char* board, int row, int col) {
    while (row < SIZE && board[row * SIZE + col] != '.') {
        col++;
        if (col == SIZE) {
            col = 0;
            row++;
        }
    }

    if (row == SIZE) {
        return true;
    }

    for (char num = '1'; num <= '9'; num++) {
        if (valid_numb(board, row, col, num)) {
            board[row * SIZE + col] = num;

            if (sudoku_recursive(board, row, col)) {
                return true;
            }

            //intentar con otro numero
            board[row * SIZE + col] = '.';
        }
    }

    return false;
}

__global__ void sudoku_kernel(char* board, bool* solved) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    char local_board[SIZE * SIZE];
    bool local_solved = false;

    //mem local
    for (int i = 0; i < SIZE * SIZE; i++) {
        local_board[i] = board[i];
    }

    //sync
    __syncthreads();  //no he sabido por que no me deja usar __syncthreads 

    //recursion
    local_solved = sudoku_recursive(local_board, 0, 0);

    if (local_solved && !(*solved)) {
        *solved = true;

        //mem global
        for (int i = 0; i < SIZE * SIZE; i++) {
            board[i] = local_board[i];
        }
    }
}

bool sudoku(char* board) {
    char* d_board;
    bool* d_solved;
    bool solved = false;

    //host-dev
    cudaMalloc(&d_board, SIZE * SIZE * sizeof(char));
    cudaMemcpy(d_board, board, SIZE * SIZE * sizeof(char), cudaMemcpyHostToDevice);

    cudaMalloc(&d_solved, sizeof(bool));
    cudaMemcpy(d_solved, &solved, sizeof(bool), cudaMemcpyHostToDevice);

    sudoku_kernel << <(SIZE * SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_board, d_solved);

    //dev-host
    cudaMemcpy(&solved, d_solved, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_board);
    cudaFree(d_solved);

    return solved;
}

void print_board(char* board) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << board[i * SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    //tablero de ejemplo
    char board[SIZE * SIZE] = {
        '5', '3', '.', '.', '7', '.', '.', '.', '.',
        '6', '.', '.', '1', '9', '5', '.', '.', '.',
        '.', '9', '8', '.', '.', '.', '.', '6', '.',
        '8', '.', '.', '.', '6', '.', '.', '.', '3',
        '4', '.', '.', '8', '.', '3', '.', '.', '1',
        '7', '.', '.', '.', '2', '.', '.', '.', '6',
        '.', '6', '.', '.', '.', '.', '2', '8', '.',
        '.', '.', '.', '4', '1', '9', '.', '.', '5',
        '.', '.', '.', '.', '8', '.', '.', '7', '9'
    };

    std::cout << "Original board:" << std::endl;
    print_board(board);

    if (sudoku(board)) {
        std::cout << "\nSolved:" << std::endl;
        print_board(board);
    }
    else {
        std::cout << "\nFailed to solve Sudoku :c " << std::endl;
    }

    return 0;
}

