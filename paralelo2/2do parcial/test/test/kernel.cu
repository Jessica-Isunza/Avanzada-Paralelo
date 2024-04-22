#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <Windows.h>  // Include Windows header for Sleep function

// The width and height of a sudoku board
#define BOARD_DIM 9

// The width and height of a square group in a sudoku board
#define GROUP_DIM 3

// The number of boards to pass to the solver at one time
#define BATCH_SIZE 1  // Since we're solving only one board

/**
 * A board is an array of 81 cells. Each cell is encoded as a 16-bit integer.
 */
typedef struct board {
    uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;

// Declare a few functions.
void print_board(board_t* board);
__host__ __device__ uint16_t digit_to_cell(int digit);
__host__ __device__ int cell_to_digit(uint16_t cell);

/**
 * This is the kernel to solve the sudoku boards in GPU.
 *
 * \param boards      An array of boards that should be solved.
 */
__global__ void cell_solver(board_t* boards) {
    size_t cell_idx = threadIdx.x;
    uint16_t current_cell;
    size_t votes;

    // shared memory for all the threads in the block.
    __shared__ board_t board;
    // copy the contents of the board into the shared memory
    board.cells[cell_idx] = boards[0].cells[cell_idx];  // Only one board in this case
    // wait for all the threads to finish copying the boards.
    __syncthreads();

    do {
        current_cell = board.cells[cell_idx];
        if (cell_to_digit(current_cell) != 0) break;
        // loop through the col
        size_t col_idx = cell_idx % 9;
        for (size_t index = col_idx; index < col_idx + 9 * 9; index += 9) {
            if (index == cell_idx) continue;
            int digit_result = cell_to_digit(board.cells[index]);
            if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
        }
        if (cell_to_digit(current_cell) != 0) break;
        // loop through the row
        size_t start_idx = cell_idx - col_idx;
        for (size_t index = start_idx; index < start_idx + 9; index++) {
            if (index == cell_idx) continue;
            int digit_result = cell_to_digit(board.cells[index]);
            if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
        }
        if (cell_to_digit(current_cell) != 0) break;
        // find the index of the top left corner of the square
        // reduced_index is the index of cell that has the same column
        // index but is in the first row.
        size_t reduced_index = cell_idx - (cell_idx / 27) * 27;
        size_t minor_row = reduced_index / 9;
        size_t minor_col = (reduced_index - minor_row * 9) % 3;
        // start_index is the index of cell at the top left corner that
        // share the same square of the current cell.
        size_t start_index = cell_idx - minor_col - minor_row * 9;
        // loop through the square
        for (size_t row = 0; row < 3; row++) {
            for (size_t col = 0; col < 3; col++) {
                size_t index = start_index + col + row * 9;
                if (index == cell_idx) continue;
                int digit_result = cell_to_digit(board.cells[index]);
                if (digit_result != 0) board.cells[cell_idx] &= ~(1 << digit_result);
            }
        }
        votes = __syncthreads_count(board.cells[cell_idx] != current_cell);

    } while (votes != 0);

    boards[0].cells[cell_idx] = board.cells[cell_idx];  // Only one board in this case
}

/**
 * Take an array of boards and solve them all.
 *
 * \param boards      An array of boards that should be solved.
 * \param num_boards  The number of boards in the boards array
 */
void solve_boards(board_t* cpu_boards, size_t num_boards) {
    // allocate memory in GPU
    board_t* gpu_boards;
    if (cudaMalloc(&gpu_boards, sizeof(board_t) * num_boards) != cudaSuccess) {
        perror("cuda malloc failed.");
        exit(2);
    }
    // copy the content over to GPU
    if (cudaMemcpy(gpu_boards, cpu_boards, sizeof(board_t) * num_boards, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        perror("cuda memcpy failed. ");
        exit(2);
    }
    // run the kernel over BATCH_SIZE blocks and 81 threads
    cell_solver << <1, 81 >> > (gpu_boards);  // Only one board in this case
    // wait for all the threads to finish
    if (cudaDeviceSynchronize() != cudaSuccess) {
        perror("Synchronized failed.");
        exit(2);
    }
    // copy contents from GPU to CPU.
    if (cudaMemcpy(cpu_boards, gpu_boards, sizeof(board_t) * num_boards, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        perror("cuda memcpy failed. ");
        exit(2);
    }
}

/**
 * Take as input an integer value 0-9 (inclusive) and convert it to the encoded
 * cell form used for solving the sudoku. This encoding uses bits 1-9 to
 * indicate which values may appear in this cell.
 *
 * For example, if bit 3 is set to 1, then the cell may hold a three. Cells that
 * have multiple possible values will have multiple bits set.
 *
 * The input digit 0 is treated specially. This value indicates a blank cell,
 * where any value from one to nine is possible.
 *
 * \param digit   An integer value 0-9 inclusive
 * \returns       The encoded form of digit using bits to indicate which values
 *                may appear in this cell.
 */
__host__ __device__ uint16_t digit_to_cell(int digit) {
    if (digit == 0) {
        // A zero indicates a blank cell. Numbers 1-9 are possible, so set bits 1-9.
        return 0x3FE;
    }
    else {
        // Otherwise we have a fixed value. Set the corresponding bit in the board.
        return 1 << digit;
    }
}

__host__ __device__ int count_trailing_zeros(uint16_t value) {
    int count = 0;
    while ((value & 1) == 0) {
        value >>= 1;
        count++;
    }
    return count;
}


/*
 * Convert an encoded cell back to its digit form. A cell with two or more
 * possible values will be encoded as a zero. Cells with one possible value
 * will be converted to that value.
 *
 *
 * \param cell  An encoded cell that uses bits to indicate which values could
 *              appear at this point in the board.
 * \returns     The value that must appear in the cell if there is only one
 *              possibility, or zero otherwise.
 */
__host__ __device__ int cell_to_digit(uint16_t cell) {
    // Get the index of the least-significant bit in this cell's value
    int lsb = count_trailing_zeros(cell);

    // Is there only one possible value for this cell? If so, return it.
    // Otherwise return zero.
    if (cell == 1 << lsb)
        return lsb;
    else
        return 0;
}

// Function to print the Sudoku board
void print_board(board_t* board) {
    for (int i = 0; i < BOARD_DIM; i++) {
        for (int j = 0; j < BOARD_DIM; j++) {
            int digit = cell_to_digit(board->cells[i * BOARD_DIM + j]);
            if (digit != 0) {
                printf("%d ", digit);
            }
            else {
                printf(". ");
            }
        }
        printf("\n");
    }
}



/**
 * Entry point for the program
 */
int main(int argc, char** argv) {

    // Predefined sudoku board
    char board[BOARD_DIM][BOARD_DIM] = {
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

    void print_board(board_t * board);

    // Initialize the sudoku board with the predefined values
    board_t sudoku_board;
    for (int i = 0; i < BOARD_DIM; i++) {
        for (int j = 0; j < BOARD_DIM; j++) {
            char current_char = board[i][j];
            int digit;
            if (current_char >= '1' && current_char <= '9') {
                digit = current_char - '0';  // Convert character to integer
            }
            else {
                digit = 0;  // Treat '.' as 0 (blank cell)
            }
            sudoku_board.cells[i * BOARD_DIM + j] = digit_to_cell(digit);
        }
    }


    // Measure the start time
    clock_t start_time = clock();

    // Solve the sudoku board
    solve_boards(&sudoku_board, 1); 

    // Measure the end time
    clock_t end_time = clock();

    // Calculate the elapsed time in milliseconds
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;


    // Print the solved board
    print_board(&sudoku_board);
    printf("Elapsed time: %.2f milliseconds\n", elapsed_time);

    return 0;
}
