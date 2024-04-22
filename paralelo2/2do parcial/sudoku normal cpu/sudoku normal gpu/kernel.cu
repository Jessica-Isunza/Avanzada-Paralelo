#include <iostream>
#include <vector>

#define SIZE 9

bool is_valid_cpu(char board[SIZE][SIZE], int row, int col, char num) {
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
    //subcuadro
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

void print_board_cpu(char board[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << board[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

//funcion recursiva
bool solve_sudoku_recursive_cpu(char board[SIZE][SIZE], int row, int col) {
    while (row < SIZE && board[row][col] != '.') {
        col++;
        if (col == SIZE) {
            col = 0;
            row++;
        }
    }

    //no hay celdas vacias
    if (row == SIZE) {
        return true;
    }

    for (char num = '1'; num <= '9'; num++) {
        if (is_valid_cpu(board, row, col, num)) {
            //colocar numero
            board[row][col] = num;

            if (solve_sudoku_recursive_cpu(board, row, col)) {
                return true;
            }

            //volver a intentar
            board[row][col] = '.';
        }
    }
    return false;
}

//llamar recursion
bool solve_sudoku_cpu(char board[SIZE][SIZE]) {
    return solve_sudoku_recursive_cpu(board, 0, 0);
}

int main() {
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
    print_board_cpu(board);

    if (solve_sudoku_cpu(board)) {
        std::cout << "\nsolucion:" << std::endl;
        print_board_cpu(board);
    }
    else {
        std::cout << "\nno se pudo resolver el sudoku :c" << std::endl;
    }

    return 0;
}