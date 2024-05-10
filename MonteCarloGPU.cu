#include <cuda.h>

#include "board.h"
#include "util.h"

int mcGPU_move(BoardState *state)
{
    // convert BoardState to a format that can be used by the GPU
    int board[BOARD_H * BOARD_W];
    for (int x = 0; x < BOARD_W; x++)
    {
        for (int y = 0; y < BOARD_H; y++)
        {
            board[BOARD_W * y + x] = state->board[x][y];
        }
    }
    int activePlayer = state->active_player;
    bool passed = state->passed;
}

__global__ void mcGPU_kernel(int *board, int activePlayer, bool passed)
{
    // copy board
    int boardCopy[BOARD_H * BOARD_W];
    for (int x = 0; x < BOARD_W; x++)
    {
        for (int y = 0; y < BOARD_H; y++)
        {
            boardCopy[BOARD_W * y + x] = board[BOARD_W * y + x];
        }
    }
}

__device__ int *get_valid_moves(int *board, int activePlayer, bool passed)
{
    int movesBuffer[BOARD_W * BOARD_H];
    // initialize buffer with -1
    for (int i = 0; i < BOARD_W * BOARD_H; i++)
    {
        movesBuffer[i] = -1;
    }

    int bufferIndex = 0;

    for (int x = 0; x < BOARD_W; x++)
    {
        for (int y = 0; y < BOARD_H; y++)
        {
            if (board[BOARD_W * y + x] == activePlayer)
            {
                auto f = [&](int _y, int _x)
                {
                    if (board[BOARD_W * _y + _x] == OTHER(activePlayer))
                    {

                        const int dx = _x - x;
                        const int dy = _y - y;

                        while (true)
                        {
                            _x += dx;
                            _y += dy;

                            if (!BOUNDS(_y, _x))
                                break;

                            if (board[BOARD_W * _y + _x] == activePlayer)
                                break;

                            if (board[BOARD_W * _y + _x] == EMPTY)
                            {
                                movesBuffer[bufferIndex] = BOARD_W * _y + _x;
                                bufferIndex++;
                                break;
                            }
                        }
                    }
                };

                map_adjacent(y, x, f);
            }
        }
    }

    return movesBuffer;
}

template <typename T>
__device__ void map_adjacent(const int y, const int x, const T f)
{

    if (y > 0)
    {
        f(y - 1, x);
        if (x > 0)
            f(y - 1, x - 1);
        if (x < (BOARD_W - 1))
            f(y - 1, x + 1);
    }

    if (y < (BOARD_H - 1))
    {
        f(y + 1, x);
        if (x > 0)
            f(y + 1, x - 1);
        if (x < (BOARD_W - 1))
            f(y + 1, x + 1);
    }

    if (x > 0)
        f(y, x - 1);

    if (x < (BOARD_W - 1))
        f(y, x + 1);
}