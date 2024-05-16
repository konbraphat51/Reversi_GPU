#pragma once

#include <iostream>
#include "MonteCarloGPU.cuh"
#include "board.h"
#include "util.h"

struct Moves
{
    int *moves;
    int length;
};

__device__ __host__ void update_valid_moves(const int x, const int y, int current_x, int current_y, int *board, int activePlayer, int *movesBuffer, int &bufferIndex)
{
    if (board[BOARD_W * current_y + current_x] == OTHER(activePlayer))
    {
        const int dx = current_x - x;
        const int dy = current_y - y;

        while (true)
        {
            current_x += dx;
            current_y += dy;

            if (!BOUNDS(current_y, current_x))
                break;

            if (board[BOARD_W * current_y + current_x] == activePlayer)
                break;

            if (board[BOARD_W * current_y + current_x] == EMPTY)
            {
                int move = BOARD_W * current_y + current_x;

                // avoid duplication
                bool duplicate = false;
                for (int i = 0; i < bufferIndex; i++)
                {
                    if (movesBuffer[i] == move)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate)
                {
                    // not duplicated
                    movesBuffer[bufferIndex] = move;
                    bufferIndex++;
                }

                break;
            }
        }
    }
}

__host__ __device__ void get_moves_adjacent(int x, int y, int *board, int activePlayer, int *movesBuffer, int &bufferIndex)
{
    if (y > 0)
    {
        update_valid_moves(x, y, x, y - 1, board, activePlayer, movesBuffer, bufferIndex);
        if (x > 0)
            update_valid_moves(x, y, x - 1, y - 1, board, activePlayer, movesBuffer, bufferIndex);
        if (x < (BOARD_W - 1))
            update_valid_moves(x, y, x + 1, y - 1, board, activePlayer, movesBuffer, bufferIndex);
    }

    if (y < (BOARD_H - 1))
    {
        update_valid_moves(x, y, x, y + 1, board, activePlayer, movesBuffer, bufferIndex);
        if (x > 0)
            update_valid_moves(x, y, x - 1, y + 1, board, activePlayer, movesBuffer, bufferIndex);
        if (x < (BOARD_W - 1))
            update_valid_moves(x, y, x + 1, y + 1, board, activePlayer, movesBuffer, bufferIndex);
    }

    if (x > 0)
        update_valid_moves(x, y, x - 1, y, board, activePlayer, movesBuffer, bufferIndex);

    if (x < (BOARD_W - 1))
        update_valid_moves(x, y, x + 1, y, board, activePlayer, movesBuffer, bufferIndex);
}

__device__ __host__ Moves *get_valid_moves(int *board, int activePlayer)
{
    int movesBuffer[64];
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
                get_moves_adjacent(x, y, board, activePlayer, movesBuffer, bufferIndex);
            }
        }
    }

    Moves *moves = new Moves();

    // copy array
    moves->moves = (int *)malloc(bufferIndex * sizeof(int));
    for (int cnt = 0; cnt < bufferIndex; cnt++)
    {
        moves->moves[cnt] = movesBuffer[cnt];
    }

    moves->length = bufferIndex;

    delete movesBuffer;

    return moves;
}

__device__ __host__ void update_applying_move(const int x, const int y, int current_x, int current_y, int *board, int activePlayer)
{
    if (board[BOARD_W * current_y + current_x] == OTHER(activePlayer))
    {

        const int dx = current_x - x;
        const int dy = current_y - y;

        while (true)
        {
            current_x += dx;
            current_y += dy;

            if (!BOUNDS(current_y, current_x))
                break;

            if (board[BOARD_W * current_y + current_x] == activePlayer)
            {
                while (true)
                {
                    current_x -= dx;
                    current_y -= dy;

                    if (board[BOARD_W * current_y + current_x] == activePlayer)
                        break;

                    board[BOARD_W * current_y + current_x] = activePlayer;
                }
                break;
            }

            if (board[BOARD_W * current_y + current_x] == EMPTY)
            {
                break;
            }
        }
    }
}

__host__ __device__ void apply_move_adjacent(int x, int y, int *board, int activePlayer)
{
    if (y > 0)
    {
        update_applying_move(x, y, x, y - 1, board, activePlayer);
        if (x > 0)
            update_applying_move(x, y, x - 1, y - 1, board, activePlayer);
        if (x < (BOARD_W - 1))
            update_applying_move(x, y, x + 1, y - 1, board, activePlayer);
    }

    if (y < (BOARD_H - 1))
    {
        update_applying_move(x, y, x, y + 1, board, activePlayer);
        if (x > 0)
            update_applying_move(x, y, x - 1, y + 1, board, activePlayer);
        if (x < (BOARD_W - 1))
            update_applying_move(x, y, x + 1, y + 1, board, activePlayer);
    }

    if (x > 0)
        update_applying_move(x, y, x - 1, y, board, activePlayer);

    if (x < (BOARD_W - 1))
        update_applying_move(x, y, x + 1, y, board, activePlayer);
}

__device__ void apply_move(
    const int move,
    int *board,
    int &active_player,
    bool &passed)
{
    // pass
    if (move == -1)
    {
        passed = true;
    }
    else
    {
        passed = false;
        assert(board[move] == EMPTY);
        board[move] = active_player;

        auto f = [&](int y, int x)
        {
            if (board[BOARD_W * y + x] == OTHER(active_player))
            {

                const int dx = x - move % BOARD_W;
                const int dy = y - move / BOARD_W;

                while (true)
                {
                    x += dx;
                    y += dy;

                    if (!BOUNDS(y, x))
                        break;

                    if (board[BOARD_W * y + x] == active_player)
                    {
                        while (true)
                        {
                            x -= dx;
                            y -= dy;

                            if (board[BOARD_W * y + x] == active_player)
                                break;

                            board[BOARD_W * y + x] = active_player;
                        }
                        break;
                    }

                    if (board[BOARD_W * y + x] == EMPTY)
                    {
                        break;
                    }
                }
            }
        };

        apply_move_adjacent(move % BOARD_W, move / BOARD_W, board, active_player);
    }

    active_player = OTHER(active_player);
}

__device__ int winner(
    const int *board,
    const int me,
    const int other)
{
    int my_score = 0;
    int other_score = 0;

    for (int i = 0; i < BOARD_H; ++i)
    {
        for (int j = 0; j < BOARD_W; ++j)
        {
            if (board[i * BOARD_W + j] == me)
                my_score++;
            else if (board[i * BOARD_W + j] == other)
                other_score++;
        }
    }

    if (my_score > other_score)
        return me;
    if (my_score < other_score)
        return other;
    return -1;
}

__device__ __host__ int ComputeRandom(int &seed, int maxExclusive)
{
    // https://dl.acm.org/doi/10.1145/159544.376068
    int raw = (48271 * seed) % 2147483647;

    seed++;

    return raw % maxExclusive;
}

__global__ void mcGPU_kernel(int *board, int activePlayer, bool passed, int *movesCount, int *movesWins)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = threadId;

    int me = activePlayer;
    int other = OTHER(activePlayer);

    // copy board
    int boardCopy[64];
    for (int x = 0; x < BOARD_W; x++)
    {
        for (int y = 0; y < BOARD_H; y++)
        {
            boardCopy[BOARD_W * y + x] = board[BOARD_W * y + x];
        }
    }

    // monte carlo simulation
    bool first = true;
    int firstMoveIndex = -1;
    for (;;)
    {
        Moves *validMoves = get_valid_moves(boardCopy, activePlayer);

        if (validMoves->length == 0)
        {
            if (passed)
            {
                // both passed, game is over
                delete validMoves;
                break;
            }
            else
            {
                // first pass
                passed = true;
            }
        }
        else
        {
            // choose a random move
            int moveIndex = ComputeRandom(seed, validMoves->length);
            int move = validMoves->moves[moveIndex];
            apply_move(move, boardCopy, activePlayer, passed);

            if (first)
            {
                first = false;
                firstMoveIndex = moveIndex;
            }
        }

        delete validMoves;
    }

    // report result
    atomicAdd(&movesCount[firstMoveIndex], 1);
    if (winner(boardCopy, me, other) == me)
    {
        atomicAdd(&movesWins[firstMoveIndex], 1);
    }

    delete boardCopy;
}

extern "C" int mcGPU_move(BoardState *state, int threads)
{
    printf("mcGPU_move called\n");

    // convert BoardState to a format that can be used by the GPU
    int *board = (int *)malloc(BOARD_H * BOARD_W * sizeof(int));
    for (int x = 0; x < BOARD_W; x++)
    {
        for (int y = 0; y < BOARD_H; y++)
        {
            board[BOARD_W * y + x] = state->board[x][y];
        }
    }
    int activePlayer = state->active_player;
    bool passed = state->passed;

    // pass if no valid moves
    Moves *validMoves = get_valid_moves(board, activePlayer);
    if (validMoves->length == 0)
    {
        state->apply(PASS);
        return false;
    }

    // set up GPU
    int *h_movesCount = (int *)malloc(validMoves->length * sizeof(int));
    int *h_movesWins = (int *)malloc(validMoves->length * sizeof(int));
    for (int i = 0; i < validMoves->length; i++)
    {
        h_movesCount[i] = 0;
        h_movesWins[i] = 0;
    }
    int *d_board;
    int *d_movesCount;
    int *d_movesWins;

    cudaMalloc((void **)&d_board, BOARD_H * BOARD_W * sizeof(int));
    cudaMalloc((void **)&d_movesCount, validMoves->length * sizeof(int));
    cudaMalloc((void **)&d_movesWins, validMoves->length * sizeof(int));

    cudaMemcpy(d_board, board, BOARD_H * BOARD_W * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_movesCount, h_movesCount, validMoves->length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_movesWins, h_movesWins, validMoves->length * sizeof(int), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 64;
    // dim3 dimGrid(threads / threadsPerBlock, 1);
    // dim3 dimBlock(threadsPerBlock, 1, 1);
    // DEBUG
    dim3 dimGrid(5, 1);
    dim3 dimBlock(32, 1, 1);

    printf("Launching kernel \n");

    mcGPU_kernel<<<dimGrid, dimBlock>>>(d_board, activePlayer, passed, d_movesCount, d_movesWins);

    printf("Kernel finished \n");

    // copy results back
    cudaMemcpy(h_movesCount, d_movesCount, validMoves->length * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_movesWins, d_movesWins, validMoves->length * sizeof(int), cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_board);
    cudaFree(d_movesCount);
    cudaFree(d_movesWins);
    free(board);

    // get result
    printf("validMoves->length: %d\n", validMoves->length);
    double *winRate = (double *)malloc(validMoves->length * sizeof(double));
    for (int cnt = 0; cnt < validMoves->length; cnt++)
    {
        winRate[cnt] = (double)h_movesWins[cnt] / h_movesCount[cnt];
        printf("Move %d: %d wins, %d total, win rate: %f\n", validMoves->moves[cnt], h_movesWins[cnt], h_movesCount[cnt], winRate[cnt]);
    }

    // find best move
    int bestMove = -1;
    double bestWinRate = -1;
    for (int cnt = 0; cnt < validMoves->length; cnt++)
    {
        if (winRate[cnt] > bestWinRate)
        {
            bestWinRate = winRate[cnt];
            bestMove = validMoves->moves[cnt];
        }
    }

    // apply best move
    Point p = Point(bestMove % BOARD_W, bestMove / BOARD_W);
    state->apply(p);

    return true;
}
