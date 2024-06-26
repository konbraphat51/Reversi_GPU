#pragma once

#include <iostream>
#include "board.h"
#include "util.h"

namespace MonteCarlo
{
    struct Moves
    {
        int *moves;
        int length;
    };

    template <typename T>
    void _map_adjacent(const int y, const int x, const T f)
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

    Moves *get_valid_moves(int *board, int activePlayer)
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

                    _map_adjacent(y, x, f);
                }
            }
        }

        Moves *moves = new Moves();

        // copy array
        moves->moves = new int[bufferIndex];
        int arrayIndex = 0;
        for (int bufferCnt = 0; bufferCnt < bufferIndex; bufferCnt++)
        {
            int thisMove = movesBuffer[bufferCnt];

            // avoid duplication
            bool duplicate = false;
            for (int i = 0; i < arrayIndex; i++)
            {
                if (moves->moves[i] == thisMove)
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
            {
                moves->moves[arrayIndex] = thisMove;
                arrayIndex++;
            }
        }
        moves->length = arrayIndex;

        return moves;
    }

    void apply_move(
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

            _map_adjacent(move / BOARD_W, move % BOARD_W, f);
        }

        active_player = OTHER(active_player);
    }

    int winner(
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

    int ComputeRandom(int &seed, int maxExclusive)
    {
        // https://dl.acm.org/doi/10.1145/159544.376068
        int raw = (48271 * seed) % 2147483647;

        seed++;

        return raw % maxExclusive;
    }

    void mcGPU_kernel(int seed, int *board, int activePlayer, bool passed, int *movesCount, int *movesWins)
    {
        int me = activePlayer;
        int other = OTHER(me);

        // copy board
        int boardCopy[BOARD_H * BOARD_W];
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
        }

        // report result
        movesCount[firstMoveIndex]++;
        if (winner(boardCopy, me, other) == me)
        {
            movesWins[firstMoveIndex]++;
        }
    }

    extern "C" int mc_move(BoardState *state, int threads)
    {
        printf("mcGPU_move called\n");

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

        // pass if no valid moves
        Moves *validMoves = get_valid_moves(board, activePlayer);
        if (validMoves->length == 0)
        {
            state->apply(PASS);
            return false;
        }

        // set up GPU
        int *d_board = (int *)malloc(BOARD_H * BOARD_W * sizeof(int));
        int *d_movesCount = (int *)malloc(validMoves->length * sizeof(int));
        int *d_movesWins = (int *)malloc(validMoves->length * sizeof(int));

        // memcpy(d_board, board, BOARD_H * BOARD_W * sizeof(int));
        for (int x = 0; x < BOARD_W; x++)
        {
            for (int y = 0; y < BOARD_H; y++)
            {
                d_board[BOARD_W * y + x] = board[BOARD_W * y + x];
            }
        }

        printf("Launching kernel \n");

        for (int cnt = 0; cnt < threads; cnt++)
        {
            mcGPU_kernel(cnt, d_board, activePlayer, passed, d_movesCount, d_movesWins);
        }

        // get result
        double *winRate = (double *)malloc(validMoves->length * sizeof(double));
        for (int cnt = 0; cnt < validMoves->length; cnt++)
        {
            winRate[cnt] = (double)d_movesWins[cnt] / d_movesCount[cnt];
        }
        // free memory
        free(d_board);
        free(d_movesCount);
        free(d_movesWins);

        // DEBUG: print result
        for (int cnt = 0; cnt < validMoves->length; cnt++)
        {
            printf("Move %d: %f\n", validMoves->moves[cnt], winRate[cnt]);
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
}
