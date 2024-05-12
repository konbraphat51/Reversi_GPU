#include <cuda.h>

#include "board.h"
#include "util.h"

int mcGPU_move(BoardState *state, int threads)
{
    // pass if no valid moves
    auto valid_moves = state->moves();
    if (valid_moves.size() == 0)
    {
        state->apply(PASS);
        return false;
    }

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

    // set up GPU
    int *d_board;
    int *d_movesCount;
    int *d_movesWins;

    cudaMalloc(&d_board, BOARD_H * BOARD_W * sizeof(int));
    cudaMalloc(&d_movesCount, valid_moves.size() * sizeof(int));
    cudaMalloc(&d_movesWins, valid_moves.size() * sizeof(int));

    cudaMemcpy(d_board, board, BOARD_H * BOARD_W * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_movesCount, 0, valid_moves.size() * sizeof(int));
    cudaMemset(d_movesWins, 0, valid_moves.size() * sizeof(int));

    dim3 dimBlock(threads);
    dim3 dimGrid(1);

    mcGPU_kernel<<<dimGrid, dimBlock>>>(d_board, activePlayer, passed, d_movesCount, d_movesWins);

    // get result
    double *winRate = (double *)malloc(valid_moves.size() * sizeof(double));
    for (int cnt = 0; cnt < valid_moves.size(); cnt++)
    {
        winRate[cnt] = (double)d_movesWins[cnt] / d_movesCount[cnt];
    }

    // free memory
    cudaFree(d_board);
    cudaFree(d_movesCount);
    cudaFree(d_movesWins);
}

__global__ void mcGPU_kernel(int *board, int activePlayer, bool passed, int *movesCount, int *movesWins)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = threadId;

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
    if (winner(boardCopy) == activePlayer)
    {
        movesWins[firstMoveIndex]++;
    }
}

struct Moves
{
    int *moves;
    int length;
};

__device__ Moves *get_valid_moves(int *board, int activePlayer)
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

    Moves *moves = new Moves();
    moves->moves = movesBuffer;
    moves->length = bufferIndex;

    return moves;
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

        map_adjacent(move / BOARD_W, move % BOARD_W, f);
    }

    active_player = OTHER(active_player);
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

__device__ int winner(
    const int *board)
{
    int w_score = 0;
    int b_score = 0;

    for (int i = 0; i < BOARD_H; ++i)
    {
        for (int j = 0; j < BOARD_W; ++j)
        {
            if (board[i * BOARD_W + j] == WHITE)
                w_score++;
            else if (board[i * BOARD_W + j] == BLACK)
                b_score++;
        }
    }

    if (w_score > b_score)
        return WHITE;
    if (w_score < b_score)
        return BLACK;
    return EMPTY;
}

__device__ __host__ int ComputeRandom(int &seed, int maxExclusive)
{
    // https://dl.acm.org/doi/10.1145/159544.376068
    int raw = (48271 * seed) % 2147483647;

    seed++;

    return raw % maxExclusive;
}