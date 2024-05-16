#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include <cmath>
#include <cassert>
#include <string>

#include "board.h"
#include "util.h"
#include "minimax.h"
#include "basic.h"
#include "uct.h"
#include "ucb.h"
#include "MonteCarloGPU.cuh"
#include "MonteCarlo.h"

using namespace std;
using namespace std::placeholders;

// board evaluation functions
eval_func eval_sampling_10 = bind(eval_sampling, _1, _2, 10);
eval_func eval_sampling_100 = bind(eval_sampling, _1, _2, 100);
eval_func eval_sampling_1000 = bind(eval_sampling, _1, _2, 1000);
// eval_pieces
// eval_inv_pieces

move_func strategies[] = {
    io_move,                                 // player
    random_move,                             // random
    bind(greedy_move, _1, eval_pieces),      // greedy
    bind(greedy_move, _1, eval_inv_pieces),  // generous
    bind(greedy_move, _1, eval_sampling_10), // greedy with random sampling
    bind(greedy_move, _1, eval_sampling_100),
    bind(greedy_move, _1, eval_sampling_1000),
    bind(uct_move, _1, 10), // UCT with various amounts of sampling
    bind(uct_move, _1, 100),
    bind(uct_move, _1, 1000),
    bind(ucb1_move, _1, 10), // UCB1 with various amounts of sampling
    bind(ucb1_move, _1, 100),
    bind(ucb1_move, _1, 1000),
    bind(minimax_move, _1, eval_sampling_10, 3), // Minimax by sampling
    bind(minimax_move, _1, eval_pieces, 3),      // Minimax by piece count
    bind(minimax_move, _1, eval_pieces, 4),      // Minimax by piece count
    bind(minimax_move, _1, eval_pieces, 5),      // Minimax by piece count
    bind(mcGPU_move, _1, 1e3),                   // Monte Carlo Tree Search (GPU)
    bind(MonteCarlo::mc_move, _1, 32 * 500)      // Monte Carlo Tree Search (CPU)
};

void testMcGPU()
{
  int seed = 0;

  const int round_per_agent = 100;

  for (int opponent = 1; opponent <= 16; opponent++)
  {
    int wins = 0;
    for (int i = 0; i < round_per_agent; i++)
    {
      wins += singleGame(17, opponent) == 17;
    }

    double win_rate = (double)wins / round_per_agent;

    cout << "Agent " << opponent << " win rate: " << win_rate * 100 << "%" << endl;
  }
}

int singleGame(int playerId1, int playerId2, bool print_result = false)
{
  BoardState state;

  move_func player_1 = strategies[playerId1]; // black
  move_func player_2 = strategies[playerId2];

  bool passed = false;
  while (true)
  {
    bool pass = !player_1(&state);
    if (pass && passed)
    {
      break;
    }
    passed = pass;

    swap(player_1, player_2);
  }

  int w_score = eval_pieces(&state, WHITE);
  int b_score = eval_pieces(&state, BLACK);

  if (print_result)
  {
    cout << "Player 1 score: " << b_score << endl;
    cout << "Player 2 score: " << w_score << endl;
  }

  // return winner
  if (w_score > b_score)
    return 2;
  else if (w_score < b_score)
    return 1;
  else
    return 0;
}

int main(int argc, char **argv)
{
  testMcGPU();
}