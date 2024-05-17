# GPU MonteCarlo Reversi Fighter
## Original Repository
SPECIAL THANKS:  
https://github.com/psaikko/mcts-reversi

infrastucture and other algorithms AIs.

## Set up
docker set up
```
docker build -t reversi .
docker run -it --rm reversi
```

running
```
cd /app/source
make
./run
```

## Result
Against 32*500 times MonteCarlo Simulator. (0.1s per turn)
|Opponent|Opponent's time per turn|Win Rate|
|---|---|---|
|random_move|0s|100%|
|greedy move|0s|100%|
|generous move|0s|100%|
|greedy move with 10 sampling|0.01s|55%|
|greedy move with 100 sampling|0.07s|25%|
|greedy move with 1000 sampling|0.5s|15%|
|UCT with 10 sampling|0.004s|60%|
|UCT with 100 sampling|0.06s|15%|
|UCT with 1000 sampling|0.5s|5%|
|UCB1 with 10 sampling|0.004s|55%|
|UCB1 with 100 sampling|0.05s|15%|
|UCB1 with 1000 sampling|0.5s|15%|
|MinMax eval_sampling_10 3|0.1s|20%|
|MinMax eval_pieces 3|0.002s|100%|
|MinMax eval_pieces 4|0.02s|0%|
|MinMax eval_pieces 5|0.1s|100%|
