# (1)コンパイラ
CC  = nvc++
# (2)コンパイルオプション
CFLAGS    =
# (3)実行ファイル名
TARGET  = Reversi
# (4)コンパイル対象のソースコード
SRCS    =  main.cpp
SRCS 	+= minimax.h
SRCS 	+= board.h
SRCS 	+= ucb.h
SRCS 	+= uct.h
SRCS 	+= basic.h
SRCS 	+= MonteCarloGPU.cuh
SRCS 	+= MonteCarloGPU.cu
# (5)オブジェクトファイル名
OBJS    = $(SRCS:.cpp=.o)

# (6)インクルードファイルのあるディレクトリパス
INCDIR  = 

# (7)ライブラリファイルのあるディレクトリパス
LIBDIR  = 

# (8)追加するライブラリファイル
LIBS    = 

# (9)ターゲットファイル生成
$(TARGET): $(OBJS)
    $(CC) -o $@ $^ $(LIBDIR) $(LIBS)

# (10)オブジェクトファイル生成
$(OBJS): $(SRCS)
    $(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

# (11)"make all"で make cleanとmakeを同時に実施。
all: clean $(OBJS) $(TARGET)
# (12).oファイル、実行ファイル、.dファイルを削除
clean:
    -rm -f $(OBJS) $(TARGET) *.d
