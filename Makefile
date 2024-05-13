PROG := TARGET_NAME
SRCS := $(wildcard *.cpp) $(wildcard *.h) $(wildcard *.cu) $(wildcard *.cuh)
OBJS := $(SRCS:%.cpp=%.o) # %マクロを用いて置換 SRC配列を元に"<ファイル名>.o"の配列を作る。 中間オブジェクトファイル用。
DEPS := $(SRCS:%.cpp=%.d) # %マクロを用いて置換 SRC配列を元に"<ファイル名>.d"の配列を作る。 依存ファイル用。

# 各種設定を変数として定義
CC := nvc++
CCFLAGS := 
INCLUDEPATH := -I/usr/local/include
LIBPATH := -L/usr/local/lib
LIBS := -framework Cocoa -framework OpenGL -lz -ljpeg -lpng

# これが主レシピ
all: $(DEPENDS) $(PROG)

# リンク
$(PROG): $(OBJS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LIBPATH) $(LIBS)

# コンパイル
.cpp.o:
	$(CC) $(CCFLAGS) $(INCLUDEPATH) -MMD -MP -MF $(<:%.cpp=%.d) -c $< -o $(<:%.cpp=%.o)

# "make clean"でターゲットと中間ファイルを消去できるようにする
.PHONY: clean
clean:
	$(RM) $(PROG) $(OBJS) $(DEPS)

-include $(DEPS) # include "ファイル名" でそのファイルの内容をここにコピペしたのと同じ効果を得られる
