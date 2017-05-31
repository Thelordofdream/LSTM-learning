# LSTM-learning
A simple primer of Long Short-Term Memory network and Natural Language Processing
## Word Embedding
人类的自然语言文本，本质上是人类能够处理并理解的视觉符号，但是它们并不能直接被机器处理，因而自然语言处理的首要工作就是将自然语言文本处理成机器能够处理的表示形式。  
对于机器而言，所谓处理的本质就是数据的运算，因而人们通过各种词嵌入模型将自然语言中的词汇映射成“词向量”，再将其输入机器中进行处理。
### one-hot 模型
one-hot模型是最简单的词嵌入模型，可将文本中的词汇映射成数值编码。 该模型的处理流程如下：

* 统计所有待处理文本中出现的单词，来构建词汇表（全集数量为n）并做编号
* 将词汇表中的单词进行稀疏编码——用一个第j（词汇的编号j）维为1、其他维为0的n维向量表征

如对词汇库{A, B, C, D}做稀疏编码：

* A -> (1, 0, 0, 0)
* B -> (0, 1, 0, 0)
* C -> (0, 0, 1, 0)
* D -> (0, 0, 0, 1)

那么，文本“ACBD”的ont-hot表征就是：
\\begin{bmatrix}
1 & 0 & 0 & 0 \\\
0 & 0 & 1 & 0 \\\
0 & 1 & 0 & 0 \\\
0 & 0 & 0 & 1 \\\
\\end{bmatrix}
因为该模型中词向量中只有一位数为1，因而得名“one-hot”。但one-hot模型存在两大问题：

* 词向量的数值不包含词汇的语义信息，也无法刻画词与词之间的关系
* 在解决某些任务的时候（比如构建语言模型），稀疏表示法会造成维数灾难

由于原因一，one-hot模型丢失了大量的语义信息，对较为复杂的自然语言处理处理问题的表现并不理想。
### 词袋模型
词袋模型（Bag of Words, BoW）也是一种传统简单的词嵌入模型，该模型主要依据文本中词汇出现的频次。该模型的处理流程如下：

* 通过统计所有待处理文本中出现的单词构建词汇表
* 对词汇表中每个单词在文本中出现的频次进行统计
* 用单词在文本中出现的频次表征单词
* 将一整个文本映射成由词频组成文本向量，称之为“词袋”

如对以下文本进行词嵌入：

* John likes to watch movies. Mary likes too.
* John also likes to watch football games.通过统计构建词汇表如下：  
<div align='center'><table>
<tr><td>to</td><td>football</td><td>games</td><td>likes</td><td>movies</td><td>also</td><td>John</td><td>watch</td><td>Mary</td><td>too</td></tr>
</table></div>
那么，通过统计词频表征词汇，文本的词袋表征就是：
\\begin{bmatrix}
1 & 2 & 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1
\\end{bmatrix}
\\begin{bmatrix}
1 & 1 & 1 & 1 & 0 & 1 & 1 & 1 & 0 & 0
\\end{bmatrix}
词袋模型存在两大优势：

* 文本向量长度只与文本长度有关而与词汇库大小无关，因而不存在维度爆炸的问题
* 词袋模型很大程度上压缩了文本表示的维数同时又保留了相当多的信息，使得简单的回归模型都能在文本分类等问题上有很好的表现

但是词袋模型丢失了词汇的时序信息，同时也无法刻画词汇本身的含义及词汇与词汇间的关系，所以在文本问答、文本理解等复杂的自然语言处理问题上，词袋模型的效果并不理想。
### 基于概率的词嵌入模型
概率模型的基本思路是，通过词汇与之前词汇间的相互非确定性的概率关系刻画其向量表示，将词汇映射成低维的词向量。  
其中最为流行的概率模型有C&W 2008、M&H 2008和Mikolov 2010。而这里介绍一个最经典的模型，也就是Bengio等人在《A Neural probabilistic language model》中提出的一种用神经网络构建二元语言模型的框架。文章中提出了一个三层神经网络的来构建语言模型，结构如下所示：

### Google Word2Vec模型
## Recurrent Memory Network
### FNN
### RNN
### LSTM
#### BasicLSTMCell
### Bi-Directional RNN
### Attensive Reader
#### LSTMCell
