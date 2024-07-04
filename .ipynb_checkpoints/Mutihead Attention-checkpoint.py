import torch
import torch.nn as nn
import torch.functional as F
import math
# 输入模型超参数
batch_model = 2
time_model = 3
d_model = 4
n_head = 4
# 随机创建输入, qkv
X = torch.randn(batch_model, time_model, d_model)
X
# 创建模型结构
# 问题1：前面这一这段代码的含义是什么，python相关
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        # 注意这里super要传入自己和self，不能只传入自己，你要多学一些python的基础知识了
        super(MultiHeadAttention, self).__init__()
        
        # 初始化超参数
        self.n_head = n_head
        self.d_model = d_model
        # 创建qkv的三个线性映射层，用一个矩阵和qkv相乘，让qkv变得可学习
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 问题2：这个combine的作用是什么，矩阵经过怎样的运算
        self.combine = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, q, k, v):
        batch, time, dimension = q.shape
        # 得到每个头的dimension
        n_d = d_model // n_head
        # 让qkv进入线性层
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 拆分qkv
        # 问题3:为什么需要permute这个东西： 
        # 问题4:这个转置函数怎么用的，结果是什么
        # 问题5:这两个四维矩阵是怎么乘起来的
        # 为了更好的并行计算，变换之后，qkv矩阵形状变成(batch, n_head, time, n_d)，之后k要再次变换成(batch, n_head, n_d, time)
        # 最后q @ k结果的shape是(batch, n_head, time, time)两个四维矩阵相乘，前面两个维度都不参与运算。
        q = q.view(batch, time, n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, n_head, n_d).permute(0, 2, 1, 3)
        # 都是交换维度，只是交换维度的数量区别
        print(k.shape)
        print('------------')
        print(k.permute(0, 1, 3, 2).shape)
        print(k.transpose(2, 3).shape)
        # 计算注意力结果 q*k转置/n_d
        # score的shape为(batch, n_head, time, time)
        score = q @ k.transpose(2,3) / math.sqrt(n_d)
        # 生成mask，一个左下角为1的下三角矩阵
        mask = torch.tril(torch.ones(time, time, dtype=bool))
        # 让mask掩盖score中不应该被注意的数值,全部变成负无穷
        # 问题6：这个masked_fill函数是哪里的，矩阵自带的吗，怎么用， pytorch张量自带的方法，用于用特定值填充掩码位置。
        # 为什么需要combine
        # 在多头注意力机制中，将多个头的输出重新组合成一个张量后，还需要通过一个线性变换来生成最终的输出。这是因为每个头的输出仅仅代表该头的注意力结果，最后通过线性层来融合各个头的信息，生成最终的表示。
        # 1. 线性变换：尽管 permute 和 view 将多个头的输出重新组合成了一个张量，但是每个头的输出还需要进一步线性变换，以便在融合信息的同时对其进行适当的缩放和变换。
        # 2. 参数化：combine 层通过线性层实现参数化，使得整个模型能够学习如何将多个头的注意力结果组合成最终的输出表示。这是模型学习的关键部分。
        # 3. 增加模型表达能力：线性层通过增加可学习参数，提升了模型的表达能力，使其能够更好地捕捉复杂的模式和关系。
        score = score.masked_fill(mask == 0, float('-inf'))
        # 经过softmax然后@v, 点乘之后最后两个维度相乘，又变成(batch, n_head, time, n_d)了
        score = self.softmax(score) @ v

        # 把score的形状变换回来
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        # combine score
        # 问题7：这个combine的作用是什么
        output = self.combine(score)
        return output
attention = MultiHeadAttention(d_model, n_head)
output = attention(X, X, X)
print(output, output.shape)