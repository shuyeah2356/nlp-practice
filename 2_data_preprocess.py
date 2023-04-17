import d2l.torch as d2l
import re

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + './timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
# print(f'# 文本总行数: {len(lines)}')
# print(lines[0])
# print(lines[10])
print(lines)



# 词元化
def tokenize(lines, token="word"):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        return("未知词元类型"+token)

token_res = tokenize(lines)
print(token_res)

# 词元的类型是字符串，而模型需要数字类型的输入，需要构建一个字典(词表vocabulary)，将字符串类型的词元映射到0开始的数字索引中。
# 首先将训练集中的所有词元合并到一起，对它们的唯一词元进行统计，得到的统计结果称之为语料(corpus)。
# 根据每一个唯一词元出现的频率，为其分配一个数字索引，很少出现的词元被移除，可以降低复杂性
# 语料库中不存在或者已经被删除的词元将映射到一个特定的未知词元“<unk>”,