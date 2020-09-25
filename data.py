'''
@author: rxy
@desc: 加载训练和测试数据，对数据的中空格转为龕，训练的数据超过max_len做切割
@date: 2020/09/22
@version 0.1

'''
import numpy as np

class DataGenerator(object):
    """数据生成器模版
    """
    def __init__(self, data, batch_size=32, buffer_size=None):
        self.data = data
        self.batch_size = batch_size
        if hasattr(self.data, '__len__'):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None
        self.buffer_size = buffer_size or batch_size * 1000

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        """采样函数，每个样本同时返回一个is_end标记
        """
        if random:
            if self.steps is None:

                def generator():
                    caches, isfull = [], False
                    for d in self.data:
                        caches.append(d)
                        if isfull:
                            i = np.random.randint(len(caches))
                            yield caches.pop(i)
                        elif len(caches) == self.buffer_size:
                            isfull = True
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)

            else:

                def generator():
                    indices = list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]

            data = generator()
        else:
            data = iter(self.data)

        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self):
        while True:
            for d in self.__iter__(True):
                yield d


def load_data(filename):
    D = []
    char_set = set()
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        split_text = '\r\n' if '\r\n' in f else '\n'
        for l in f.split(split_text+split_text):
            if not l:
                continue
            d = []
            for c in l.split(split_text):
                if not c:
                    continue
                char, label = c.split('\t')
                d.append([char, label])
                char_set.add(char)
            if d:
                D.append(d)
    return D, char_set

# def sequence_padding(inputs, length=None, padding=0):
#     """Numpy函数，将序列padding到同一长度
#     """
#     if length is None:
#         length = max([len(x) for x in inputs])
#
#     pad_width = [(0, 0) for _ in np.shape(inputs[0])]
#     outputs = []
#     for x in inputs:
#         x = x[:length]
#         pad_width[0] = (0, length - len(x))
#         x = np.pad(x, pad_width, 'constant', constant_values=padding)
#         outputs.append(x)
#
#     return np.array(outputs)


