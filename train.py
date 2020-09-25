import numpy as np
import os
import sys
import keras
from copy import deepcopy
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from data import load_data, DataGenerator
from model import load_model
from config import train_file, test_file, batch_size, epochs, tag2id, id2tag, model_save_dir, test_result_data_dir, best_model_name, end_model_name, learing_rate_min, vocab, cws_label2id, stop_dela
# 区分模型和测试结果保存的目录
_time = sys.argv[1]
# 使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 模型和测试结果保存的目录
this_model_save_dir = os.path.join(model_save_dir, _time)
this_test_result_dir = os.path.join(test_result_data_dir, _time)
os.makedirs(this_model_save_dir, exist_ok=True)
os.makedirs(this_test_result_dir, exist_ok=True)


# 标注数据
print('start ######load train data #####')
train_data, train_char_set = load_data(train_file)
print('train data len： ' + str(len(train_data)))
print('end ######load train data #####')
print('start ######load test data #####')
test_data, test_char_set = load_data(test_file)
print('test data len： ' + str(len(test_data)))
print('end ######load test data #####')

train_char_set = train_char_set.union(test_char_set)
tokenizer = Tokenizer(num_words=None, char_level=True, filters='', lower=False)
tokenizer.fit_on_texts(''.join(list(train_char_set)))  # 对所有的字进行编码
word_index = tokenizer.word_index
# print(word_index)
with open(vocab, 'wb') as handle:
    pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)


# load model
print('start #######load model#####')
model = load_model(len(word_index)+4)
print('end ######load model #####')
# 建立分词器
# tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_labels, batch_cws_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels, cws_labels = [len(word_index)+2], [0], [0]
            for w, l in item:
                token_ids.append(word_index.get(w, 0))
                labels.append(tag2id[l])
                if l.startswith('B'):
                    cws_labels.append(cws_label2id['B'])
                elif l.startswith('I'):
                    cws_labels.append(cws_label2id['I'])
                else:
                    cws_labels.append(cws_label2id['O'])
            token_ids.append(len(word_index)+3)
            labels.append(0)
            cws_labels.append(0)
            batch_token_ids.append(token_ids)
            batch_labels.append(labels)
            batch_cws_labels.append(cws_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_labels = pad_sequences(batch_labels)
                batch_cws_labels = pad_sequences(batch_cws_labels)
                batch_labels = np.expand_dims(batch_labels, 2)
                batch_cws_labels = np.expand_dims(batch_cws_labels, 2)
                # print(batch_token_ids)
                yield batch_token_ids, [batch_cws_labels, batch_labels]
                batch_token_ids, batch_labels, batch_cws_labels = [], [], []


def predict(text):
    """
    预测函数
    :param text: str
    :return: list B- I- O
    """
    token_ids = [word_index.get(s, 0) for s in text]
    token_ids = [len(word_index)+2] + token_ids + [len(word_index) + 3]
    token_ids = pad_sequences([token_ids])
    result = model.predict(token_ids)[1][0]
    result = [np.argmax(i) for i in result[1:-1]]
    tags = [id2tag[i] for i in result]
    return tags


def evaluate(data):
    """
    评测函数
    :param data: list 测试的数据
    :return: int, list 全句正确的条数和测试的结果
    """
    true_num = 0
    test_result = deepcopy(data)
    for _i, d in enumerate(deepcopy(data)):
        ture_tags = [l[1] for l in d]
        text = ''.join([l[0] for l in d])
        pred_tags = predict(text)
        true_num += 1 if ture_tags == pred_tags else 0
        for i_l, dl in enumerate(test_result[_i]):
            dl.append(pred_tags[i_l])
    return true_num, test_result


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_all_acc = 0
        self.delate = 0

    def on_epoch_end(self, epoch, logs=None):
        # 每两个epoch 学习率变为当前的一半
        if epoch % 3 == 1:
            current_lr = float(K.get_value(model.optimizer.lr))
            if current_lr >= learing_rate_min:
                K.set_value(model.optimizer.lr, current_lr/2)
        true_num, test_result = evaluate(test_data)
        all_true = true_num/len(test_data)
        result = []
        for t in test_result:
            tmp_l = [d[0] + '\t' + d[1] + '\t' + d[2] + '\n' for d in t] + ['\n']
            result.extend(tmp_l)
        with open(os.path.join(this_test_result_dir, str(epoch) + '_' + str(true_num)), 'w', encoding='utf8') as f:
            f.writelines(result)
        # 保存最优
        if all_true >= self.best_all_acc:
            self.delate = 0
            self.best_all_acc = all_true
            model.save_weights(os.path.join(this_model_save_dir, best_model_name))
        else:
            self.delate += 1
            if self.delate >= stop_dela:
                model.stop_training = True
        print(
            'all entity acc: %.5f, best all ture acc: %.5f\n' %
            (all_true, self.best_all_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    model.save_weights(os.path.join(this_model_save_dir, end_model_name))
