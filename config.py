import os

cws_label2id = {'O': 0, 'B': 1, 'I': 2}
cws_num = len(cws_label2id)
label_list = []
tag2id = {'O': 0}
for i, label in enumerate(label_list):
    tag2id['B-'+label] = len(tag2id)
    tag2id['I-'+label] = len(tag2id)
id2tag = {id: tag for tag, id in tag2id.items()}
num_labels = len(tag2id)

embedding_dim = 300
rnn_dim = 256
learing_rate = 1e-3
learing_rate_min = 1e-6
epochs = 100
batch_size = 256

data_dir = 'data'
train_file = os.path.join(data_dir, 'train_data')
test_file = os.path.join(data_dir, 'test_data')

vocab = os.path.join(data_dir, 'vocab.pickle')

test_result_data_dir = 'test_result_data'

model_save_dir = 'model_save'
best_model_name = 'best_model.weights'
end_model_name = 'end_model.weights'

stop_dela = 10 # 当整句准确率连续10轮没有上升时 就停止训练