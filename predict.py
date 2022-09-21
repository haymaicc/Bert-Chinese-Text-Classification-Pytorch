import torch
from importlib import import_module

key = {
    0: '房地产开发',
    1: '通信设备',
    2: '计算机应用',
    3: '饮料制造',
    4: '通用机械',
    5: '银行',
    6: '环保工程及服务',
    7: '其他轻工制造',
    8: '钢铁',
    9: '煤炭开采',
    10: '石油化工',
    11: '证券',
    12: '贸易',
    13: '化学制品',
    14: '保险',
    15: '半导体',
    16: '金属非金属新材料',
    17: '生物制品',
    18: '电力',
    19: '物流',
    20: '通信运营',
    21: '医疗服务',
    22: '汽车零部件',
    23: '包装印刷',
    24: '航空运输',
    25: '家用轻工',
    26: '医疗器械'
}

model_name = 'bert'
x = import_module('models.' + model_name)
config = x.Config('data')
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))


def build_predict_text(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    ids = torch.LongTensor([token_ids]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    return ids, seq_len, mask


def predict(text):
    """
    单个文本预测
    :param text:
    :return:
    """
    data = build_predict_text(text)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
        print(outputs[0])
        print(outputs[0][num])
        if outputs[0][num] < 6:
            print(key[int(num)])
            return ''
    return key[int(num)]


if __name__ == '__main__':
    print(predict("商业贸易行业跟踪周报2022年第1期：复盘&展望，密尔克卫喜迎新年晨晖"))
