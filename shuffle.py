import random
from collections import Counter

classes = [
    '种植业',
    '渔业',
    '林业',
    '饲料',
    '农产品加工',
    '畜禽养殖',
    '动物保健',
    '石油开采',
    '煤炭开采',
    '采掘服务',
    '石油化工',
    '化学原料',
    '化学制品',
    '化学纤维',
    '塑料',
    '橡胶',
    '钢铁',
    '金属非金属新材料',
    '半导体',
    '元件',
    '光学光电子',
    '电子制造',
    '汽车整车',
    '汽车零部件',
    '汽车服务',
    '其他交运设备',
    '白色家电',
    '视听器材',
    '饮料制造',
    '食品加工制造',
    '纺织制造',
    '服装家纺',
    '造纸',
    '包装印刷',
    '家用轻工',
    '其他轻工制造',
    '化学制药',
    '中药',
    '生物制品',
    '医药商业',
    '医疗器械',
    '医疗服务',
    '电力',
    '环保工程及服务',
    '港口',
    '高速公路',
    '公交',
    '航空运输',
    '机场',
    '航运',
    '铁路运输',
    '物流',
    '房地产开发',
    '园区开发',
    '贸易',
    '景点',
    '旅游综合',
    '银行',
    '证券',
    '保险',
    '多元金融',
    '通用机械',
    '计算机设备',
    '计算机应用',
    '通信运营',
    '通信设备'
]

hibor_dict = [
    '医药生物',
    '房地产开发',
    '电气设备',
    '通信设备',
    '计算机应用',
    '传媒',
    '信息服务',
    '汽车',
    '饮料制造',
    '电子',
    '化工',
    '通用机械',
    '银行',
    '金融服务',
    '新能源',
    '有色金属',
    '建筑建材',
    '航空航天军工',
    '家用电器',
    '环保工程及服务',
    '农林牧渔',
    '餐饮旅游',
    '其他轻工制造',
    '交通运输',
    '公用事业',
    '钢铁',
    '纺织服装',
    '煤炭开采',
    '新能源汽车及整车',
    '消费行业（综合）',
    '石油化工',
    '证券',
    '零售',
    '网络服务',
    '教育与服务',
    '贸易',
    '装饰园林',
    '商业贸易',
    '化学制品',
    '保险',
    '半导体',
    '金属非金属新材料',
    '生物制品',
    '电力',
    '高端装备制造',
    'TMT（综合）',
    '物流',
    '化工新材料',
    '光伏',
    '通信运营',
    '智能装备制造',
    '水泥制造',
    '医疗服务',
    '汽车零部件',
    '信息设备',
    '有色金属冶炼与加工',
    '包装印刷',
    '工程机械',
    '航空运输',
    '综合',
    '家用轻工',
    '酒店与餐饮',
    '医疗器械',
    '燃气水务',
    '石油开采',
    '多元金融',
    '建筑工程',
    '食品加工制造',
    '汽车整车',
    '造纸',
    '采掘',
    '电子制造',
    '中药',
    '港口',
    '其他通用机械',
    '种植业',
    '元件',
    '玻璃制造',
    '化学制药',
    '风电',
    '光学光电子',
    '旅游综合',
    '畜禽养殖',
    '白色家电',
    '动力电池及材料',
    '动物保健',
    '航运',
    '服装家纺',
    '化学原料',
    '医药商业',
    '其他建材',
    '橡胶',
    '采掘服务',
    '其他交运设备',
    '高速公路',
    '塑料',
    '计算机设备',
    '农产品加工',
    '化学纤维',
    '其他电子',
    '铁路与轨交设备',
    '汽车服务',
    '能源通用机械',
    '配套基础设施',
    '纺织制造',
    '机场',
    '铁路运输',
    '建材',
    '核电',
    '机床工具',
    '博彩业',
    '船舶与海工装备',
    '公交',
    '视听器材',
    '景点',
    '渔业',
    '饲料',
    '药用辅料',
    '生物质能',
    '林业',
    '工业服务',
    '园区开发',
    '机械基础件',
    '储能电站与设备'
]


def shuffle():
    out = open("./data/industry/train.txt", 'w')
    lines = []
    with open("./data/industry/title.txt", 'r') as infile:
        for line in infile:
            lines.append(line)
    random.shuffle(lines)
    for line in lines:
        out.write(line)


def get_label_count():
    lines = []
    with open("./data/industry/train.txt", 'r') as infile:
        for line in infile:
            if line.split("\t")[0] in classes:
                lines.append(line.split("\t")[0])
    word_counts = Counter(lines)
    print(word_counts)


def get_names():
    for item in hibor_dict:
        if item in classes:
            print(item)


if __name__ == '__main__':
    get_label_count()
    # get_names()
