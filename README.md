# NER_ToolBox

| MODEL            | Task    |    F1    |
| ---------------- | ------- |--------- |
| **RoBERTa-wwm**  | CLUENER |   0.60   | 


## Todo
- [x] BERT 
- [x] BERT-CRF
- [ ] BiLSTM-CRF
- [ ] BiGRU-CRF
- [ ] BiLSTM-CNN-CRF
- [ ] BERT-BiLSTM-CRF
- [ ] BERT-MRC
- [ ] CRF
- [ ] HMM
- [ ] ...

## Reference
### data
* [CLUENER 细粒度命名实体识别](https://github.com/CLUEbenchmark/CLUENER2020)

数据分为10个标签类别，分别为: 
地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记）。地址是标记尽量完全的, 标记到最细。
书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。
公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。
游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。
政府（government）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。
电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。
姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。
组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。
职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。
景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。

训练集：10748
验证集集：1343

### article
* [你的CRF层的学习率可能不够大](https://kexue.fm/archives/7196)