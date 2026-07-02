AG News 少样本文本分类实验说明

1. 实验目的

在 AG News 新闻标题与摘要数据集上验证 Prompt Learning（提示学习） 在少样本（Few-shot）场景下的有效性。给定一段新闻文本，要求模型从 World、Sports、Business、Sci/Tech 四个类别中正确判断其所属类别。实验设置 5-shot、10-shot、20-shot 三种训练数据规模，旨在观察预训练语言模型在极少标注样本下的分类性能，并探索 Prompt 模板与标签词映射（Verbalizer）对结果的影响。

2. 数据集

采用 AG News 数据集（来自 fast-ai 官方源），包含 4 个类别：

0: World（世界新闻）
1: Sports（体育新闻）
2: Business（商业新闻）
3: Sci/Tech（科技新闻）

原始数据集划分：

(1)训练集：120,000 条
(2)测试集：7,600 条

在 Few-shot 设置下，我们分别从每个类别中随机抽取 k 条（k=5, 10, 20）作为训练集，总计 4×k 条训练样本，测试集保持不变（全量 7,600 条）。

数据以 CSV 格式存放，每行包含 label、title、description 三列，实验中将 title 与 description 拼接成 text 字段作为模型输入。

3. 实验结构总览

┌─────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌─────────────┐
│  数据准备   │ → │ Prompt 构造      │ → │ 预训练模型微调 │ → │  测试评估   │
│ (CSV)      │   │ + 标签词映射     │   │ (BERT+MLM)    │   │ (准确率)    │
└─────────────┘   └──────────────────┘   └───────────────┘   └─────────────┘
      ↑                    ↑                       ↑
 本地加载           模板: "This news is about [MASK]."   5/10/20-shot
 (train.csv)        Verbalizer: world/sports/business/tech


整个流程基于 Hugging Face Transformers 库实现，使用 Trainer API 进行微调。

4. 基线方法与改进尝试

4.1 基线方法（v1）

采用 BERT-base-uncased 预训练模型，配合 Prompt Learning 范式：

(1)Prompt 模板："{text} This news is about [MASK]."
(2)Verbalizer（标签词映射）：
   0 → "world"
   1 → "sports"
   2 → "business"
   3 → "tech"
(3)微调参数：
   学习率：2e-5
   Batch size：4
   训练轮数：5
   优化器：AdamW
   随机种子：42
(4)损失函数：遮蔽语言模型（MLM）交叉熵损失，仅对 [MASK] 位置计算梯度。

该方法直接将分类任务转化为完形填空，利用 BERT 对 [MASK] 的预测能力进行分类。由于训练样本极少（最低 20 条），采用较小学习率和较少轮次防止过拟合。

4.2 改进尝试

本实验针对关键超参数做了敏感性分析：

1.Prompt 模板：尝试过 "[MASK] news: {text}"，发现效果相近，说明 BERT 对模板位置不敏感。
2.学习率与轮次：3e-5 配合 10 轮可使 20-shot 准确率提升约 1%，但 5-shot 时反而下降，因此固定 2e-5 + 5 轮作为最终配置。
3.随机种子：固定种子确保结果可复现，但仍存在因采样造成的波动（未做多次平均）。

由于时间限制，未探索更大模型（如 RoBERTa）或更复杂的 Prompt 设计。

5. 实验步骤（完整运行流程）

5.1 数据准备

1. 从 fast-ai 官方源 下载 ag_news_csv.tgz，解压得到 train.csv 和 test.csv。
2. 将两个文件放入项目目录下的 data/ 文件夹。
3. 下载 bert-base-uncased 模型文件（从 Hugging Face 镜像站 下载），放置于 ./bert-base-uncased/，确保包含 config.json、pytorch_model.bin、vocab.txt、tokenizer.json、tokenizer_config.json。

5.2 少样本采样与预处理

1.对每个 shot 设置，分别从 train_data 中按类别过滤，每类随机抽取 k 条，合并成 few_train（共 4k 条）。
2.使用 preprocess_function 将每条样本转化为 Prompt 形式，并标记 [MASK] 位置的正确标签（对应的 token id）。
3.将处理后的数据集传入 Trainer。

5.3 微调与评估

1.每次重新加载预训练模型，确保各 shot 设置相互独立。
2.使用 DataCollatorForTokenClassification（label_pad_token_id=-100）自动对齐序列长度。
3.训练完成后，在全部测试集上评估，函数 evaluate_model 会为每个测试样本生成 Prompt，预测 [MASK] 位置的 token，通过 id_to_label 映射回类别，最后计算准确率。

5.4 结果记录

程序自动打印每个 shot 的测试准确率，并在最后输出汇总表格（如下）。

6. 代码文件说明

本项目主要包含一个核心脚本 fewshot_agnews.py，其关键函数与作用：

函数名 作用
sample_few_shot_data(dataset, k_shot) 从训练集中为每个类别抽取 k 条样本
preprocess_function(examples) 构造 Prompt 并生成带 labels（仅 [MASK] 位置）的 tokenized 数据
evaluate_model(model, tokenizer, test_dataset, batch_size) 在测试集上逐批预测并返回准确率
main (循环) 分别运行 5/10/20-shot，输出结果

此外，DataCollatorForTokenClassification 负责动态填充，Trainer 驱动训练过程。

7. 实验结果与分析

7.1 全局结果

Shot 训练样本总数 测试准确率
5 20 46.91%
10 40 62.57%
20 80 66.70%

基线解读：随机猜测准确率为 25%，5-shot 即达到 46.91%，说明 BERT 通过 Prompt 方式能从极少样本中快速学习分类知识。随着 shot 增加，准确率稳步上升，20-shot 达到 66.70%，验证了更多样本带来性能提升的普遍规律。

7.2 详细分析

(1)5-shot 表现：训练样本仅 20 条，模型难以充分捕捉各类别的语义边界，但已显著优于随机，表明预训练模型具备强大的先验知识。
(2)10-shot → 20-shot 的提升：样本翻倍（40→80），准确率提升约 4.13 个百分点，边际收益递减，提示未来在 50+ shot 时可能趋于饱和。
(3)与传统微调对比：若直接使用 BERT 分类头（[CLS] 线性层）在 20-shot 下准确率通常低于 50%，而 Prompt Learning 借助掩码语言建模，能更充分地发挥预训练知识，因此在本任务中表现更优。

7.3 错误案例观察（以 20-shot 模型为例）

(1)部分体育新闻被误分为 Business（如涉及球队转会费金额的报道）。
(2)科技类新闻中涉及“市场”或“股价”时，容易与 Business 混淆。
(3)短文本（如仅 5 个词）的预测准确率显著低于长文本，说明模型需要更多上下文才能准确判断。

7.4 改进方向与局限

(1)模板优化：尝试多种 Prompt（如 "Category: [MASK]. {text}"）可能进一步提升准确率。
(2)集成方法：多次随机采样取平均，可降低采样方差，使结果更稳定。
(3)模型升级：使用 RoBERTa 或 DeBERTa 等更强预训练模型，有望在同样 few-shot 设置下提升 5~10 个百分点。
(4)动态阈值：当前仅使用 argmax 预测，若引入置信度阈值可减少错误预测。

8. 总结

本实验验证了 BERT + Prompt Learning 在 AG News 少样本分类任务上的有效性。在 5-shot、10-shot、20-shot 下分别取得 46.91%、62.57%、66.70% 的准确率，且随样本增加性能持续提升。该范式无需复杂特征工程，仅通过设计合适的完形填空模板即可激发预训练模型的知识，非常适合标注数据稀缺的场景。

最终推荐配置：

(1)模型：bert-base-uncased
(2)Prompt 模板："{text} This news is about [MASK]."
(3)Verbalizer：world/sports/business/tech
(4)训练参数：lr=2e-5, batch_size=4, epochs=5
(5)最高性能（20-shot）准确率：66.70%

后续工作可聚焦于更优模板设计、命名实体增强及自适应阈值策略。


9. 代码与实验复现

所有代码已上传至 GitHub，
运行方式：

```bash
pip install torch transformers datasets pandas scikit-learn
python fewshot_agnews.py
```

确保 data/ 目录下有 train.csv 和 test.csv，bert-base-uncased/ 包含完整模型文件。程序会自动执行所有 shot 并输出汇总结果。

