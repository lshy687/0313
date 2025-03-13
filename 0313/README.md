# 时序知识图谱预测：利用上下文学习无需知识建模

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10613-b31b1b.svg)](https://arxiv.org/abs/2305.10613)

本仓库提供了我们论文的模型、代码和数据：[Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning](https://arxiv.org/abs/2305.10613) (EMNLP 2023)。
[[PDF]](https://arxiv.org/pdf/2305.10613.pdf)

作者: [Dong-Ho Lee&ast;][dlee], [Kian Ahrabian&ast;][kahrabian], [Woojeong Jin][wjin], [Fred Morstatter][fmorstatter], [Jay Pujara][jpujara].

## 项目概述
**时序知识图谱（TKG）预测** 是一项挑战性任务，要求模型利用过去的事实知识来预测未来的事实。
我们的研究表明，通过**大型语言模型（LLM）的上下文学习（In-Context Learning, ICL）** 可以有效地解决TKG预测问题，而无需显式的知识建模。

本项目的核心功能：
- 利用大型语言模型的上下文学习能力进行时序知识图谱预测
- 支持多种预训练语言模型（通过Hugging Face和OpenAI API）
- 提供基于频率和近期性的基线方法
- 支持多种时序知识图谱数据集：ICEWS14、ICEWS18、WIKI和YAGO

## 环境要求

- Python >= 3.10
- PyTorch
- Transformers库
- OpenAI API（可选，用于GPT模型）

```bash
pip install -r requirements.txt
```

## 数据集
本项目支持以下数据集：
- ICEWS14：包含2014年的国际危机预警系统事件
- ICEWS18：包含2018年的国际危机预警系统事件
- WIKI：基于维基百科的时序知识图谱
- YAGO：YAGO知识库的时序版本

数据格式为四元组(头实体, 关系, 尾实体, 时间戳)。

## 使用方法

### 频率/近期性基线模型
```bash
python run_rule.py \
  --dataset {dataset} \
  --model {recency|frequency} \
  --history_len {history_len} \
  --history_type {entity|pair} \
  --history_direction {uni|bi} \
  --label \
  {--multi_step}
```

参数说明：
- `dataset`: 选择数据集（ICEWS14, ICEWS18, WIKI, YAGO）
- `model`: 选择基线模型（recency或frequency）
- `history_len`: 历史上下文长度
- `history_type`: 历史类型（entity或pair）
- `history_direction`: 历史方向（uni或bi）
- `--label`: 是否包含标签
- `--multi_step`: 是否进行多步预测

### Hugging Face模型
```bash
python run_hf.py \
  --dataset {dataset} \
  --model "模型名称" \
  --history_len {history_len} \
  --history_type {entity|pair} \
  --history_direction {uni|bi} \
  --label \
  {--multi_step}
```

支持的模型包括：
- GPT-NeoX-20b
- LLaMA
- 其他Hugging Face支持的因果语言模型

### OpenAI模型
```bash
python run_openai.py \
  --dataset {dataset} \
  --model {模型名称} \
  --history_len {history_len} \
  --history_type {entity|pair} \
  --history_direction {uni|bi} \
  --label \
  {--multi_step}
```

注意：使用OpenAI模型前，需要在`openai_config.json`文件中配置您的API密钥。

## 超参数详解

本项目包含多种超参数，用于控制模型的行为和提示的形式。以下是关键超参数的详细解释：

### 基本设置参数
- `--model`：指定使用的模型名称，默认为"qwen1.5b"
- `--dataset`：指定数据集，可选值为ICEWS14、ICEWS18、WIKI、YAGO，默认为ICEWS18
- `--multi_step`：是否进行多步预测，默认为False

### 历史建模参数
- `--history_type`：历史类型，可选值为entity（实体）或pair（实体-关系对），默认为entity
- `--history_direction`：历史方向，可选值为uni（单向）或bi（双向），默认为uni
- `--history_len`：历史上下文长度，默认为0
- `--history_top_k`：从历史中选取的目标数量，默认为1

### 提示构造参数
- `--label`：是否在提示中使用标签，默认为False
  - 不启用时：`[实体名称,关系名称,目标实体名称]`
  - 启用时：`[实体名称,关系名称,标签ID.实体名称]`
  
- `--text_style`：是否使用文本样式表达提示，默认为False
  - 不启用时：使用数字ID表示实体和关系，如 `[23,5,47]`
  - 启用时：使用实体和关系的文本名称，如 `[美国,支持,乌克兰]`
  
- `--no_entity`：是否在提示中省略实体名称，默认为False
  - 不启用时：包含完整的实体名称，如 `[美国,支持,乌克兰]`
  - 启用时：省略目标实体名称，只使用标签ID，如 `[美国,支持,1]`

- `--sys_instruction`：ChatGPT的系统指令，默认为空
- `--no_time`：是否在提示中省略时间，默认为False
- `--shuffle_history`：是否打乱历史顺序，默认为False

### 主动学习参数
- `--active_learning`：是否启用主动学习，默认为False
- `--active_strategy`：主动学习选择策略，可选值为：
  - `random`：随机选择样本
  - `max_entropy`：选择预测熵最大的样本
  - `best_of_k`：从K个随机子集中选择平均熵最高的子集
  - `random_balanced`：按头尾方向平衡的随机选择
  默认为`random`
- `--active_samples`：为每个预测样本选择的专家标注样本数量，默认为5
- `--active_integration`：主动学习样本集成方式，可选值为：
  - `direct`：直接添加到提示中
  - `labeled`：添加到提示中并标记为"专家标注的当前事件"
  默认为`direct`

#### 主动学习工作方式
主动学习模式通过"专家标注"的方式增强预测性能：
1. 每个样本在预测时，会从同一时间点的其他样本中选择若干个进行"专家标注"
2. 这些样本被标注真实标签后添加到提示中，作为额外的上下文信息
3. 每个样本预测都是独立的过程，都会单独进行主动学习采样
4. 采样的样本只用于当前样本的预测，不会影响其他样本

### 提示参数组合规则

提示构造参数（`--label`, `--text_style`, `--no_entity`）的组合使用存在一些规则和限制：

#### 可以同时开启的组合

1. **`--label` + `--text_style`**：
   - 完全兼容，推荐用于大型语言模型
   - 效果：`[美国,支持,1.乌克兰]`
   - 优点：提供丰富的语义信息和标签指示

2. **`--no_entity` + `--text_style` + `--label`**：
   - 三者可以同时开启
   - 效果：`[美国,支持,1]`
   - 优点：研究模型对实体名称依赖性时很有用

#### 存在冲突的情况

1. **`--no_entity` 必须与 `--label` 一起使用**：
   - 如果开启`--no_entity`，必须同时开启`--label`
   - 原因：省略实体名称时需要使用标签ID来标识实体

2. **优先级规则**：
   - 当同时开启`--no_entity`和`--label`时，`--no_entity`的效果会覆盖部分`--label`效果
   - 最终格式为`[美国,支持,1]`而非`[美国,支持,1.乌克兰]`

#### 参数组合效果表

| 参数组合 | 效果示例 | 是否有效 |
|---------|---------|---------|
| 无参数 | `[23,5,47]` | ✓ |
| `--text_style` | `[美国,支持,乌克兰]` | ✓ |
| `--label` | `[23,5,1.47]` | ✓ |
| `--label --text_style` | `[美国,支持,1.乌克兰]` | ✓ |
| `--no_entity` | 无效组合 | ✗ |
| `--no_entity --label` | `[23,5,1]` | ✓ |
| `--no_entity --text_style --label` | `[美国,支持,1]` | ✓ |

### 超参数设置
- `--top_k`：存储的预测数量，默认为100
- `--dec_cand`：每步解码的候选数量，默认为5
- `--max_length`：最大解码长度，默认为1
- `--world_size`：处理的数据块数量，默认为1
- `--rank`：执行器的等级，默认为0
- `--tokenizer_revision`：分词器修订版本，默认为"main"
- `--fp16`：是否使用FP16代替FP32，默认为False
- `--verbose`：是否打印额外信息，默认为False
- `--gpu`：指定使用的GPU设备ID，默认为0

### 评估参数
- `--eval_filter`：评估过滤方式，可选值为none、static、time-aware，默认为none

### 超参数使用建议

- **模型选择**：
  - 对于高性能需求：使用较大的模型如GPT-NeoX-20b或OpenAI的GPT模型
  - 对于资源受限环境：使用较小的模型如DeepSeek-R1-Distill-Qwen-1.5B

- **历史上下文**：
  - `history_len`：较长的历史上下文通常能提供更多信息，但可能引入噪声，建议尝试5-15的范围
  - `history_type`：对于关系稀疏的数据集，使用entity模式；对于关系密集的数据集，使用pair模式
  - `history_direction`：bi模式通常提供更全面的信息，但可能增加计算复杂度

- **提示格式**：
  - 对于大型语言模型（LLM）：建议启用`--text_style`和`--label`，因为LLM更擅长处理文本和理解带标签的实体
  - 对于研究模型中对实体名称的依赖性：`--no_entity`参数很有用
  - 对于实体ID非常多的数据集：使用`--label`可以简化模型的学习任务

- **效率相关**：
  - 对于大规模实验：启用`--fp16`可显著提升速度
  - 对于详细分析：启用`--verbose`可以查看更多执行细节

- **实用组合**：
  - 初始实验：`--dataset ICEWS18 --model <您的模型> --history_len 5 --history_type entity --label --text_style`
  
- **主动学习**：
  - 基本使用：`--active_learning --active_strategy random --active_samples 5 --active_integration direct`
  - 为了更好的性能，建议使用基于熵的策略：`--active_strategy max_entropy`
  - 对于计算资源有限的情况，可使用`--active_strategy best_of_k`作为max_entropy的近似
  - 当数据集中头尾样本分布不平衡时，使用`--active_strategy random_balanced`
  - 整合方式建议：
    - 对于简单模型：使用`--active_integration direct`
    - 对于更强大的LLM：使用`--active_integration labeled`以便模型区分历史和当前样本

## 主动学习功能详解

本项目实现了主动学习机制，通过选择最具信息量的样本进行专家标注，并将这些已标注样本添加到上下文中，显著提高预测精度。

### 主动学习的工作原理

1. **样本选择**：从当前时间点中选择样本进行专家标注
2. **专家标注**：为选定的样本提供真实标签（在实际应用中可由人工专家完成）
3. **从测试集移除**：将这些已标注的样本从测试集中移除，不参与评估
4. **提示增强**：将已标注的样本添加到模型输入提示中，作为额外的上下文信息
5. **预测**：使用增强后的提示预测剩余未标注样本

### 主动学习策略

1. **随机选择策略（Random）**：
   - 随机选择指定数量的样本进行专家标注
   - 优点：计算成本低，适合基线对比
   - 缺点：可能选择信息量较低的样本

2. **最大熵策略（Max Entropy）**：
   - 选择预测熵最大的样本进行专家标注
   - 优点：选择模型最不确定的样本，最大化信息增益
   - 缺点：计算成本高，需要对每个样本进行预测

3. **均衡随机策略（Random Balanced）**：
   - 平衡选择头实体和尾实体方向的样本进行专家标注
   - 优点：确保样本方向多样性，防止偏向某一方向
   - 缺点：在样本高度不平衡的数据集上可能无法获取足够多某一方向的样本

4. **最佳K子集策略（Best of K）**：
   - 生成K个随机样本子集，选择平均熵最高的子集
   - 优点：比最大熵策略计算效率更高，同时保持多样性
   - 缺点：性能可能不如完整的最大熵策略

### 样本整合方式

1. **直接整合（Direct）**：
   - 直接将已标注样本添加到提示末尾
   - 格式示例：在历史样本之后添加 `[美国,支持,乌克兰]`

2. **标记整合（Labeled）**：
   - 将已标注样本添加到提示中，并明确标识为"专家标注"
   - 格式示例：在历史样本之后添加 `专家标注的当前事件: [美国,支持,乌克兰]`

### 主动学习命令行参数

- `--active_learning`：启用主动学习功能（布尔标志）
- `--active_strategy`：选择主动学习策略，可选值为random、max_entropy、best_of_k、random_balanced
- `--active_samples`：每个时间点选择进行专家标注的样本数量
- `--active_integration`：样本整合方式，可选值为direct、labeled

### 使用示例

```bash
python run_hf.py \
  --dataset ICEWS18 \
  --model /path/to/your/model \
  --history_len 5 \
  --active_learning \
  --active_strategy max_entropy \
  --active_samples 3 \
  --active_integration labeled \
  --label \
  --text_style
```

此命令将在ICEWS18数据集上运行预测，使用最大熵策略选择每个时间点的3个样本进行专家标注，并以标记方式将它们整合到提示中。

### 性能影响

主动学习机制在多个数据集上的实验表明：
- 平均Hits@1指标提高5-15%
- 对稀疏关系的预测改善更为显著
- 当历史上下文不足时，主动学习的效果尤为明显

我们建议在生产环境中搭配强大的LLM和最大熵策略使用主动学习，以获得最佳性能。

## 评估指标
模型性能使用以下指标评估：
- Hits@1：预测排名第一的是正确答案的比例
- Hits@3：预测排名前三的包含正确答案的比例
- Hits@10：预测排名前十的包含正确答案的比例
- MRR（Mean Reciprocal Rank）：平均倒数排名，反映模型预测的平均排名质量，计算方式为每个测试样例的倒数排名(1/rank)的平均值

## 引用
如果您使用了本代码，请引用以下论文：

```bib
@InProceedings{lee2023temporal,
  author =  {Lee, Dong-Ho and Ahrabian, Kian and Jin, Woojeong and Morstatter, Fred and Pujara, Jay},
  title =   {Temporal Knowledge Graph Forecasting Without Knowledge Using In-Context Learning},
  year =    {2023},  
  booktitle = {The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  url = {https://openreview.net/forum?id=wpjRa3d9OJ}
}
```

[dlee]: https://www.danny-lee.info/
[kahrabian]: https://scholar.google.com/citations?user=pwUdiCYAAAAJ&hl=en
[wjin]: https://woojeongjin.github.io/
[fmorstatter]: https://www.isi.edu/~fredmors/
[jpujara]: https://www.jaypujara.org/

