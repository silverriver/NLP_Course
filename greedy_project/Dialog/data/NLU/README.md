# Data for the NLU project

本目录包含了用于构建NLU模型的数据集。
数据来源是SMP2019 ECDT中文人机对话技术测评任务一自然语言理解。包含约2.5K个对话，29个意图和63个槽位。这份部分的原始数据见`raw_train.json`文件
并且还混入了一些开放领域对话数据。这部分的原始数据见`open_domain.txt`文件。

运行`processing.py`脚本后，会生成若干个文件：

- `train.txt`: 包含训练数据
- `dev.txt`: 包含验证数据
- `test.txt`: 包含测试数据
- `word_vocab.txt`: 包含词汇表
- `intent_vocab.txt`: 包含意图表
- `slot_vocab.txt`: 包含槽位表
- `ECDT2019.txt`: 临时文件

NLU项目中用到了`train.txt`，`dev.txt`，`test.txt`三个文件。
