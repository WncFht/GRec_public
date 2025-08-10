# Unify_Multimodal_Generative_Recommendation

代码的初始结构是基于下面这篇论文的开源代码

> [Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation](https://ieeexplore.ieee.org/abstract/document/10597986/)

## 环境构建
依照requirements.txt参考

可能torch等包的版本需要根据现有硬件的cuda版本来调整

## 整个项目的Pipeline

### 数据集描述

Amazon-2018，共9个类别的数据。

处理后，每个类别下包含序列交互信息，每个物品的图文数据对。文本包含四个字段 （title, description, brand, categories）。图片根据url下载下来。

先提供一个Instruments数据集在[Google Drive](https://drive.google.com/file/d/1xZg4DY-R1eAGsa0DBQUu0IxH2uprLzxl/view?usp=drive_link)上供下载, 下载解压后将其置于data目录下 (注：Google Drive可在服务器上直接用命令下载，不清楚的话可以查阅一些用法。)


### 使用GPT4.1 对 原始数据集的文本内容进行丰富
**已经完成，数据保存在Instruments/Instruments.item_enriched_v2.json中**

这个版本相比于原始版本主要在Prompt强调了需要依照物品原本的图像和文本信息进行文本增强。增强的字段包括
- enhanced_title
- tags
- highlights
- characteristics

To Do:
- 对丰富的文本的本身质量的一个评估：我们需要优先验证这样生成的文本确实补充了物品的一些细节和事实性知识，有利于推荐。因此需要用一个外接的推荐模型来验证。
推荐使用[MMRec](https://github.com/enoche/MMRec)框架来进行实验验证 (这部分我后续先来做)


### 使用Qwen2.5-VL-7B 进行 多模态表征提取

我们优先选用Qwen2.5-VL-7B模型进行物品表征的提取（TIGER中使用的是纯文本信息，而我们将物品的图像，原始文本和增强后的文本统一提取表征）。
> 表征提取的代码存储在data_process/qwen_embeddings.py中，推荐阅读，理解我们提取表征的流程

> **已经完成，数据存储在Instruments/Qwen/Qwen2.5-VL-7B-Instruct_rep.npy中**


### 使用RQ-VAE对上一步提取的表征进行离散化处理

**关于RQ-VAE的所有代码运行都在index/文件夹中**，已经生成的一版tokens 存储在 Instruments/Instruments.index_qwen7B.json。

在这个版本中，没有对Token ID 碰撞做特别的处理，因此还是有少量的不同物品被分配到了同样的tokens下

**请务必熟悉其离散化的代码和流程，这里需要跑通，后续也需要对离散化的方式等做消融和对比实验！！！（优先做）**

已跑通代码，需要后续仔细阅读

### Qwen-VL Finetuning
####  微调细节

对于每个数据集微调一个LoRA Adapter.

微调过程中仅更新下列两类参数

- LoRA 微调，仅更新Attention Projection and FFN
- 冻结原始的Word Embeddings, 仅更新每个数据集中物品对应的新添加的Token ID Emebddings.

训练完成后仅保存LoRA Adapter, Extra Token Embeddings.

#### 已经实现的部分
已经实现的版本参见主目录下multimodal_finetune_lora.py 以及 multimodal_finetune_lora.sh 

- 没有加_lora的是全量微调的版本，资源耗费大，且数据量小的时候太容易过拟合，暂时不考虑。
- 加了_dis的分布式微调的版本

#### 多任务学习
仔细阅读代码可以看到，目前的微调涉及到四个任务
- seqrec,
- mmitem2index,
- mmindex2item,
- mmitemenrich

阅读相关的代码即可知道对应什么任务，所有任务的Prompt设计都在prompt.py中

**To Do** :
- 首先务必认真读懂并跑通lora微调的代码，务必读懂数据分配机制（collotor.py中的代码也需要读懂）。在资源量较小的情况下，优先微调一个Qwen2.5-VL-3B-Instruct的LoRA版本 （优先做）
- 调研多阶段的模型训练范式 
- 调研Model Merge: 加权平均LoRA参数

### 测评benchmark的搭建

#### 物品序列推荐的任务
目前完成的部分在multimodal_test.sh中，可以完成LoRA的加载以及序列推荐任务的评估。

To Do:
- 在上一步的基础上，请务必跑通multimodal_test.sh, 并合理测试出目前微调模型在序列推荐任务上的表现，需要有结果（优先做）
- 跑出三个基线（TIGER， LC-Rec）在Instruments这个数据集中的序列推荐表现 （论文应该都看了）
    1. TIGER 和 LC-Rec 仅用到文本特征, 因此统一使用Instruments/Insturments.emb-llama-td.npy 特征即可，需要进一步用RQ-VAE量化以及生成新的Token IDs (直接使用Index里面提供的代码版本就好) （优先做）
    已完成
    2. MQL4Rec 提供了多模的Token IDs索引，也不用再生成了，直接用就好。（一个潜在的问题是，其每个物品有4个文本tokens + 4个图像tokens, 本质上有8个tokens, 严格意义上并不算公平比较，可以先放一放）
    2. TIGER 和 MQL4Rec 都是微调T5, 这部分代码需要迁移到我们的项目中来，建立一个子目录来完成 （优先做）


#### 文本生成任务的评测
目前已经完成的部分在 text_generation 中，还很粗糙。

文本生成任务主要是评估模型依据原始的图片和文本来生成新的文本的能力，对应于我们的text_enrich任务。新的文本包括为物品生成
enhanced_title，tags，highlights，characteristics。在最开始用GPT4.1生成的内容就作为我们的标签数据来评测。

To Do:
- 评估我们的模型：与上一个评测类似，使用同样的加载方式加载模型，然后需要设计一个文本丰富任务的测试集，通过合理的Prompt来测试生成的文本的质量 （评估指标见text_generation/evaluate.py） （优先做）
- 还需要引入一些基线作为对比。可以考虑的基线有:
原始的没有微调的QwenVL, BLIP2, InstructBLIP等，这些模型不需要训练，仅需要完成相应的加载和生成文本的脚本，将对应的生成文本保存下来即可。(优先做)
即 prompt.py 中的 Task9 Textencrichment 任务。
这里具体的评测就是，用 GPT4.1 生成的文本当作 Ground Truth / 标签，然后使用我们微调的模型来生成文本，然后计算BLEU, ROUGE等指标。

这里要能够有几个模型加载，一个是我们自己微调的模型，一个是其他的多模态大语言模型。

#### Embedding 质量的评测 （放到后面再做吧）
我们需要对模型最终学到的Embedding的质量进行评测。
每个物品对应若干个Token IDs. 这些Token IDs在LoRA微调阶段被不断地更新，融合学习了推荐的协同信号和多模态的语义信息。因此理论上，每个物品的若干Token IDs的信息聚合即可作为其的Embedding表征，供下游使用。

To Do:
- 在LoRA微调模型后，物品对应的Token ID Embeddings也会存储在adapter_model.safetensors中。在加载模型之后，我们可以根据每个物品对应的Token IDs 提取出其相应的Token Embeddings。目前先考虑Average操作来生成物品的Embeddings.（优先做）

- 得到Embeddings后，构建I2I的评估框架。

