# 生成文本对推荐的帮助

验证生成文本对推荐的帮助

## 原始的数据集

保存在 Instruments/Instruments.item.json 中

```json
{
    "0": {
        "title": "Blue Microphones enCORE 100 Studio-Grade Dynamic Performance Microphone.",
        "description": "",
        "brand": "Blue Microphones",
        "categories": "Musical Instruments,Microphones & Accessories,Microphones,Dynamic Microphones,Multipurpose"
    },
    "1": {
        "title": "HDE USB to MIDI Interface Cable Keyboard Synthesizer Drum Pad Instrument MIDI 5 Pin Adapter for Windows PCs.",
        "description": " USB to MIDI Interface is a complete solution for hooking up MIDI enabled keyboards, drum pads, mixers, synthesizers and other instruments to your Windows PC. Plug and Play device makes it easy to get started in home audio production, whether you are recording audio, remixing, creating chiptunes or doing your own podcast. Device is compatible only with 32 bit Windows versions Vista / 7 / 8 / 10 | NOT compatible with Mac operating systems including OSX and macOS Troubleshooting Tips:.",
        "brand": "HDE",
        "categories": "Musical Instruments,Live Sound & Stage,Stage & Studio Cables,MIDI Cables"
    },
```

## 什么是增强的文本

使用GPT4.1 对 原始数据集的文本内容进行丰富
**已经完成，数据保存在Instruments/Instruments.item_enriched_v2.json中**

这个版本相比于原始版本主要在Prompt强调了需要依照物品原本的图像和文本信息进行文本增强。增强的字段包括

- enhanced_title
- tags
- highlights
- characteristics

```json
{
  "0": {
    "title": "Blue Microphones enCORE 100 Studio-Grade Dynamic Performance Microphone.",
    "description": "",
    "brand": "Blue Microphones",
    "categories": "Musical Instruments,Microphones & Accessories,Microphones,Dynamic Microphones,Multipurpose",
    "enhanced_title": "Blue Microphones enCORE 100 Studio-Grade Dynamic Microphone",
    "tags": [
      "Blue Microphones",
      "enCORE 100",
      "dynamic microphone",
      "studio microphone",
      "performance microphone",
      "musical instruments",
      "handheld microphone",
      "multipurpose",
      "professional audio"
    ],
    "highlights": [
      "Studio-grade dynamic microphone for versatile performance",
      "Designed for professional audio clarity",
      "Multipurpose use suitable for various musical applications"
    ],
    "characteristics": [
      "Brand: Blue Microphones",
      "Model: enCORE 100",
      "Type: Dynamic Microphone",
      "Handheld design",
      "Studio-grade construction"
    ],
    "has_image": true
  },
  "1": {
    "title": "HDE USB to MIDI Interface Cable Keyboard Synthesizer Drum Pad Instrument MIDI 5 Pin Adapter for Windows PCs.",
    "description": " USB to MIDI Interface is a complete solution for hooking up MIDI enabled keyboards, drum pads, mixers, synthesizers and other instruments to your Windows PC. Plug and Play device makes it easy to get started in home audio production, whether you are recording audio, remixing, creating chiptunes or doing your own podcast. Device is compatible only with 32 bit Windows versions Vista / 7 / 8 / 10 | NOT compatible with Mac operating systems including OSX and macOS Troubleshooting Tips:.",
    "brand": "HDE",
    "categories": "Musical Instruments,Live Sound & Stage,Stage & Studio Cables,MIDI Cables",
    "enhanced_title": "HDE USB to MIDI Interface Cable – 5 Pin Adapter for Keyboard, Synthesizer, and Drum Pad, Compatible with Windows PCs",
    "tags": [
      "HDE",
      "USB to MIDI",
      "MIDI Interface Cable",
      "5 Pin Adapter",
      "Keyboard",
      "Synthesizer",
      "Drum Pad",
      "Windows PC",
      "Plug and Play",
      "Stage & Studio Cable"
    ],
    "highlights": [
      "Easily connect MIDI-enabled instruments to Windows PCs",
      "Plug and play functionality for quick setup",
      "Supports MIDI keyboards, synthesizers, and drum pads",
      "Ideal for home audio production, remixing, and recording",
      "Clear labeling for MIDI IN and OUT connections"
    ],
    "characteristics": [
      "Connection Type: USB to 5-pin MIDI connector",
      "Compatibility: 32-bit Windows Vista / 7 / 8 / 10 only",
      "Cable End Types: USB Type-A, MIDI IN, MIDI OUT",
      "Device Type: MIDI interface cable",
      "System Requirement: Not compatible with Mac operating systems (OSX, macOS)"
    ],
    "has_image": true
  },
```

## 图片在哪里

图片一般保存在 `Instruments/images` 中

```python
# 遍历所有物品
for num_id, item_data in tqdm(
    item_info.items(), desc="Processing items"
):
    num_id = int(num_id)
    item_id = id2item[num_id]
    # 加载图片
    image_path = os.path.join(image_dir, f"{item_id}.jpg")
```

## 一些映射关系

```Instruments.item2id
B002SQJL9U	0
B00D3QFHN8	1
B00I0RHU8K	2
B0002ZPLP2	3
B004LWH79A	4
B00E87OK1G	5
B00R6OT6GW	6
B00EK1OTZC	7
```

## 交互历史

```Instruments.inter.json
{
    "0": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    ],
    "1": [
        8,
        9,
        10,
        11,
        12,
        13
    ],
```

## 已经有的表征提取脚本

data_process/qwen_embeddings.py

## 需要做的实验

- 原始的文本 + 图片 提一组表征
- 原始的文本 + 增强的文本+ 图片提一组表征
- 增强的文本 + 图片 提一组表征

- 拿到表征之后，可以做 I2I 任务： 
  - 对于任意A, 得到共现最多的那个物品，作为标签 （B）
  - 这里的共现指的是使用 inter 的历史，看每个历史记录中两个物品同时出现的次数，那么这样就可以得到一个矩阵。每个物品都有一个共现最多的物品。
  - 对于每一个物品A，然后使用提取的表征做召回即可 （Hit@1, Hit@5, Hit@10），这里的意思就是召回1，5，10个看有没有 ground truth 在里面。
  - 所以我们总共会有三组实验，使用三种不同方案提取得到的表征来进行召回，然后召回看gt。