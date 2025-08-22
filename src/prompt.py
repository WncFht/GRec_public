sft_prompt = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{instruction}\n\n### Response:{response}"
)


all_prompt = {}

# =====================================================
# Task 1 -- Sequential Recommendation -- 17 Prompt
# =====================================================

seqrec_prompt = []

#####——0
prompt = {}
prompt["instruction"] = (
    "The user has interacted with items {inters} in chronological order. Can you predict the next possible item that the user may expect?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = (
    "I find the user's historical interactive items: {inters}, and I want to know what next item the user needs. Can you help me decide?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = (
    "Here are the user's historical interactions: {inters}, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order."
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = (
    "Based on the items that the user has interacted with: {inters}, can you determine what item would be recommended to him next?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = (
    "The user has interacted with the following items in order: {inters}. What else do you think the user need?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = (
    "Here is the item interaction history of the user: {inters}, what to recommend to the user next?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = (
    "Which item would the user be likely to interact with next after interacting with items {inters}?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = (
    "By analyzing the user's historical interactions with items {inters}, what is the next expected interaction item?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = (
    "After interacting with items {inters}, what is the next item that could be recommended for the user?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = (
    "Given the user's historical interactive items arranged in chronological order: {inters}, can you recommend a suitable item for the user?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = (
    "Considering the user has interacted with items {inters}. What is the next recommendation for the user?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = (
    "What is the top recommended item for the user who has previously interacted with items {inters} in order?"
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——12
prompt = {}
prompt["instruction"] = (
    "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user most desires based on the given interaction records."
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

# prompt = {}
# prompt["instruction"] = "The user has interacted with the following items in the past in order: {inters}. Please predict the next item that the user is most likely to interact with based on the given interaction record. Note that his most recently interacted item is {}."
# prompt["response"] = "{item}"
# prompt["task"] = "sequential"
# prompt["id"] = "1-13"
#
# seqrec_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = (
    "Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. The historical interactions are provided as follows: {inters}."
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = (
    "You can access the user's historical item interaction records: {inters}. Now your task is to recommend the next potential item to him, considering his past interactions."
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = (
    "You have observed that the user has interacted with the following items: {inters}, please recommend a next item that you think would be suitable for the user."
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = (
    "You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. Using this history as a reference, please select the next item to recommend to the user."
)
prompt["response"] = "{item}"
seqrec_prompt.append(prompt)

all_prompt["seqrec"] = seqrec_prompt


# ========================================================
# Task 2 -- Item2Index -- 19 Prompt
# ========================================================
# Remove periods when inputting

item2index_prompt = []

# ========================================================
# Title2Index

#####——0
prompt = {}
prompt["instruction"] = 'Which item has the title: "{title}"?'
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = 'Which item is assigned the title: "{title}"?'
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = (
    'An item is called "{title}", could you please let me know which item it is?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = 'Which item is called "{title}"?'
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = (
    'One of the items is named "{title}", can you tell me which item this is?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = 'What is the item that goes by the title "{title}"?'
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

# prompt = {}
# prompt["instruction"] = "Which item is referred to as \"{title}\"?"
# prompt["response"] = "{item}"
# item2index_prompt.append(prompt)

# ========================================================
# Description2Index

#####——6
prompt = {}
prompt["instruction"] = (
    'An item can be described as follows: "{description}". Which item is it describing?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = (
    'Can you tell me what item is described as "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = (
    'Can you provide the item that corresponds to the following description: "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)


# prompt = {}
# prompt["instruction"] = "What is the item described as follows: \"{description}\"?"
# prompt["response"] = "{item}"
# item2index_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = (
    'Which item has the following characteristics: "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = (
    'Which item is characterized by the following description: "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = (
    'I am curious to know which item can be described as follows: "{description}". Can you tell me?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

# ========================================================
# Title and Description to index

#####——12
prompt = {}
prompt["instruction"] = (
    'An item is called "{title}" and described as "{description}", can you tell me which item it is?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = (
    'Could you please identify what item is called "{title}" and described as "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = (
    'Which item is called "{title}" and has the characteristics described below: "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = (
    'Please show me which item is named "{title}" and its corresponding description is: "{description}".'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)


# prompt = {}
# prompt["instruction"] = "Here is an item called \"{title}\" and described as \"{description}\". Which item is it?"
# prompt["response"] = "{item}"
# item2index_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = (
    'Determine which item this is by its title and description. The title is: "{title}", and the description is: "{description}".'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——17
prompt = {}
prompt["instruction"] = (
    'Based on the title: "{title}", and the description: "{description}", answer which item is this?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

#####——18
prompt = {}
prompt["instruction"] = (
    'Can you identify the item from the provided title: "{title}", and description: "{description}"?'
)
prompt["response"] = "{item}"
item2index_prompt.append(prompt)

all_prompt["item2index"] = item2index_prompt


# ========================================================
# Task 3 -- Index2Item --17 Prompt
# ========================================================
# Remove periods when inputting

index2item_prompt = []

# ========================================================
# Index2Title

#####——0
prompt = {}
prompt["instruction"] = "What is the title of item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = "What title is assigned to item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = "Could you please tell me what item {item} is called?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——3
prompt = {}
prompt["instruction"] = "Can you provide the title of item {item}?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——4
prompt = {}
prompt["instruction"] = "What item {item} is referred to as?"
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = (
    "Would you mind informing me about the title of item {item}?"
)
prompt["response"] = "{title}"
index2item_prompt.append(prompt)

# ========================================================
# Index2Description

#####——6
prompt = {}
prompt["instruction"] = "Please provide a description of item {item}."
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = "Briefly describe item {item}."
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——8
prompt = {}
prompt["instruction"] = (
    "Can you share with me the description corresponding to item {item}?"
)
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = "What is the description of item {item}?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = "How to describe the characteristics of item {item}?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = "Could you please tell me what item {item} looks like?"
prompt["response"] = "{description}"
index2item_prompt.append(prompt)


# ========================================================
# index to Title and Description

#####——12
prompt = {}
prompt["instruction"] = "What is the title and description of item {item}?"
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——13
prompt = {}
prompt["instruction"] = (
    "Can you provide the corresponding title and description for item {item}?"
)
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——14
prompt = {}
prompt["instruction"] = (
    "Please tell me what item {item} is called, along with a brief description of it."
)
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——15
prompt = {}
prompt["instruction"] = (
    "Would you mind informing me about the title of the item {item} and how to describe its characteristics?"
)
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

#####——16
prompt = {}
prompt["instruction"] = (
    "I need to know the title and description of item {item}. Could you help me with that?"
)
prompt["response"] = "{title}\n\n{description}"
index2item_prompt.append(prompt)

all_prompt["index2item"] = index2item_prompt


# ========================================================
# Task 4 -- FusionSequentialRec -- Prompt
# ========================================================


fusionseqrec_prompt = []

#####——0
prompt = {}
prompt["instruction"] = (
    "The user has sequentially interacted with items {inters}. Can you recommend the next item for him? Tell me the title of the item？"
)
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)

#####——1
prompt = {}
prompt["instruction"] = (
    "Based on the user's historical interactions: {inters}, try to predict the title of the item that the user may need next."
)
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)

#####——2
prompt = {}
prompt["instruction"] = (
    "Utilizing the user's past ordered interactions, which include items {inters}, please recommend the next item you think is suitable for the user and provide its title."
)
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)


#####——3
prompt = {}
prompt["instruction"] = (
    "After interacting with items {inters}, what is the most probable item for the user to interact with next? Kindly provide the item's title."
)
prompt["response"] = "{title}"
fusionseqrec_prompt.append(prompt)


#####——4
prompt = {}
prompt["instruction"] = (
    "Please review the user's historical interactions: {inters}, and describe what kind of item he still needs."
)
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)

#####——5
prompt = {}
prompt["instruction"] = (
    "Here is the item interaction history of the user: {inters}, please tell me what features he expects from his next item."
)
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)

#####——6
prompt = {}
prompt["instruction"] = (
    "By analyzing the user's historical interactions with items {inters}, can you infer what the user's next interactive item will look like?"
)
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)

#####——7
prompt = {}
prompt["instruction"] = (
    "Access the user's historical item interaction records: {inters}. Your objective is to describe the next potential item for him, taking into account his past interactions."
)
prompt["response"] = "{description}"
fusionseqrec_prompt.append(prompt)


#####——8
prompt = {}
prompt["instruction"] = (
    "Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"
)
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)

#####——9
prompt = {}
prompt["instruction"] = (
    "I possess a user's past interaction history, denoted by the title sequence of interactive items: {inter_titles}, and I am interested in knowing the user's next most desired item. Can you help me?"
)
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)

#####——10
prompt = {}
prompt["instruction"] = (
    "Considering the title sequence of user history interaction items: {inter_titles}. What is the next recommendation for the user?"
)
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)

#####——11
prompt = {}
prompt["instruction"] = (
    "You have obtained the ordered title list of user historical interaction items, as follows: {inter_titles}. Based on this historical context, kindly choose the subsequent item for user recommendation."
)
prompt["response"] = "{item}"
fusionseqrec_prompt.append(prompt)


all_prompt["fusionseqrec"] = fusionseqrec_prompt


# ========================================================
# Task 7 -- Multimodal Item2Index -- 15 Prompt
# ========================================================

mmitem2index_prompt = []

# 变体1：Title、Description、Category、Brand - item ID
prompt = {}
prompt["instruction"] = (
    'Analyze the product image and associated information (Title: "{title}", Description: "{description}", Category: "{categories}", Brand: "{brand}") to output the item ID.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体2：Brand、Title、Description、Category - item token id
prompt = {}
prompt["instruction"] = (
    'Based on the product image and provided details (Brand: "{brand}", Title: "{title}", Description: "{description}", Category: "{categories}"), generate the item token id.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体3：Category、Brand、Title、Description - item identifier tokens
prompt = {}
prompt["instruction"] = (
    'Process the product image and given information (Category: "{categories}", Brand: "{brand}", Title: "{title}", Description: "{description}") to produce the item identifier tokens.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体4：Description、Category、Brand、Title - unique item identifier
prompt = {}
prompt["instruction"] = (
    'Examine the product image and associated data (Description: "{description}", Category: "{categories}", Brand: "{brand}", Title: "{title}") to output the unique item identifier.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体5：Title、Brand、Description、Category - item tokens
prompt = {}
prompt["instruction"] = (
    'Using the product image and provided information (Title: "{title}", Brand: "{brand}", Description: "{description}", Category: "{categories}"), determine the item tokens.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体6：Brand、Category、Description、Title - item code
prompt = {}
prompt["instruction"] = (
    'Analyze the product image along with the details (Brand: "{brand}", Category: "{categories}", Description: "{description}", Title: "{title}") to generate the item code.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体7：Description、Title、Brand、Category - item representation
prompt = {}
prompt["instruction"] = (
    'From the product image and given attributes (Description: "{description}", Title: "{title}", Brand: "{brand}", Category: "{categories}"), create the item representation.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体8：Category、Description、Title、Brand - product identifier
prompt = {}
prompt["instruction"] = (
    'Process the product image and associated information (Category: "{categories}", Description: "{description}", Title: "{title}", Brand: "{brand}") to output the product identifier.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体9：Title、Category、Brand、Description - item encoding
prompt = {}
prompt["instruction"] = (
    'Based on the product image and provided data (Title: "{title}", Category: "{categories}", Brand: "{brand}", Description: "{description}"), generate the item encoding.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体10：Brand、Description、Category、Title - item token sequence
prompt = {}
prompt["instruction"] = (
    'Utilize the product image and given information (Brand: "{brand}", Description: "{description}", Category: "{categories}", Title: "{title}") to produce the item token sequence.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体11：Description、Brand、Title、Category - unique token ID
prompt = {}
prompt["instruction"] = (
    'Analyze the product image and provided details (Description: "{description}", Brand: "{brand}", Title: "{title}", Category: "{categories}") to output the unique token ID.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

# 变体12：Category、Title、Description、Brand - item hash
prompt = {}
prompt["instruction"] = (
    'From the product image and associated information (Category: "{categories}", Title: "{title}", Description: "{description}", Brand: "{brand}"), generate the item hash.'
)
prompt["response"] = "{item}"
mmitem2index_prompt.append(prompt)

all_prompt["mmitem2index"] = mmitem2index_prompt

# ========================================================
# Task 8 -- Multimodal Index2Item
# ========================================================

mmindex2item_prompt = []

# 变体1：只输出title
prompt = {}
prompt["instruction"] = (
    "Analyze the product image and provide the item title for item {item}."
)
prompt["response"] = "Title: {title}"
mmindex2item_prompt.append(prompt)

# 变体2：输出title和description
prompt = {}
prompt["instruction"] = (
    "Based on visual analysis of the product image, please identify the title and description for item {item}."
)
prompt["response"] = "Title: {title}\nDescription: {description}"
mmindex2item_prompt.append(prompt)

# 变体3：输出title和brand
prompt = {}
prompt["instruction"] = (
    "Examine the product image and determine the title and brand information for item {item}."
)
prompt["response"] = "Title: {title}\nBrand: {brand}"
mmindex2item_prompt.append(prompt)

# 变体4：输出title和categories
prompt = {}
prompt["instruction"] = (
    "From the product image, identify the title and category information for item {item}."
)
prompt["response"] = "Title: {title}\nCategory: {categories}"
mmindex2item_prompt.append(prompt)

# 变体5：输出title、description和brand
prompt = {}
prompt["instruction"] = (
    "Using visual analysis of the product image, provide the title, description, and brand details for item {item}."
)
prompt["response"] = (
    "Title: {title}\nDescription: {description}\nBrand: {brand}"
)
mmindex2item_prompt.append(prompt)

# 变体6：输出title、description和categories
prompt = {}
prompt["instruction"] = (
    "Analyze the product image to extract the title, description, and category information for item {item}."
)
prompt["response"] = (
    "Title: {title}\nDescription: {description}\nCategory: {categories}"
)
mmindex2item_prompt.append(prompt)

# 变体7：输出title、brand和categories
prompt = {}
prompt["instruction"] = (
    "Based on the product image, identify the title, brand, and category details for item {item}."
)
prompt["response"] = "Title: {title}\nBrand: {brand}\nCategory: {categories}"
mmindex2item_prompt.append(prompt)

# 变体8：全部输出，不同表述
prompt = {}
prompt["instruction"] = (
    "Examine the product image and extract comprehensive information for item {item} including title, description, brand, and categories."
)
prompt["response"] = (
    "Title: {title}\nDescription: {description}\nBrand: {brand}\nCategory: {categories}"
)
mmindex2item_prompt.append(prompt)

# 变体9：只输出title，不同表述
prompt = {}
prompt["instruction"] = (
    "What is the product title for item {item} based on the visual content of the image?"
)
prompt["response"] = "Title: {title}"
mmindex2item_prompt.append(prompt)

# 变体10：输出title和brand，不同表述
prompt = {}
prompt["instruction"] = (
    "Identify the product name and brand for item {item} from the given product image."
)
prompt["response"] = "Title: {title}\nBrand: {brand}"
mmindex2item_prompt.append(prompt)

all_prompt["mmindex2item"] = mmindex2item_prompt


# ========================================================
# Task 9 -- Multimodal Text Enrich
# ========================================================
textenrich_prompt = []

prompt = {}
prompt["instruction"] = (
    'Based on the product image, original title "{title}", brand "{brand}", categories "{categories}", description "{description}", and item tokens {item}, please enrich the product information by providing enhanced title, relevant tags, key highlights, and main characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

prompt = {}
prompt["instruction"] = (
    'Given the product image and basic information (Title: "{title}", Brand: "{brand}", Categories: "{categories}", Description: "{description}") along with item tokens {item}, generate comprehensive product details including enhanced title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体1：item ids在前，Title、Brand、Categories、Description
prompt = {}
prompt["instruction"] = (
    'Using item ids {item}, the product image, and original information (Title: "{title}", Brand: "{brand}", Categories: "{categories}", Description: "{description}"), please generate enhanced product descriptions including improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体2：item ids在前，Brand、Title、Description、Categories
prompt = {}
prompt["instruction"] = (
    'Based on item ids {item}, the product image, and original information (Brand: "{brand}", Title: "{title}", Description: "{description}", Categories: "{categories}"), generate enhanced product descriptions with improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体3：item ids在前，Categories、Description、Title、Brand
prompt = {}
prompt["instruction"] = (
    'Given item ids {item}, the product image, and original information (Categories: "{categories}", Description: "{description}", Title: "{title}", Brand: "{brand}"), create enhanced product descriptions with improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体4：item ids在前，Description、Categories、Brand、Title
prompt = {}
prompt["instruction"] = (
    'Analyze item ids {item}, the product image, and provided details (Description: "{description}", Categories: "{categories}", Brand: "{brand}", Title: "{title}") to produce enhanced product descriptions including optimized title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体5：item ids在前，Title、Categories、Description、Brand
prompt = {}
prompt["instruction"] = (
    'From item ids {item}, the product image, and original information (Title: "{title}", Categories: "{categories}", Description: "{description}", Brand: "{brand}"), generate enhanced product information: title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体6：item ids在前，Brand、Categories、Title、Description
prompt = {}
prompt["instruction"] = (
    'Utilize item ids {item}, the product image, and original data (Brand: "{brand}", Categories: "{categories}", Title: "{title}", Description: "{description}") to develop enhanced product descriptions with better title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体7：item ids在前，Categories、Title、Brand、Description
prompt = {}
prompt["instruction"] = (
    'Process item ids {item}, the product image, and product information (Categories: "{categories}", Title: "{title}", Brand: "{brand}", Description: "{description}") to generate enhanced product descriptions including refined title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体8：item ids在前，Description、Brand、Categories、Title
prompt = {}
prompt["instruction"] = (
    'Examine item ids {item}, the product image, and the original product information (Description: "{description}", Brand: "{brand}", Categories: "{categories}", Title: "{title}") to generate enhanced product descriptions featuring improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体9：item ids在前，Title、Description、Brand、Categories
prompt = {}
prompt["instruction"] = (
    'Using item ids {item}, the product image, along with original details (Title: "{title}", Description: "{description}", Brand: "{brand}", Categories: "{categories}"), please create enhanced product descriptions with improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体10：item ids在前，Brand、Description、Title、Categories
prompt = {}
prompt["instruction"] = (
    'Based on item ids {item}, the product image, and product data (Brand: "{brand}", Description: "{description}", Title: "{title}", Categories: "{categories}"), generate enhanced product descriptions including optimized title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)


all_prompt["text_enrich"] = textenrich_prompt

# ====================
# textenrich without item id
# ====================

textenrich_without_id_prompt = []

prompt = {}
prompt["instruction"] = (
    'Based on the product image, original title "{title}", brand "{brand}", categories "{categories}", description "{description}", please enrich the product information by providing enhanced title, relevant tags, key highlights, and main characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)


prompt = {}
prompt["instruction"] = (
    'Given the product image and basic information (Title: "{title}", Brand: "{brand}", Categories: "{categories}", Description: "{description}"), generate comprehensive product details including enhanced title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体1：item ids在前，Title、Brand、Categories、Description
prompt = {}
prompt["instruction"] = (
    'Using the product image, and original information (Title: "{title}", Brand: "{brand}", Categories: "{categories}", Description: "{description}"), please generate enhanced product descriptions including improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体2：item ids在前，Brand、Title、Description、Categories
prompt = {}
prompt["instruction"] = (
    'Based on the product image, and original information (Brand: "{brand}", Title: "{title}", Description: "{description}", Categories: "{categories}"), generate enhanced product descriptions with improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体3：item ids在前，Categories、Description、Title、Brand
prompt = {}
prompt["instruction"] = (
    'Given the product image, and original information (Categories: "{categories}", Description: "{description}", Title: "{title}", Brand: "{brand}"), create enhanced product descriptions with improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体4：item ids在前，Description、Categories、Brand、Title
prompt = {}
prompt["instruction"] = (
    'Analyze the product image, and provided details (Description: "{description}", Categories: "{categories}", Brand: "{brand}", Title: "{title}") to produce enhanced product descriptions including optimized title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_prompt.append(prompt)

# 变体5：item ids在前，Title、Categories、Description、Brand
prompt = {}
prompt["instruction"] = (
    'From the product image, and original information (Title: "{title}", Categories: "{categories}", Description: "{description}", Brand: "{brand}"), generate enhanced product information: title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体6：item ids在前，Brand、Categories、Title、Description
prompt = {}
prompt["instruction"] = (
    'Utilize the product image, and original data (Brand: "{brand}", Categories: "{categories}", Title: "{title}", Description: "{description}") to develop enhanced product descriptions with better title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体7：item ids在前，Categories、Title、Brand、Description
prompt = {}
prompt["instruction"] = (
    'Process the product image, and product information (Categories: "{categories}", Title: "{title}", Brand: "{brand}", Description: "{description}") to generate enhanced product descriptions including refined title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体8：item ids在前，Description、Brand、Categories、Title
prompt = {}
prompt["instruction"] = (
    'Examine the product image, and the original product information (Description: "{description}", Brand: "{brand}", Categories: "{categories}", Title: "{title}") to generate enhanced product descriptions featuring improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体9：item ids在前，Title、Description、Brand、Categories
prompt = {}
prompt["instruction"] = (
    'Using  the product image, along with original details (Title: "{title}", Description: "{description}", Brand: "{brand}", Categories: "{categories}"), please create enhanced product descriptions with improved title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

# 变体10：item ids在前，Brand、Description、Title、Categories
prompt = {}
prompt["instruction"] = (
    'Based on the product image, and product data (Brand: "{brand}", Description: "{description}", Title: "{title}", Categories: "{categories}"), generate enhanced product descriptions including optimized title, tags, highlights, and characteristics.'
)
prompt["response"] = (
    "Enhanced Title: {enhanced_title}\nTags: {tags}\nHighlights: {highlights}\nCharacteristics: {characteristics}"
)
textenrich_without_id_prompt.append(prompt)

all_prompt["text_enrich_without_id"] = textenrich_without_id_prompt

# ====================
# multimodal_seqrec
# ====================

multimodal_seqrec_prompt = []

# ---------- 0 ----------
p = {}
p["instruction"] = (
    "The user has sequentially interacted with the following items, each accompanied by its image: {inters}. Based on both the visual and textual cues, what is the token of the next item the user is likely to interact with?"
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 1 ----------
p = {}
p["instruction"] = (
    "Here are the user's historical interactions in order,where every item is represented by its token and an associated image: {inters}. Please predict the token of the subsequent item that best fits the user's preference."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 2 ----------
p = {}
p["instruction"] = (
    "Given the user's chronological interaction history, with each item depicted by both its token and image: {inters}, determine the token of the next recommended item."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 3 ----------
p = {}
p["instruction"] = (
    "The user has interacted with the following sequence of items, each shown as token + image: {inters}. Infer the token of the item the user is most likely to interact with next."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 4 ----------
p = {}
p["instruction"] = (
    "Observe the user's ordered interaction records, where every entry is a token paired with its image: {inters}. Provide the token of the next probable item."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 5 ----------
p = {}
p["instruction"] = (
    "You are given the user's historical item interactions, each consisting of a token and the corresponding image: {inters}. What token should be recommended as the next item?"
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 6 ----------
p = {}
p["instruction"] = (
    "Based on the user's past interaction sequence, where each item is represented by token and image: {inters}, predict the token of the next item the user will most likely choose."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 7 ----------
p = {}
p["instruction"] = (
    "Here is the user's chronological interaction list: {inters}, with each item shown as token and image. Identify the token of the next item to recommend."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 8 ----------
p = {}
p["instruction"] = (
    "The user has successively interacted with these items: {inters}, each displayed as token and image. What is the token of the next item that aligns with the user's preference?"
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# ---------- 9 ----------
p = {}
p["instruction"] = (
    "After viewing the user's interaction sequence, where each item is given by its token and image: {inters}, forecast the token of the subsequent item."
)
p["response"] = "{target_item}"
multimodal_seqrec_prompt.append(p)

# 注册到 all_prompt
all_prompt["multimodal_seqrec"] = multimodal_seqrec_prompt
