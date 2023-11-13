# LLMs-Works-View
> ***Focus on Current Works and Surveys in LLMs and its Application in Dialogue Field.***

> ***This responsibility records the survey and study of my road of NLP & LLMs Applications***.

---

### 💬 Brief Introduction of Markdown Files in This Responsibility

* `Readme.md` : Papers and Codes in Google Scholar, Baidu Scholar, Arxiv and Github.
* `Analysis_in_CN_Internet.md` : Analysis, Reviews and Comments in Chinese Internet, including CSDN, ZhiHu, etc.
* This excel records the paper in this file:

| Books | Surveys | Base Model | Fine-Tune | Casuality | Agents | Retrieval | Datasets |
|-------|---------|------------|-----------|-----------|--------|----------|----------|
| **1** | **18**  | **3**      |   **3**   |   **7**   | **3**  |  **3**   |  **3**   |


![Galaxy is our dream](images/Galaxy.png)

---

### 📙 Books
***1. 大规模语言模型：从理论到实践（复旦大学自然语言处理实验室，2023.09.10 v1）*** : 

> | Github Link | Paper Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | [Github-Page](https://github.com/intro-llm/intro-llm.github.io) | [PDF-Page](https://intro-llm.github.io/chapter/LLM-TAP.pdf) <br> [BaiDuNetDisk-Page](https://pan.baidu.com/s/1smGQ5ECzDIpvZladuCE59g?pwd=jyz6) | [OfficialWebsite-Page](https://intro-llm.github.io/#chapter) | ✔ |
>
> ***Summarize** : None*

### 🧬 Surveys
***1. A Survey of Large Language Models（Renmin University of China, 2023.03）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | [Github-Page](https://github.com/rucaibox/llmsurvey) | [Arxiv-Page](https://arxiv.org/abs/2303.18223) | [PaperWithCode-Page](https://paperswithcode.com/paper/a-survey-of-large-language-models) | ✔ |
>
> ***· Tags** : **`Comprehensive Review`** , **`Guidance and Conclusion of Large Language Models`***
>
> ***· Summarize** : A Brief and Comprehensive Introduction of Large Language Models, including Application Fields, Evaluations, Training Methods, Shortage, etc. This thesis elaborated the background and timeline of the LLMs comprehensively. Some Table and Graphs are used to conclude and demonstrate the detail of these Large Language Models. In addition, the thesis explored the training and inference process of the LLMs, showing the relationship and dependency of LLMs and hardwares, such as CPUs and GPUs. Application and Related Method of Large Language Models is expressed in a large chapter, including QA-system, In-context Learning Method, LLMs-based Agents, etc. For example, this paper provide a total introduction of In-Context Learning and Alignment.*


***2. A Survey on LLM-based Autonomous Agents（Renmin University of China）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | [Github-Page](https://github.com/Paitesanshi/LLM-Agent-Survey) | [Arxiv-Page](https://arxiv.org/pdf/2308.11432.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/a-survey-on-large-language-model-based)  | ✔ |
>
> ***· Tags** : **`Construction and Techniques`** , **`Application`** , **`Evaluation`***
>
> ***· Summarize** : This thesis first discuss the construction of LLM-based autonomous agents, for which they propose a unified framework that encompasses a majority of the previous work. Then, they present a comprehensive overview of the diverse applications of LLM-based autonomous agents in the fields of social science, natural science, and engineering. Finally, the authors delve into the evaluation strategies commonly used forLLM-based autonomous agents. Based on the previous studies, they also present several challenges and future directions in this field*

***3. A Survey on Multimodal Large Language Models（University of Science and Technology of China & Tencent YouTu Lab）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | [Github-Page](https://github.com/bradyfu/awesome-multimodal-large-language-models) | [Arxiv-Page](https://arxiv.org/pdf/2306.13549v1.pdf)  | [PaperWithCode-Page ](https://paperswithcode.com/paper/a-survey-on-multimodal-large-language-models)     |
>
> ***· Tags** :*
>
> ***· Summarize** :*

***4. Recent Advances in Deep Learning Based Dialogue Systems Survey（Nanyang Technological University, 2021.05）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | NO GITHUB CODE | [Arxiv-Page](https://arxiv.org/pdf/2105.04387.pdf)  |            PaperWithCode-Page        | ✔ |
>
> ***· Tags** :*
>
> ***· Summarize** : A survey about the development of the dialogue system since the application of deep learning methods. This thesis demonstrate the backbone models and methods(CNN, Transformer, Reinforcement Learning), Task-oriented Dialouge system, Open-domain Dialogue system, evaluation methods, datasets. However, since 2022, Large Language Models have take the domainated position of Dialogue System Tasks. So this survey is still limited.*

***5. A Survey on Table Question Answering: Recent Advances（Harbin Institute of Technology (Shenzhen) & Peng Cheng Laboratory）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | NO GITHUB CODE | [Arxiv-Page](https://arxiv.org/pdf/2207.05270.pdf)  |  [PaperWithCode-Page ](https://paperswithcode.com/paper/a-survey-on-table-question-answering-recent)    |
>
> ***· Tags** :*
>
> ***· Summarize** :*

***6. Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing（Carnegie Mellon University & National University of Singapore, 2023.01）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | [Github-Page](https://github.com/mingkaid/rl-prompt) | [Arxiv-Page](https://dl.acm.org/doi/pdf/10.1145/3560815) | [PaperWithCode-Page](https://github.com/mingkaid/rl-prompt)  | ✔ |
>
> ***· Tags** : **`Prompt Methods of LLMs`***
>
> ***· Summarize** : A survey of Prompting methods applied in NLP. In this thesis, there are 4 paradigms in NLP: Fully Supervised Learning(Non-NN), Fully Supervised Learning(NN), PreTrain-FineTune(After2018) and PreTrain-Prompt-Predict. With the development of Transformer and Language Models, there are more NLP tasks. If researchers still use PreTrain-FineTune to tune Large Language Model in specific tasks, it would cost massive computing resources, which is extremely unrealistic. Hence, PreTrain-Prompt-Predict is a significant and efficient paradigms that makes LLMs to fit different tasks without tuning of re-training.*


***7. Talking with Machines: A Comprehensive Survey of Emergent Dialogue Systems（University of California, Berkeley）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | NO GITHUB CODE | [Arxiv-Page](https://arxiv.org/pdf/2305.16324.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/talking-with-machines-a-comprehensive-survey)  |
>
> ***· Tags** :*
>
> ***· Summarize** :*

***8. Reasoning with Language Model Prompting: A Survey（Zhejiang University）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|-------------|
> | [Github-Page](https://github.com/zjunlp/Prompt4ReasoningPapers) | [Arxiv-Page](https://arxiv.org/pdf/2212.09597.pdf) |            PaperWithCode-Page        |
>
> ***· Tags** :*
>
> ***· Summarize** : Survey and Evaluation of QA-System*

***9. A Survey on Evaluation of Large Language Models（Jilin University & Microsoft, 2023.08）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/mlgroupjlu/llm-eval-survey) | [Arxiv-Page](https://arxiv.org/pdf/2307.03109v7.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/a-survey-on-evaluation-of-large-language) | ✔ |
>
> ***· Tags** : **`Evaluation Methods of LLMs`***
>
> ***· Summarize** : With the development of Large Language models, How to evaluate the performance of LLMs is extremely important to researchers. This thesis conclude the current method of the LLMs evaluation, and disscuss the problems and future trend.*

***10. Evaluating Open-QA Evaluation（Westlake University & Northeastern University, 2023.08）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/wangcunxiang/qa-eval) | [Arxiv-Page](https://arxiv.org/pdf/2305.12421v3.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/evaluating-open-question-answering-evaluation) | ✔ |
>
> ***· Tags** :*
>
> ***· Summarize** : A Research and Evaluation of QA-Task. This thesis proposed a QA-Evaluation Task and related Dataset EVOUNA.*

***11. A Survey on In-context Learning （Peking University & Shanghai AI Lab, 2023.06）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/dqxiu/icl_paperlist) | [Arxiv-Page](https://arxiv.org/pdf/2301.00234v3.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/a-survey-for-in-context-learning) | ✔ |
>
> ***· Tags** : **`Important Survey of ICL`***
>
> ***· Summarize** : Survey and Evaluation of ICL. In-Context Learning is firstly proposed in the work of GPT-3. However, it was used as tricks before the conclusion in the paper, and play an indispensable role in the training process of LLMs. ICL uses the demonstration of the tasks or show some samples of these tasks. Then, a designed mode is used to combine these samples to a natural language prompt and add to the input of LLMs. The efficiency and effectiveness of ICL is influenced by the design of samples. In the pre-training phase, the LLM essentially encodes the implicit model through its parameters. In the forward computation phase, LLM can implement learning algorithms such as gradient descent through the examples provided in ICL, or directly compute closed solutions to update these models. In addition Chain-of-Thought(CoT) is proposed to optimize the performance of LLMs in complex inference tasks, which can be combined with ICL in the situation of Few-Shot or Zero-Shot. It's worth noting that In-Context Learning is not as same as the Long Context, which require Large Language Models to store more information from the past interaction.*


***12. The Rise and Potential of Large Language Model Based Agents: A Survey（Fudan University, 2023.09）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/woooodyy/llm-agent-paper-list) | [Arxiv-Page](https://arxiv.org/pdf/2309.07864v3.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/the-rise-and-potential-of-large-language) | ✔ |
>
> ***· Tags** : **`Agents`***
>
> ***· Summarize** : Survey of LLM-based Agents. The idea and demonstration of the LLM-based Multi-Agent System is extremely inspired. A traditional and excellent work of Multi-Agents is showed in[ 16. Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf](https://paperswithcode.com/paper/deep-model-fusion-a-survey).*

***13. Data-centric Artificial Intelligence: A Survey （Rice University & Texas A&M University, 2023.07）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/daochenzha/data-centric-ai) | [Arxiv-Page](https://arxiv.org/pdf/2303.10158v3.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/data-centric-artificial-intelligence-a-survey) | ✔ |
>
> ***· Tags** : **`Conclusion of Data Processing Methods and Theories`***
>
> ***· Summarize** : A survey about theories, process flows, augmentations, benchmarks and etc of data. In this paper, the concept and demonstration of Data-centric Artificial Intelligence instead of Data-driven Artificial Intelligence is very inspired. As the saying goes that Model determine the lower limit of the performance and the data determine the upper limit. In my view, the process, methods and tricks of processing original or raw data, or Data Engineering should be given the same status as Model Construction in the Solution.*


***14. Aligning Large Language Models with Human: A Survey （Huawei Noah's Ark Lab, 2023.07）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/garyyufei/alignllmhumansurvey) | [Arxiv-Page](https://arxiv.org/pdf/2307.12966v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/aligning-large-language-models-with-human-a) | ✔ |
>
> ***· Tags** : **`Aligement of LLMs`***
>
> ***· Summarize** : Alignment aims to align the LLM's behavior with human values or preferences. Though LLMs show unparalleled performance in generation, they sometimes generate some words that may mislead users, or is spurious.*


***15. Deep Model Fusion: A Survey（National University of Defense Technology & JD Explore Academy & Beijing Institute of Technology, 2023.09）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | Github-Page | [Arxiv-Page](https://arxiv.org/pdf/2309.15698v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/deep-model-fusion-a-survey) | ✔ |
>
> ***· Tags** : **`Fusion`***
>
> ***· Summarize** : Excellient thesis that conclude most deep model fusion methods. The existing methods are summarized into four categories: Mode Connectivity, Alignment, Weight Average, Ensemble Learning, whick contains well-known applications in the past 10 years. Each category is demonstrated by academic terms and formulas. It is the first thesis that the work of deep model fusion in these years are complete summarized. In addition, it is worth noting that the chapter of Alignment is similar to the thesis [14. Aligning Large Language Models with Human: A Survey （Huawei Noah's Ark Lab, 2023.07）](https://arxiv.org/pdf/2307.12966v1.pdf). The former focused on the whole development and most practical techniques of Alignment and the later layed emphasis on the Alignment in Large Language model.*


***16. Explainability for Large Language Models: A Survey （New Jersey Institute of Technology etc, 2023.09）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | Github-Page | [Arxiv-Page](https://arxiv.org/pdf/2309.01029v2.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/explainability-for-large-language-models-a) | ✔ |
>
> ***· Tags** : **`Explainability of LLMs`***
>
> ***· Summarize** : An paper about Explainability for Large Language Models. This thesis demonstrated the explainability in lots of directions, such as gradient explainability, attanion explainability.*


***17. An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning（Westlake University &  WeChat, 2023.08）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/luoxiaoheics/continual-tune) | [Arxiv-Page](https://browse.arxiv.org/pdf/2308.08747.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/an-empirical-study-of-catastrophic-forgetting) | ✔ |
>
> ***· Tags** : **`Catastrophic Forgetting of LLMs`***
>
> ***· Summarize** : Unsupervised Learning is main training mode of Large Language Models for big training dataset. However, when fine-tuning LLMs for downstream tasks, Catastrophic Forgetting will happen.*


***18. A Survey on Causal Inference（Alibaba Group, 2021.08）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | [Github-Page](https://github.com/dmachlanski/iads-summer-school-causality-2021) | [Arxiv-Page](https://arxiv.org/pdf/2002.02770v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/a-survey-on-causal-inference) | ✔ |
>
> ***· Tags** : **`Causal Inference`***
>
> ***· Summarize** : Causal inference is a critical research topic across many domains, such as statistics, computer science, education, public policy and economics, for decades. Nowadays, estimating causal effect from observational data has become an appealing research direction owing to the large amount of available data and low budget requirement, compared with randomized controlled trials. Embraced with the rapidly developed machine learning area, various causal effect estimation methods for observational data have sprung up. In this survey, we provide a comprehensive review of causal inference methods under the potential outcome framework, one of the well known causal inference framework. The methods are divided into two categories depending on whether they require all three assumptions of the potential outcome framework or not. For each category, both the traditional statistical methods and the recent machine learning enhanced methods are discussed and compared. The plausible applications of these methods are also presented, including the applications in advertising, recommendation, medicine and so on. Moreover, the commonly used benchmark datasets as well as the open-source codes are also summarized, which facilitate researchers and practitioners to explore, evaluate and apply the causal inference methods.*

---

### 💊 Large Language Base Models

***1. ChatGLM-6B/130B & ChatGLM2-6B/12B/32B/66B/130B & CHATGLM3-6B [Based on GLM & GLM-130B]（TsingHua University, 2022.10）***: 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github(GLM-130B)](https://github.com/thudm/glm-130b) <br> [Github(ChatGLM)](https://github.com/THUDM/ChatGLM-6B) <br> [Github(ChatGLM2)](https://github.com/THUDM/ChatGLM2-6B) <br> [Github(ChatGLM3)](https://github.com/THUDM/ChatGLM3) | [Arxiv(GLM-130B)](https://arxiv.org/pdf/2210.02414.pdf) <br> [Arxiv(GLM)](https://arxiv.org/pdf/2103.10360.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/glm-130b-an-open-bilingual-pre-trained-model) |[OfficialWeb-Page]( https://chatglm.cn/blog) <br> [HuggingFace-Page](https://huggingface.co/THUDM) | GLM-130B ✔ <br> GLM ✔ |
>
> ***· Tags** : **`Pretrained Base Model`** , **`Efficiency`** , **`Low Dependence of Hardwares`***
>
> ***· Summarize** : GLM and GLM-130B faced a lot of challenge when it was firstly proposed, including Lack of CPUs and GPUs, Lack of a Robust Pre-Training Methods, Lack of fast inference solution especially in low computing resources. TsingHua Researchers proposed the training details GLM-130B aiming to provide others a reference about how to train a Large Language Model(100B-Scale), because "It is not easy to train a Large Language Model". GLM-130B use GLM as the backbone, with the methods of Layer Norm, Positional Encoding, FFNs. In the Training Stability Stage, Mixed-Precision(FP16 & FP32) and Embedding Layer Gradient Shrink is used. In addition, GLM-130B was distrubuted in 96 NVIDIA A100-40GB for training(60 Days). In this process, Research teams usually faced the challenges of the lack of GPUs(Even they gained some support from several platforms, they have to edit their codes to fit these platforms or even re-code in C++). Therefore, in the inference stage of GLM-130B, researchers use INT8 and even INT4 to quantize the model weights and the activations, which makes the model can run the inference on RTX 2080Ti.*


***2. LLaMA-7B/13B/33B/65B & LLaMA2-34B/70B（FaceBook/Meta AI, 2023.02）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/facebookresearch/llama) | [Arxiv-Page ](https://arxiv.org/pdf/2302.13971v1.pdf)| [PaperWithCode-Page](https://paperswithcode.com/paper/llama-open-and-efficient-foundation-language-1)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Pretrained Base Model`** , **`LLMs SOTA`***
>
> ***· Summarize** : None*


***3. Baichuan2-7B/13B（Baichuan Intelligent Technology）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/baichuan-inc/Baichuan2) | Arxiv-Page | [PaperWithCode-Page](https://paperswithcode.com/paper/baichuan-2-open-large-scale-language-models)  |[OfficialWeb-Page](https://www.baichuan-ai.com/home) | |
>
> ***· Tags** : **`Pretrained Base Model`***
>
> ***· Summarize** : None*


***4. Yi-6B/34B（01 AI, 2023.11）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/01-ai/Yi) | Arxiv-Page | [PaperWithCode-Page]()  |[OfficialWeb-Page](https://www.lingyiwanwu.com/) <br> [HuggingFace-Page](https://huggingface.co/01-ai)  | ✔ |
>
> ***· Tags** : **`Pretrained Base Model`***
>
> ***· Summarize** : None*

---

### 💡 Fine-Tuning Methods

***1. LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models （Chinese University of Hong Kong & MIT, 2023.09）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/dvlab-research/longlora) | [Arxiv-Page](https://arxiv.org/pdf/2309.12307v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/longlora-efficient-fine-tuning-of-long)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Fine-tune`** , **`Long-Context`** , , **`SOTA`***
>
> ***· Summarize** : We present LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs), with limited computation cost. Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. For example, training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048. In this paper, we speed up the context extension of LLMs in two aspects. On the one hand, although dense global attention is needed during inference, fine-tuning the model can be effectively and efficiently done by sparse local attention. The proposed shift short attention effectively enables context extension, leading to non-trivial computation saving with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only two lines of code in training, while being optional in inference. On the other hand, we revisit the parameter-efficient fine-tuning regime for context expansion. Notably, we find that LoRA for context extension works well under the premise of trainable embedding and normalization. LongLoRA demonstrates strong empirical results on various tasks on LLaMA2 models from 7B/13B to 70B. LongLoRA adopts LLaMA2 7B from 4k context to 100k, or LLaMA2 70B to 32k on a single 8x A100 machine. LongLoRA extends models' context while retaining their original architectures, and is compatible with most existing techniques, like FlashAttention-2. In addition, to make LongLoRA practical, we collect a dataset, LongQA, for supervised fine-tuning. It contains more than 3k long context question-answer pairs.*

***2. QLoRA: Efficient Finetuning of Quantized LLMs （University of Washington, 2023.05）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/artidoro/qlora) | [Arxiv-Page](https://arxiv.org/pdf/2305.14314v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/qlora-efficient-finetuning-of-quantized-llms)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Fine-tune`** , **`Long-Context`** , , **`Efficiency`***
>
> ***· Summarize** :*


***3. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts （University of California, 2020.10）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/ucinlp/autoprompt) | [Arxiv-Page](https://arxiv.org/pdf/2010.15980v2.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/autoprompt-eliciting-knowledge-from-language)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Prompt Method`** , **`Automatic Prompt`** , , **`Opti`***
>
> ***· Summarize** : The remarkable success of pretrained language models has motivated the study of what kinds of knowledge these models learn during pretraining. Reformulating tasks as fill-in-the-blanks problems (e.g., cloze tests) is a natural approach for gauging such knowledge, however, its usage is limited by the manual effort and guesswork required to write suitable prompts. To address this, we develop AutoPrompt, an automated method to create prompts for a diverse set of tasks, based on a gradient-guided search. Using AutoPrompt, we show that masked language models (MLMs) have an inherent capability to perform sentiment analysis and natural language inference without additional parameters or finetuning, sometimes achieving performance on par with recent state-of-the-art supervised models. We also show that our prompts elicit more accurate factual knowledge from MLMs than the manually created prompts on the LAMA benchmark, and that MLMs can be used as relation extractors more effectively than supervised relation extraction models. These results demonstrate that automatically generated prompts are a viable parameter-free alternative to existing probing methods, and as pretrained LMs become more sophisticated and capable, potentially a replacement for finetuning.*

---

### ⏳ Development and Extension of Large Language Models with Causality Learning

***1. XInsight: eXplainable Data Analysis Through The Lens of Causality（Hong Kong University of Science and Technology, 2023.07）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page]() | [Arxiv-Page](https://arxiv.org/pdf/2207.12718v4.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/xinsight-explainable-data-analysis-through)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Causality`** , **`Theory`***
>
> ***· Summarize** : In light of the growing popularity of Exploratory Data Analysis (EDA), understanding the underlying causes of the knowledge acquired by EDA is crucial. However, it remains under-researched. This study promotes a transparent and explicable perspective on data analysis, called eXplainable Data Analysis (XDA). For this reason, we present XInsight, a general framework for XDA. XInsight provides data analysis with qualitative and quantitative explanations of causal and non-causal semantics. This way, it will significantly improve human understanding and confidence in the outcomes of data analysis, facilitating accurate data interpretation and decision making in the real world. XInsight is a three-module, end-to-end pipeline designed to extract causal graphs, translate causal primitives into XDA semantics, and quantify the quantitative contribution of each explanation to a data fact. XInsight uses a set of design concepts and optimizations to address the inherent difficulties associated with integrating causality into XDA. Experiments on synthetic and real-world datasets as well as a user study demonstrate the highly promising capabilities of XInsight.*


***2. Evaluation Methods and Measures for Causal Learning Algorithms （Arizona State University, 2022.07）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page]() | [Arxiv-Page](https://arxiv.org/pdf/2202.02896v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/evaluation-methods-and-measures-for-causal)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Causality Learning`***
>
> ***· Summarize** : In this survey, we review commonly-used datasets, evaluation methods, and measures for causal learning using an evaluation pipeline similar to conventional machine learning. We focus on the two fundamental causal-inference tasks and causality-aware machine learning tasks. Limitations of current evaluation procedures are also discussed. We then examine popular causal inference tools/packages and conclude with primary challenges and opportunities for benchmarking causal learning algorithms in the era of big data. The survey seeks to bring to the forefront the urgency of developing publicly available benchmarks and consensus-building standards for causal learning evaluation with observational data. In doing so, we hope to broaden the discussions and facilitate collaboration to advance the innovation and application of causal learning.*

***3. Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond （Israel Institute of Technology, 2021.09）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/knowlab/bi-weekly-paper-presentation) | [Arxiv-Page](https://arxiv.org/pdf/2109.00725v2.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/causal-inference-in-natural-language)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Causal Inference`***
>
> ***· Summarize** : A fundamental goal of scientific research is to learn about causal relationships. However, despite its critical role in the life and social sciences, causality has not had the same importance in Natural Language Processing (NLP), which has traditionally placed more emphasis on predictive tasks. This distinction is beginning to fade, with an emerging area of interdisciplinary research at the convergence of causal inference and language processing. Still, research on causality in NLP remains scattered across domains without unified definitions, benchmark datasets and clear articulations of the challenges and opportunities in the application of causal inference to the textual domain, with its unique properties. In this survey, we consolidate research across academic areas and situate it in the broader NLP landscape. We introduce the statistical challenge of estimating causal effects with text, encompassing settings where text is used as an outcome, treatment, or to address confounding. In addition, we explore potential uses of causal inference to improve the robustness, fairness, and interpretability of NLP models. We thus provide a unified overview of causal inference for the NLP community.*

***4. Deep End-to-end Causal Inference（University of Massachusetts Amherst etc, 2022.04）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/microsoft/causica) | [Arxiv-Page](https://arxiv.org/pdf/2202.02195v2.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/deep-end-to-end-causal-inference)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Causal Inference`***
>
> ***· Summarize** : Causal inference is essential for data-driven decision making across domains such as business engagement, medical treatment and policy making. However, research on causal discovery has evolved separately from inference methods, preventing straight-forward combination of methods from both fields. In this work, we develop Deep End-to-end Causal Inference (DECI), a single flow-based non-linear additive noise model that takes in observational data and can perform both causal discovery and inference, including conditional average treatment effect (CATE) estimation. We provide a theoretical guarantee that DECI can recover the ground truth causal graph under standard causal discovery assumptions. Motivated by application impact, we extend this model to heterogeneous, mixed-type data with missing values, allowing for both continuous and discrete treatment decisions. Our results show the competitive performance of DECI when compared to relevant baselines for both causal discovery and (C)ATE estimation in over a thousand experiments on both synthetic datasets and causal machine learning benchmarks across data-types and levels of missingness.*


***5. Benchmarking Framework for Performance-Evaluation of Causal Inference Analysis （IBM Research, 2018.02）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework) | [Arxiv-Page](https://arxiv.org/pdf/1802.05046.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/benchmarking-framework-for-performance)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Causal Inference`** , **`Benchmark`** , **`Dataset`***
>
> ***· Summarize** : Causal inference analysis is the estimation of the effects of actions on outcomes. In the context of healthcare data this means estimating the outcome of counter-factual treatments (i.e. including treatments that were not observed) on a patient's outcome. Compared to classic machine learning methods, evaluation and validation of causal inference analysis is more challenging because ground truth data of counter-factual outcome can never be obtained in any real-world scenario. Here, we present a comprehensive framework for benchmarking algorithms that estimate causal effect. The framework includes unlabeled data for prediction, labeled data for validation, and code for automatic evaluation of algorithm predictions using both established and novel metrics. The data is based on real-world covariates, and the treatment assignments and outcomes are based on simulations, which provides the basis for validation. In this framework we address two questions: one of scaling, and the other of data-censoring. The framework is available as open source code at https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework.*


***6. e-CARE: a New Dataset for Exploring Explainable Causal Reasoning （Harbin Institute of Technology, 2022.05）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/waste-wood/e-care) | [Arxiv-Page](https://arxiv.org/pdf/2205.05849v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/e-care-a-new-dataset-for-exploring-1)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Causal Inference`** , **`Casual Explaniation`** , **`Natural Language Dataset`***
>
> ***· Summarize** : Understanding causality has vital importance for various Natural Language Processing (NLP) applications. Beyond the labeled instances, conceptual explanations of the causality can provide deep understanding of the causal facts to facilitate the causal reasoning process. However, such explanation information still remains absent in existing causal reasoning resources. In this paper, we fill this gap by presenting a human-annotated explainable CAusal REasoning dataset (e-CARE), which contains over 21K causal reasoning questions, together with natural language formed explanations of the causal questions. Experimental results show that generating valid explanations for causal facts still remains especially challenging for the state-of-the-art models, and the explanation information can be helpful for promoting the accuracy and stability of causal reasoning models.*

---

### 🗃 Works of Large Language Model Based Agents

***1. Agents: An Open-source Framework for Autonomous Language Agents（AIWaves Inc. & Zhejiang University, 2023.09）*** : 


> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/aiwaves-cn/agents ) | [Arxiv-Page](https://arxiv.org/pdf/2309.07870.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/agents-an-open-source-framework-for) |[OfficialWeb-Page](http://www.aiwaves-agents.com/) | |
>
> ***· Tags** :*
>
> ***· Summarize** : None*


***2. Generative Agents: Interactive Simulacra of Human Behavior（Stanford University, 2023.04）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/joonspk-research/generative_agents) | [Arxiv-Page](https://arxiv.org/pdf/2304.03442v2.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/generative-agents-interactive-simulacra-of)  | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Multi-Agents`** , **`AI-Town`** , **`Techniques`***
>
> ***· Summarize** : Multi-Agents is an attractive research direction now. This project construct a Agents Society that includes 25 LLM-based Agents. Researchers provides the design methods of this AI-Agents Towns and technique details, evaluation methods. This thesis is extremely important for LLM-based Multi-Agents Projects.*


***3. Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf（Tsinghua University, 2023.09）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Is Read? |
> |-------------|------------|--------------------|------------|
> | Github-Page | [Arxiv-Page](https://arxiv.org/pdf/2309.04658v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/exploring-large-language-models-for) |  |
>
> ***· Tags** :*
>
> ***· Summarize** : An paper about Multi-Agents.*

---

### 📂 Retrieval LLM Work

***1. REPLUG: Retrieval-Augmented Black-Box Language Models（University of Washington etc, 2023.01）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | Github-Page | [Arxiv-Page](https://arxiv.org/pdf/2301.12652v4.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/replug-retrieval-augmented-black-box-language) | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Retrieval`** , **`LLM Black Box`***
>
> ***· Summarize** : We introduce REPLUG, a retrieval-augmented language modeling framework that treats the language model (LM) as a black box and augments it with a tuneable retrieval model. Unlike prior retrieval-augmented LMs that train language models with special cross attention mechanisms to encode the retrieved text, REPLUG simply prepends retrieved documents to the input for the frozen black-box LM. This simple design can be easily applied to any existing retrieval and language models. Furthermore, we show that the LM can be used to supervise the retrieval model, which can then find documents that help the LM make better predictions. Our experiments demonstrate that REPLUG with the tuned retriever significantly improves the performance of GPT-3 (175B) on language modeling by 6.3%, as well as the performance of Codex on five-shot MMLU by 5.1%.*


***2. Atlas: Few-shot Learning with Retrieval Augmented Language Models（Meta AI , 2022.08）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/facebookresearch/atlas) | [Arxiv-Page](https://arxiv.org/pdf/2208.03299v3.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/few-shot-learning-with-retrieval-augmented) | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Retrieval`***
>
> ***· Summarize** : *


***3. REALM: Retrieval-Augmented Language Model Pre-Training （University of Washington etc, 2020.10）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/google-research/language/tree/master/language/realm) | [Arxiv-Page](https://arxiv.org/pdf/2002.08909v1.pdf) | [PaperWithCode-Page](https://github.com/google-research/language/tree/master/language/realm) | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Retrieval`***
>
> ***· Summarize** : Language model pre-training has been shown to capture a surprising amount of world knowledge, crucial for NLP tasks such as question answering. However, this knowledge is stored implicitly in the parameters of a neural network, requiring ever-larger networks to cover more facts. To capture knowledge in a more modular and interpretable way, we augment language model pre-training with a latent knowledge retriever, which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference. For the first time, we show how to pre-train such a knowledge retriever in an unsupervised manner, using masked language modeling as the learning signal and backpropagating through a retrieval step that considers millions of documents. We demonstrate the effectiveness of Retrieval-Augmented Language Model pre-training (REALM) by fine-tuning on the challenging task of Open-domain Question Answering (Open-QA). We compare against state-of-the-art models for both explicit and implicit knowledge storage on three popular Open-QA benchmarks, and find that we outperform all previous methods by a significant margin (4-16% absolute accuracy), while also providing qualitative benefits such as interpretability and modularity.*

---

### 📂 Datasets and Data Engineering

***1. Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models （The Hong Kong University of Science and Technology & AIWaves, 2023.10）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/rickyskywalker/synthesis_step-by-step_official) | [Arxiv-Page](https://arxiv.org/pdf/2310.13671v1.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/let-s-synthesize-step-by-step-iterative) | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Data Generation`***
>
> ***· Summarize** : *Data Synthesis* is a promising way to train a small model with very little labeled data. One approach for data synthesis is to leverage the rich knowledge from large language models to synthesize pseudo training examples for small models, making it possible to achieve both data and compute efficiency at the same time. However, a key challenge in data synthesis is that the synthesized dataset often suffers from a large distributional discrepancy from the *real task* data distribution. Thus, in this paper, we propose *Synthesis Step by Step* (**S3**), a data synthesis framework that shrinks this distribution gap by iteratively extrapolating the errors made by a small model trained on the synthesized dataset on a small real-world validation dataset using a large language model. Extensive experiments on multiple NLP tasks show that our approach improves the performance of a small model by reducing the gap between the synthetic dataset and the real data, resulting in significant improvement compared to several baselines: 9.48% improvement compared to ZeroGen and 2.73% compared to GoldGen, and at most 15.17% improvement compared to the small model trained on human-annotated data.*


***2. Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs（The University of Hong Kong & DAMO etc, 2023.05）*** : 

> | Github Link | Paper Link | PaperWithCode Link | Official Website Link | Is Read? |
> |-------------|------------|--------------------|--------------------|------------|
> | [Github-Page](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) | [Arxiv-Page](https://arxiv.org/pdf/2305.03111.pdf) | [PaperWithCode-Page](https://paperswithcode.com/paper/can-llm-already-serve-as-a-database-interface) | OfficialWeb-Page | ✔ |
>
> ***· Tags** : **`Benchmark`** , **`Text-to-SQL`***
>
> ***· Summarize** : Text-to-SQL parsing, which aims at converting natural language instructions into executable SQLs, has gained increasing attention in recent years. In particular, Codex and ChatGPT have shown impressive results in this task. However, most of the prevalent benchmarks, i.e., Spider, and WikiSQL, focus on database schema with few rows of database contents leaving the gap between academic study and real-world applications. To mitigate this gap, we present Bird, a big benchmark for large-scale database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domains. Our emphasis on database values highlights the new challenges of dirty database contents, external knowledge between NL questions and database contents, and SQL efficiency, particularly in the context of massive databases. To solve these problems, text-to-SQL models must feature database value comprehension in addition to semantic parsing. The experimental results demonstrate the significance of database values in generating accurate text-to-SQLs for big databases. Furthermore, even the most effective text-to-SQL models, i.e. ChatGPT, only achieves 40.08% in execution accuracy, which is still far from the human result of 92.96%, proving that challenges still stand. Besides, we also provide an efficiency analysis to offer insights into generating text-to-efficient-SQLs that are beneficial to industries. We believe that BIRD will contribute to advancing real-world applications of text-to-SQL research. The leaderboard and source code are available: https://bird-bench.github.io/.*


