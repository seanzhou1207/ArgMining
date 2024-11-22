# ArgMining: Exploring Novelty and Relevance in Arguments

![rgpt](https://github.com/seanzhou1207/ArgMining/blob/main/Images/RGPT1.png "RGPT: Our recurrent ensembling of kth GPT.")

Given a topic, a premise, and many conclusions, which ones are the most valid and original? To tackle this problem in the field of argument mining, we experiment with both feature-based models and fine-tuned GPT models. Our approach uses topic data and a specialized SBERT model for understanding language to extract important features. We also incorporate knowledge graphs to identify relationships between premises and conclusions. Combining knowledge graph features with SBERT embeddings delivered the best performance.
Additionally, we explored fine-tuned GPT models like Instruction GPT and introduced a new framework called Recurrent GPT with a prompting chain. Our top model, which combines Recurrent GPT and Instruction GPT, significantly outperformed the RoBERTa baseline and achieved state-of-the-art results compared to existing models.

Collaborators: Zhengxing Cheng, Owen Lin, Sean Zhou

See full report [here](https://github.com/seanzhou1207/ArgMining/tree/main/Argmining_Report.pdf). 