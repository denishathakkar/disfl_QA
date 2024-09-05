# Disfl-QA: A Benchmark Dataset for Understanding Disfluencies in Question Answering

Disfl-QA is a targeted dataset for contextual disfluencies in an information seeking  setting, namely question answering over Wikipedia passages.  Disfl-QA builds upon the SQuAD-v2 ([Rajpurkar et al., 2018](https://www.aclweb.org/anthology/P18-2124/)) dataset, where each question in the dev set is annotated to add a contextual disfluency using the paragraph as a source of distractors.

The final dataset consists of ~12k (disfluent question, answer) pairs. Over 90\% of the disfluencies are corrections or restarts, making it a much harder test set for disfluency correction. Disfl-QA aims to fill a major gap between speech and NLP research community. We hope the dataset can serve as a benchmark dataset for testing robustness of models against disfluent inputs. 

Our expriments reveal that the state-of-the-art models are brittle when subjected to disfluent inputs from Disfl-QA. Detailed experiments and analyses can be found in our [paper](https://arxiv.org/pdf/2106.04016.pdf).

## Dataset Description
Disfl-QA consists of ~12k disfluent questions with the following train/dev/test splits:
| File      | Questions   |
|-----|-----|
|train.json  | 7182  |
|dev.json  | 1000   |
|test.json  | 3643  |


## Citation
If you use or discuss this dataset in your work, please cite it as follows:

```
@inproceedings{gupta-etal-2021-disflqa,
    title = "{Disfl-QA: A Benchmark Dataset for Understanding Disfluencies in Question Answering}",
    author = "Gupta, Aditya and Xu, Jiacheng and Upadhyay, Shyam and Yang, Diyi and Faruqui, Manaal",
    booktitle = "Findings of ACL",
    year = "2021"
}
```

## License
Disfl-QA dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

If you have a technical question regarding the dataset or publication, please create an issue in this repository.
