<div align="center">
  
## Benchmarking Visual LLMs Resilience to Unanswerable Questions on Visually Rich Documents

### Under Review
</div>
<!-- A comprehensive evaluation system for testing Visual Large Language Models' (VLLMs) robustness against corrupted questions in document understanding tasks. The framework introduces systematic corruptions at NLP, document element, and layout levels, while providing tools for corruption generation, unanswerability verification, and performance assessment. Validated through experiments on 2 benchmarks with 12 VLLMs, it offers specialized metrics for measuring No Answer precision, document element impact, and answer correlations. The project specifically addresses challenges in multi-page document processing, including handling of multimodal elements and varied layouts. -->

## Datasets

|                   | MPDocVQA | DUDE |
|-------------------|:--------:|:----:|
| Full              | [link*](https://rrc.cvc.uab.es/?ch=17&com=downloads) | [link*](https://rrc.cvc.uab.es/?ch=23&com=downloads) |
| Reduced           | [link](https://drive.google.com/drive/folders/1-SZzvuMJarRDi4rTz6svkVP8MsWTCejO?usp=drive_link) | [link](https://drive.google.com/drive/folders/1URFqchC37AoGMkl0HQP22oAeqM-lV2ns?usp=drive_link) |
| Corrupted         | [link](https://drive.google.com/drive/folders/1bMjgHAiBJTwDAZu589abNCaMTWKIOXtq?usp=drive_link) | [link](https://drive.google.com/drive/folders/11Yd9l1J-f0FB-E8S5ZTPrSse3Vjie_wl?usp=drive_link) |
| Verified          | [link](https://drive.google.com/drive/folders/1fcwycWWO2D9hRjrididVcSXoy6GyPac6?usp=drive_link) | [link](https://drive.google.com/drive/folders/12ltYWllJAoEIkJlbZegnWrrYSul9K6Oy?usp=drive_link) |

\* Link to original dataset repository

The "Reduced" row contains:
- the subsets of questions taken from the full datasets,
- the OCR and Layout analysis
- the dataset augmented
  
The "Corrupted" row contains corrupted questions.<br />
The "Verified" row contain the corrupted question verified by the Judge model.

In future, we will release the full datasets processed.

We also provide the zip files of with all datasets at this links: [MPDocVQA.zip](https://drive.google.com/file/d/1Qn4zG_nCnx0sebhTBHKHpFH41-OEsex2/view?usp=drive_link), [DUDE.zip](https://drive.google.com/file/d/1JNIB-a1vvXjWDaDedX8JsdioOVAs1_03/view?usp=drive_link)


## Models
### Current
|                   | Type | Size | License      | Link |
|-------------------|:----:|:----:|:------------:|:----:|
| QWEN 2.5-VL       | VLLM | 7B   | Apache 2.0   |  [link](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| QWEN 2.5-VL       | VLLM | 72B  | Apache 2.0   |  [link](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) |
| InternVL 3        | VLLM | 9B   | MIT          |  [link](https://huggingface.co/OpenGVLab/InternVL3-9B) |
| InternVL 3        | VLLM | 78B  | MIT          |  [link](https://huggingface.co/OpenGVLab/InternVL3-78B) |
| Phi 4 Multimodal  | VLLM | 5B   | MIT          |  [link](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) |
| Molmo             | VLLM | 7B   | Apache 2.0   |  [link](https://huggingface.co/allenai/Molmo-7B-D-0924) |
| Ovis              | VLLM | 9B   | Apache 2.0   |  [link](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-9B) |
| Gemma 3           | VLLM | 27B  | Gemma        |  [link](https://huggingface.co/google/gemma-3-27b-it) |
| Llama 3.2         | VLLM | 11B  | Llama3.2     |  [link](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision) |
| Llava 1.6         | VLLM | 34B  | Apache 2.0   |  [link](https://huggingface.co/liuhaotian/llava-v1.6-34b) |

### Discontinued*
|                   | Type | Size | License      | Link |
|-------------------|:----:|:----:|:------------:|:----:|
| InternVL 2.5      | VLLM | 8B   | MIT          |  [link](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |
| Phi 3.5 Vision    | VLLM | 4B   | MIT          |  [link](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) |
| DocOwl2           | VLLM | 8B   | Apache 2.0   |  [link](https://huggingface.co/mPLUG/DocOwl2) |
| UDOP              | VLM  | 742M | MIT          |  [link](https://huggingface.co/microsoft/udop-large) |
| LayoutLMv3        | VLM  | 125M | CC-BY-NC-SA-4.0      |  [link](https://huggingface.co/rubentito/layoutlmv3-base-mpdocvqa) |
| BLIP              | VLM  | 385M | BSD-3-Clause |  [link](https://huggingface.co/Salesforce/blip-vqa-base) |

*we are currently developing an online leaderboard to easily keep track of all models performance

### Info
Gemini 2.5 Flash has been used as judge on all verification steps

After each model is tested, it is postprocess to standardize results.
The final output file has the same verified dataset elements extended with models answers.
In detail it has a list with:
-  document pages taken in the window to provide an answer
-  model answer
-  standardize model answer

## Execution

To run the experiments, install the dependencies in the `requirements.txt` file.<br />
Currently, the experiments have been executed with `transformers==4.49.0.dev0`. 
For Llama3.2 and Llava1.6 we deploy the models using the [Ollama](https://ollama.com/) library

An example on how to run each step is provided in the notebook [example.ipynb](example.ipynb)

## Future Works
We plan to test more models with different sizes, both open source and private.
Develop a website with all model for easy comparison

## License
This project is licensed under the **CC BY-NC 4.0**. See [LICENSE](LICENSE) for more information.

## Contact
If you are interested in this work or have questions about the code, the paper or the dataset, please contact us by email at vqacorruption@gmail.com.

## Citation

If you find this project useful, please consider citing:
```bibtex
```
<!--
**vqacorruption/vqacorruption** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
