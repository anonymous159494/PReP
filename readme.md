# Perceive, Reflect, and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions

![img](images\architecture.png)

We propose a LLM agent for goal-directed navigation in complex city environment without instructions. 

The code and dataset are available in this repository.

```
git clone https://github.com/anonymous159494/PReP.git
cd PReP
pip install -r requirements.txt
```

To enable you to start the experiment faster, we save the LLaVA responses involved in the experiment (both before and after fine-tuning) as local files to be called directly, see <dataset/llava_response_local/>.

In <modules/LLM.py>, you should add your api_key;
```
openai_api_key = "sk-xxxxx"     # if you use openai GPT API
others_api_key = "xxxxxxxx"     # if you use other LLM API
```

You can run main.py to reproduce the experiments easily by selecting different parameters.

The metrics would be saved to Fold metric_files, and the LLM request-repsonse data would be saved to Fold log_data. 