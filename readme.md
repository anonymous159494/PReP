## Perceive, Reflect and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions


To help you reproduce the experiment results,  the code and dataset are available in this repository. 

```
git clone https://anonymous.4open.science/r/PReP-13B5.git
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

---




**Abstract**  This paper considers a scenario in city navigation: an AI agent is provided with language descriptions of the goal location with respect to some well-known landmarks; By only observing the scene around, including recognizing landmarks and road network connections, the agent has to make decisions to navigate to the goal location without instructions. This problem is very challenging, because it requires agent to establish self-position and acquire spatial representation of complex urban environment, where landmarks are often invisible. In the absence of navigation instructions, such abilities are vital for the agent to make high-quality decisions in long-range city navigation. With the emergent reasoning ability of large language models (LLMs), a tempting solution is to prompt LLMs to “react” on each observation and make decisions accordingly. However, this solution has very poor performance that the agent often repeatedly visits same locations and make short-sighted, inconsistent decisions. To address these issues, this paper introduces a novel agentic workflow featured by its abilities to *perceive*, *reflect* and *plan*. Specifically, we find LLaVA-7B can be fine-tuned to *perceive* the approximate direction and distance of landmarks with sufficient accuracy for city navigation. Moreover, *reflection* is achieved through a memory mechanism, where past experiences are stored and can be retrieved with current perception for effective decision argumentation. *Planning* uses reflection results to produce long-term plans, which can avoid short-sighted decisions in long-range navigation. We show the designed workflow significantly improves urban navigation ability of the LLM agent compared with the state-of-the-art baselines. 

![problem illustration](images/problem0.png)



This paper proposes an effective agentic workflow that improves the spatial cognitive ability of LLMs thus improving the goal-directed city navigation performance. We fine-tune LLaVA and find it can perceive the direction and distance of landmarks with sufficient accuracy for navigation.  Inspired by the theory of human cognition, we propose a memory scheme to help the agent form the cognitive map. The historical trajectories and observations are stored and summarized to learn an intrinsic spatial representation of the environment, *i.e.*, an internal city map. The agent combines the historical experience and current observation to evaluate current situation and infer the goal direction. To improve over short-sighted actions, we resort to long-term planning. Specifically, considering the reflections and current road network connection, the agent decompose the full path into several sub-goals, ensuring consistent and reasonable movement to the final goal during long-range navigation. These components form the '*Perceive*, *Reflect*, and *Plan*' workflow which allows the agent to perform long-range city navigation.

![Overview of the PReP Workflow](images/workflow0.png)

We collect navigation datasets reflecting CBD scenes in four cities——Beijing, Shanghai, New York and Paris. They contain complex road networks with thousands of road nodes and street view images. On the four datasets, the proposed workflow significantly outperforms methods that could be applied (but are not specific) to our task, achieving success rate of 54\% on the avergae of the four city test sets. We find the perception component produces accurate spatial relations to support city navigation, the success rate of which is only 5% lower than navigation with ground truth perception results.  Besides, we show that reflection and planning can help the LLM agent to form the cognitive map and further contribute to the success rate and make it useful when dealing with long-range navigation tasks.

![task and dataset example](images/task_dataset1.png)

![main results](images/main_results_table.png)

![further results](images/further_results.png)

