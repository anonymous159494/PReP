import os
import json
import time
import pandas as pd
from modules.Dataset import *
from modules.LLM import configure_openai, configure_llava
from global_function import convert_json_key, bfs

from agents.agent import Agent_PReP
from agents.agent_no_planning import Agent_PReP_NoPlanning
from agents.agent_no_reflection import Agent_PReP_NoReflection
from agents.agent_plain import Agent_PReP_Plain
from agents.agent_oracle import Agent_PReP_Oracle

from agents.agent_CaP import Agent_Cap
from agents.agent_CoT import Agent_CoT
from agents.agent_ProgPrompt import Agent_ProgPrompt
from agents.agent_InnerMonologue import Agent_InnerMonologue
from agents.agent_DEPS import Agent_DEPS

agent_list = [Agent_PReP,   # Agent Full 
            Agent_PReP_NoReflection, Agent_PReP_NoPlanning, Agent_PReP_Plain,   # Agent Ablation
            Agent_CoT, Agent_Cap, Agent_ProgPrompt, Agent_InnerMonologue, Agent_DEPS,       # Agent Varicant
            Agent_PReP_Oracle, Agent_PReP    # Agent Perception Ablation
        ]


class Metrics:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        if os.path.exists(self.cache_path):
            self.df = pd.read_csv(self.cache_path)
            self.tested = [(sta, des) for sta, des in zip(list(self.df['sta_id']),list(self.df['des_id']))]
        else:
            self.df = None
            self.tested = []
        self.metric = {
            "sta_id" : [],
            "des_id" : [],
            "ref_path": [],
            "ref_len" : [],
            "agent_path": [],
            "agent_len" : [],
            "agent_trace": [],
            "agent_state": [],
            "agent_success": [],
            "token_cost": []
        }
    
    def update_metrics_s(self, sta_id, des_id, ref_path):
        self.metric["sta_id"].append(sta_id)
        self.metric["des_id"].append(des_id)
        self.metric["ref_path"].append(ref_path)
        self.metric["ref_len"].append(len(ref_path)-1)

    def update_metrics_e(self, a):
        self.metric["agent_path"].append(a.path)
        self.metric["agent_trace"].append(a.trace)
        self.metric["agent_len"].append(len(a.trace))
        if a.state == "Success":
            self.metric["agent_state"].append("Success")
            self.metric["agent_success"].append(1)
        else:
            self.metric["agent_state"].append("Break") 
            self.metric["agent_success"].append(0)
        self.metric["token_cost"].append(a.token_counts)

    def cal_metrics(self):
        # Success Rate
        self.sr = sum(self.metric['agent_success']) / len(self.metric['agent_success'])

        # SPL
        temp_s = [b * l / max(l, r) for b, l, r in zip(self.metric['agent_success'], self.metric['ref_len'], self.metric['agent_len'])]
        self.spl = sum(temp_s) / len(temp_s)

        # token_cost 
        self.tokens = sum(self.metric['token_cost'])

    def save_metrics(self):
        df2 = pd.DataFrame(self.metric)
        if self.df is not None:
            df = pd.concat([self.df, df2], ignore_index=True)
        else:
            df = df2
        df.to_csv(self.cache_path)


def agent_test(city, llm_model, test_label, mode=0):

    # load dataset and testset
    dataset_cache = f"dataset/{city}/env_{city}.pkl"    # dataset
    svcache_path = f"dataset/{city}/svdat/"             # street views
    testset_path = f"testset/{city}_testset_1.json"

    env_dict = load_dataset(dataset_cache)     
    with open(testset_path, "r") as f:
        test_dict = json.load(f)
    test_dict = convert_json_key(test_dict)

    # configure models

    if "gpt" in llm_model['name']:
        configure_openai(llm_model['name'])
    else:
        configure_openai(model=llm_model['name'], is_openai=False, ip_port=llm_model['api_base'])

    if mode == 10:
        ablation_mode = "without_finetune"
        configure_llava("dataset/llava_response_local/llava_response_local_bf.json")
    else:
        ablation_mode = "normal"
        configure_llava("dataset/llava_response_local/llava_respnese_local.json")

    # metric file init
    metric_file = f"metric_files/metric_test_{llm_model['name']}_{city}_{test_label}_1.csv"
    metrics = Metrics(metric_file)

    # agent init
    testAgent = agent_list[mode]

    # test
    sum_count = 0
    for des_id in test_dict.keys():

        des = env_dict[des_id]
        des.landmark = test_dict[des_id]['landmark']

        sta_list = test_dict[des_id]['sta']

        for sta_id_dict in sta_list[:]:
            sta_id = sta_id_dict['id']
            if (sta_id, des_id) in metrics.tested: continue

            ref_path = bfs(env_dict, sta_id, des_id)
            limit_count = (len(ref_path) - 1) * 2.5

            metrics.update_metrics_s(sta_id, des_id, ref_path)

            sum_count = sum_count + 1

            log_file_name = f"log_data/{city}/{llm_model['name']}_{sta_id}_{des_id}_{test_label}.jsonl"

            a = testAgent(env_dict, sta_id, des_id, svcache_path, logpath=log_file_name, ablation_mode=ablation_mode)

            while a.state != "Success" and a.count < limit_count:
                t_start = time.time()
                try:
                    a.step()
                except Exception as e:
                    print(f"Agent Break from {e}")
                    a.state = "Break"
                    break

                cur_id = a.location.id
                if len(bfs(env_dict, cur_id, des_id)) - 1 + len(a.trace) > limit_count:
                    break
                t_end = time.time()
                print("One step time: ", t_end-t_start, "s")

            metrics.update_metrics_e(a)
            metrics.cal_metrics()

            print(f"Task {sum_count} ")
            print("Agent move:", "-> ".join(a.trace))
            print("Agent path:",a.path, len(a.path)-1)
            print("Refed Path", ref_path, len(ref_path)-1)

            print("---------------------------")
            print("Success Rate: ", metrics.sr)
            print("SPL: ", metrics.spl)
            print("Token Cost: ", metrics.tokens)
            metrics.save_metrics()

        # if sum_count >= 2: break

