import math
import re
import json
import time
import random
from dataset.landmark import landmark
from modules.LLM import *
from modules.Dataset import Location
from modules.long_term_memory import NetworkManager
from global_function import angle2dir, ang2vec, vec2ang
from global_function import get_realangle, angle_average
from global_function import cal_angle2, cal_distance

from dataset.bj_lm_recog import get_res_bj
from dataset.sh_lm_recog import get_res_sh
from dataset.pr_lm_recog import get_res_pr
from dataset.ny_lm_recog import get_res_ny


class Agent_PReP_NoPlanning:
    def __init__(
            self,
            loc_dict: dict[Location],  # environment
            sta: int,  # starter
            des: int,  # destination
            cache: str,  # street view cache
            logpath: str = None,  # path to save log
            ablation_mode: str = "normal"  # for ablation
    ):

        # location
        self.map = loc_dict
        self.location = loc_dict[sta]
        self.destination = loc_dict[des]

        # perceive
        self.mode = 0
        self.observation = None
        self.action_space = None

        # memory
        self.LTM = NetworkManager(logpath)
        self.retrieved = None
        self.location_memory = {}
        self.direction_memory = {}
        self.trajectory_memory = []

        self.trace = []
        self.path = [sta]
        self.plan = "None"
        self.action = None

        self.reflect_direction = None
        self.anticipate_direction = None
        self.face_direction = None
        self.distance = None

        # others
        self.to_print = False
        self.token_counts = 0
        self.query_count = 0
        self.count = 0
        self.cache = cache
        self.log_filepath = logpath
        self.state = "Init"
        self.ablation_mode = ablation_mode
        self.LTM_des = None

    def get_actionspace(self) -> dict[str:list]:
        """
        get the action space of current position

        output:
            action_space: {Direction: [Node_list]}
        """
        self.action_space = {}
        for connect, angle in self.location.connect:
            direction = angle2dir(angle)
            if direction not in self.action_space.keys():
                self.action_space[direction] = [connect]
            else:
                self.action_space[direction].append(connect)

        if self.to_print:
            print("\n(function) get_actionspace: ")
            print(self.action_space)
        return self.action_space

    def get_observation(self, mode="normal") -> dict:
        """
        get the visual observation of current position
        and use llava to transform vision to language
        """
        lm_list = []
        if "beijing" in self.log_filepath:
            lm_recog = get_res_bj()
            if self.location.id in lm_recog.keys():
                lm_list = lm_recog[self.location.id]
        if "shanghai" in self.log_filepath:
            lm_recog = get_res_sh()
            if self.location.id in lm_recog.keys():
                lm_list = lm_recog[self.location.id]
        if "paris" in self.log_filepath:
            lm_recog = get_res_pr()
            if self.location.id in lm_recog.keys():
                lm_list = lm_recog[self.location.id]
        if "newyork" in self.log_filepath:
            lm_recog = get_res_ny()
            if self.location.id in lm_recog.keys():
                lm_list = lm_recog[self.location.id]

        self.observation = {}
        img_path, _ = self.location.get_streetview(self.cache, path_only=True)
        for i, img in enumerate(img_path):
            for lm_id_s in lm_list:
                lm_id = int(lm_id_s)
                lm = landmark[lm_id]
                # llava predict
                text1 = f"""Is the {lm['en']} visible in the image?"""
                response1 = llava_predict_local(img, text1, port=5000, path_only=True)
                predict_angle = None
                predict_distance = None
                if response1 is not None and "yes" in response1.lower():
                    try:
                        # the bounding box of landmark in the image
                        if mode == "without_finetune":
                            voc = response1.split("the")[1].split('of')[0].strip()
                        else:
                            voc = response1.split('(')[1].split(')')[0]
                        box = [float(v) for v in voc.split(',')]

                        # use llava to estimate the real distance
                        if mode == "without_finetune":
                            text2 = f"""The {lm['en']} is visible in the image and its voc bounding box is {voc}. How far is that place actually from the camera?"""
                        else:
                            text2 = f"""The {lm['en']} is visible in the image and its bounding box is ({voc}). How far is that place away from the camera?"""
                        response2 = llava_predict_local(img, text2, port=5000, path_only=True)

                        heading = self.location.connect[i][1]
                        predict_angle = get_realangle(box, heading)
                        if mode == "without_finetune":
                            predict_distance = int(response2.split(' meters')[0].split(' ')[-1])
                        else:
                            predict_distance = int(float(response2.split('about ')[1].split(' ')[0]))
                        print(predict_distance)

                        real_angle = cal_angle2(self.location.bdxy, lm['bdxy'])
                        real_distance = cal_distance(self.location.bdxy, lm['bdxy'])

                        if self.to_print:
                            print("\n(Function)get_obversation: ")
                            print(text1, response1)
                            print(text2, response2)
                            print("predict: ", predict_angle, predict_distance)
                            print("reality: ", real_angle, real_distance)
                            messages = [
                                {"role": "system", "content": img},
                                {"role": "user", "content": text1},
                                {"role": "assitant", "content": response1},
                                {"role": "user", "content": text2}
                            ]
                            write_gpt_data(messages, response2, output_path=self.log_filepath,
                                           task_name="llava_predict", model_name="llava")

                    except Exception as e:
                        print(f"Error from ({e})")
                        # raise RuntimeError("Llava predice not response")

                if predict_angle is not None and predict_distance is not None:
                    if lm_id not in self.observation.keys():
                        self.observation[lm_id] = {}
                        self.observation[lm_id]["angle"] = predict_angle
                        self.observation[lm_id]["distance"] = predict_distance
                    else:
                        self.observation[lm_id]["angle"] = angle_average(
                            [self.observation[lm_id]["angle"], predict_angle])
                        self.observation[lm_id]["distance"] = (self.observation[lm_id][
                                                                   "distance"] + predict_distance) / 2

        return self.observation

    def direction_reasoning(self) -> str:
        """
        infer the goal direction based on landmark information,
        """
        # calculate directions based on landmark information
        angles = []
        directions = []
        distances = []
        for lm_id in self.observation.keys():
            lm = landmark[lm_id]
            lm_ang_from_cur = self.observation[lm_id]['angle']
            lm_dis_from_cur = self.observation[lm_id]['distance']

            vec1 = ang2vec(lm_ang_from_cur, lm_dis_from_cur)  # cur -> landmarkA
            for lm in self.destination.landmark.keys():
                lm_ang_from_tar = self.destination.landmark[lm]['angle']
                lm_dis_from_tar = self.destination.landmark[lm]['distance']
                vec2 = ang2vec(lm_ang_from_tar, lm_dis_from_tar)  # goal -> landmarkB

                vec3 = [j - i for i, j in zip(landmark[lm_id]['bdxy'], landmark[lm]['bdxy'])]  # landmarkA -> landmarkB

                vec = [vec1[0] - vec2[0] + vec3[0], vec1[1] - vec2[1] + vec3[1]]

                angle = vec2ang(vec)
                direct = angle2dir(angle)
                distance = math.sqrt(vec[0] ** 2 + vec[1] ** 2)

                print(angle, direct, distance)
                angles.append(angle)
                directions.append(direct)
                distances.append(distance)

        # decide the target direction
        if len(directions) == 0:
            self.face_direction = None
        elif len(directions) == 1:
            self.face_direction = directions[0]
        else:
            angle = angle_average(angles)
            self.face_direction = angle2dir(angle)
            # or randomly choose one direction
            # self.face_direction = random.choice(directions)

        if len(distances):
            self.distance = sum(distances) / len(distances)

        if self.to_print:
            print("\n", "(function) direction_reasoning: ")
            print("predict direction: ", self.face_direction, self.distance)
            print("real direction: ", angle2dir(cal_angle2(self.location.bdxy, self.destination.bdxy)),
                  cal_distance(self.location.bdxy, self.destination.bdxy))

        return self.face_direction

    def retrieve(self):
        directions = list(self.action_space.keys())
        self.retrieved = self.LTM.retrieve(directions)

    def extract_direction(self, answer, space):
        text = re.findall("[A-z-]+", answer)
        answer_list = [i.lower() for i in text]
        for dir in space:
            if dir.lower() in answer_list:
                return dir
        return None

    # TODO: 新增了一个函数，需要方位推理模块输出的方向，以及额外输出一个当前到目标的距离，浮点数
    def dire_dis_transfer(self, direction, distance):
        a = round(distance / 50)
        if direction == "North":
            return f"North {a} steps"
        if direction == "Northeast":
            return f"North {a} steps, East {a} steps"
        if direction == "East":
            return f"East {a} steps"
        if direction == "Southeast":
            return f"South {a} steps, East {a} steps"
        if direction == "South":
            return f"South {a} steps"
        if direction == "Southwest":
            return f"South {a} steps, West {a} steps"
        if direction == "West":
            return f"West {a} steps"
        if direction == "Northwest":
            return f"North {a} steps, West {a} steps"

    def anticipate_reflect(self):
        history_info = self.retrieved['direction_info']

        if "You don't have any goal inference." in history_info and self.face_direction is None:
            return None
        if self.face_direction is not None:
            user_prompt = f"{history_info}Now you infer that the goal is in {self.dire_dis_transfer(self.face_direction, self.distance)}. According to all your inferences, what are the goal coordinates most likely to be? What is the corresponding goal direction from current position?"
        else:
            user_prompt = f"{history_info}According to all your inferences, what are the goal coordinates most likely to be? What is the corresponding goal direction from current position?"
        system_prompt = f"You are a helpful navigation agent in the city. And you should follow these " \
                        f"instructions:\n(1)You are evaluating goal coordinates. Take (0,0) as origin, (0," \
                        f"+1) as one step North, (+1,0) as one step East.\n(2)You may have multiple inferences. When one of your " \
                        f"inferences differs significantly from others, you should not take it into consideration.\n(" \
                        f"3)You should think step by step to answer two questions. Answer in the json format and keys " \
                        f"are ['Thought_Q1', 'Answer_Q1', 'Thought_Q2', 'Answer_Q2'].\n(4)Format Example of " \
                        f"'Answer_Q1': 'The goal coordinates are most likely to be (3,-2).' Format Example of " \
                        f"'Answer_Q2': 'The goal (3,-2) is in Southeast(more towards east) from current position (0," \
                        f"0).' "

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response, token_cost = chatgpt_request(messages, temperature=0.65)
        self.token_counts += token_cost
        if "```json" in response:
            response = response.split('```')[1][4:]

        if self.log_filepath is not None:
            write_gpt_data(messages, response, output_path=self.log_filepath, model_name='gpt')

        if self.to_print:
            print("\n(function) anticipate_reflection:")
            print(user_prompt, "\n", response)
        answer = json.loads(response)

        output = answer['Answer_Q1'] + answer['Answer_Q2']

        return output

    def route_planning(self):
        connection_info = self.retrieved['connection_info']
        trace_info = self.retrieved['trace_info']

        ant_ref = self.anticipate_reflect()
        if ant_ref is None:
            direction_info = f"You don't have any goal inference and should explore and gather more information. "
        else:
            direction_info = ant_ref

        instruct = f"According to all information above, choose one in {list(self.action_space.keys())} as your next action."

        system_prompt = "You are a helpful navigation agent in the city. And you should follow these instructions:\n(" \
                        "1)Locations (including the goal) in the city are represented by coordinates. Take (0," \
                        "0) as origin, (0,+1) as one step North, (+1,0) as one step East.\n" + f"(2)You should think step by step to determine your next action. Answer in the json format, and keys are ['action_reason', 'action']. Note that the 'action' must be one of {list(self.action_space.keys())}. Try not to move back unless necessary.\n(3)If your trajectory memory indicates you are wandering between two directions, you should move along a certain direction to break the loop.\n(4)Even if you have arrived at your inferred goal coordinates, you haven't found the goal yet, so you should search among unvisited areas nearby."

        if trace_info is None:
            user_prompt = f"{connection_info}{direction_info}{instruct}"
        else:
            user_prompt = f"{connection_info}{direction_info}{trace_info}{instruct}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response, token_cost = chatgpt_request(messages, temperature=0.4)
        self.token_counts += token_cost
        if "```json" in response:
            response = response.split('```')[1][4:]

        if self.log_filepath is not None:
            write_gpt_data(messages, response, output_path=self.log_filepath, model_name='gpt')

        if self.to_print:
            print("\n(function) Planning & Action")
            print(user_prompt, "\n", response)

        try:
            answer = json.loads(response)
            if 'yes' in answer['yes_or_no'].lower():
                self.plan = answer['new_plan']
            pass
        except:
            pass

        try:
            self.action = self.extract_direction(answer['action'], list(self.action_space.keys()))
        except:
            self.action = None

        pass

    def spatial_oriential_perception(self):
        """
        module: Spatial Oriential Perception
        function: visual2text
        """

        self.get_actionspace()
        self.get_observation()
        self.direction_reasoning()

    def memory_working(self):
        """
        module: Memory
        function: process information
        """

        self.retrieve()
        # if len(self.trace):
        #     self.anticipate_reflect()
        # else:
        #     self.reflect_direction = self.face_direction

    def agent_action(self):
        """
        module: planning and decision
        function: plan, decision and action
        """
        try:
            self.route_planning()
            print("\naction:", self.action)
            # self.decision_making()
            nxt_id = random.choice(self.action_space[self.action])
        except:
            self.route_planning()
            print("\naction:", self.action)
            if self.action in self.action_space.keys():
                nxt_id = random.choice(self.action_space[self.action])
            else:
                self.action = random.choice(list(self.action_space.keys()))
                nxt_id = random.choice(self.action_space[self.action])

        self.trace.append(self.action)
        self.path.append(nxt_id)
        self.location = self.map[nxt_id]
        if nxt_id == self.destination.id:
            self.state = "Success"
        else:
            # if self.mode == 1:
            self.state = "Searching"
        self.count = self.count + 1
        self.LTM.save(list(self.action_space.keys()), self.face_direction, self.distance, self.action)

    def step(self):
        self.to_print = 1
        print(f"\n\n step {self.count} \n\n")

        self.spatial_oriential_perception()
        self.memory_working()
        self.agent_action()

        print(self.path)
        print("-> ".join(self.trace))
        pass

    def run(self, limit_steps=25):
        """
        run the agent until the state is Success or reach the limit_steps
        """

        while self.state != "Success" and self.count < limit_steps:
            self.step()
        return self.state
