import modules.LLM as llm
import json
import os
from modules.LLM import write_gpt_data


def cal_coord(last_coord, direction):
    x, y = last_coord
    if direction == "North":
        return x, y + 1
    if direction == "Northeast":
        return x + 1, y + 1
    if direction == "East":
        return x + 1, y
    if direction == "Southeast":
        return x + 1, y - 1
    if direction == "South":
        return x, y - 1
    if direction == "Southwest":
        return x - 1, y - 1
    if direction == "West":
        return x - 1, y
    if direction == "Northwest":
        return x - 1, y + 1


def cal_goal_coord(now_coord, direction, distance):
    x, y = now_coord
    a = round(distance / 50)
    if direction == "North":
        return x, y + a
    if direction == "Northeast":
        return x + a, y + a
    if direction == "East":
        return x + a, y
    if direction == "Southeast":
        return x + a, y - a
    if direction == "South":
        return x, y - a
    if direction == "Southwest":
        return x - a, y - a
    if direction == "West":
        return x - a, y
    if direction == "Northwest":
        return x - a, y + a


class TrajectoryNode:
    def __init__(self, coord, connect, act, n_coord):
        self.coordinates = coord
        self.connection = connect
        self.action = act
        self.next_coord = n_coord


class NetworkManager:
    def __init__(self, logpath=None):
        self.trajectory_list = []
        self.goal_coord_list = []
        self.is_model_loaded = False
        self.log_filepath = logpath

    def save(self, connect, goal_dir, goal_dis, act):
        """
        该函数请在做出决策后调用
        Input:
                connect: 当前连接，形如["North", "East"]
                goal_dir: 方位推理模块输出的方向（不是预测或反思得出的方向！）形如"North"，仅在当前观测到地标时输入，其余时候请输入None
                goal_dis: 方位推理模块输出的距离，浮点数，仅在当前观测到地标时输入，其余时候请输入None
                act: 当前做出的决策，形如"North"
        """
        if len(self.trajectory_list) == 0:
            start = TrajectoryNode((0, 0), connect, act, cal_coord((0, 0), act))
            self.trajectory_list.append(start)
        else:
            coord = self.trajectory_list[-1].next_coord
            now = TrajectoryNode(coord, connect, act, cal_coord(coord, act))
            self.trajectory_list.append(now)
            if len(self.trajectory_list) > 10:
                self.trajectory_list.pop(0)

        if goal_dir is not None:
            goal_coord = cal_goal_coord(self.trajectory_list[-1].coordinates, goal_dir, goal_dis)
            self.goal_coord_list.append(goal_coord)
        if len(self.goal_coord_list) > 3:
            self.goal_coord_list.pop(0)

    def retrieve(self, connect):
        """
        该函数请在预测-反思之前调用
        Input: connect: 当前连接，形如["North", "East"]

        Output: 一个字典，'connection_info': 当前节点连接情况，'direction_info': 目标方向记忆信息，'trace_info': 过去轨迹总结
        """
        if len(self.trajectory_list) == 0:
            now_coord = (0, 0)
        else:
            now_coord = self.trajectory_list[-1].next_coord
        connection_info = f"You are now at {now_coord}. Your current connection includes {connect}. "
        for dire in connect:
            connect_coord = cal_coord(now_coord, dire)
            state = "Unvisited"
            for Node in self.trajectory_list:
                if connect_coord == Node.coordinates:
                    state = "Visited"
                    break
            connection_info += f"{dire} is at {connect_coord}, {state}. "

        direction_info = f"You are now at {now_coord}. "
        if len(self.goal_coord_list) == 0:
            direction_info += f"You don't have any goal inference."
        if len(self.goal_coord_list) == 1:
            direction_info += f"Last time you inferred the goal was at {self.goal_coord_list[0]}."
        if len(self.goal_coord_list) == 2:
            direction_info += f"The first time you inferred the goal was at {self.goal_coord_list[0]}. The second time you inferred the goal was at {self.goal_coord_list[1]}."
        if len(self.goal_coord_list) == 3:
            direction_info += f"The first time you inferred the goal was at {self.goal_coord_list[0]}. The second time you inferred the goal was at {self.goal_coord_list[1]}. The third time you inferred the goal was at {self.goal_coord_list[2]}."

        if len(self.trajectory_list) >= 5:
            trace_info = self.llm_interface(self.trajectory_list)
        else:
            trace_info = None

        return {'connection_info': connection_info, 'direction_info': direction_info, 'trace_info': trace_info}

    def timestamp_generation(self, index):
        coord = self.trajectory_list[index].coordinates
        connect = self.trajectory_list[index].connection
        act = self.trajectory_list[index].action
        n_coord = self.trajectory_list[index].next_coord

        description = f"You were at {coord}. You could move to {connect} from here. You chose to move to {act}. You then arrived at {n_coord}.\n"

        return description

    def llm_interface(self, memory_list):

        description = "You are conducting a navigation task and here is your memory list in time sequence.\n"
        for i in range(len(memory_list)):
            description += f"{i + 1}. "
            description += self.timestamp_generation(i)

        description += "Summarize all your memory, what can you learn from it? Answer in no more than 2 sentence."

        print("\n(function) memory_summary:")

        print(description)

        system_prompt = "Format Example of Answer: 'In the past 15 steps, you initially wandered between East and West. You then headed East and then North to reach your current position. Now you are on a north-south road. If you keep going South, you will reach an intersection (1, -5) which can move to North, West and East. If you keep going North, you will reach a dead end (1, 5) with only one navigable direction.'"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": description}
        ]

        response, token_cost = llm.chatgpt_request(messages, temperature=0.4)
        if "```json" in response:
            response = response.split('```')[1][4:]

        if self.log_filepath is not None and os.path.exists(self.log_filepath):
            write_gpt_data(messages, response, output_path=self.log_filepath, model_name='gpt')
        print(response)
        # output = json.loads(response)
        output = response
        return output


if __name__ == "__main__":
    manager = NetworkManager()
