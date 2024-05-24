"""
Location class and dataset process
"""

import os
import pickle
from PIL import Image

class Location:
    def __init__(self) -> None:
        # Location Node is
        self.id = id

        # Location Node 
        self.lane_id = None         # lane id
        self.lane_name = None       # lane name (optional)
        self.lane_s = None          # lane distance

        # GPS
        self.xy = None              
        self.lnglat = None

        # baidu coordinate
        self.bdxy = None
        self.bdcoor = None      # bd lnglat

        # Location Node connect
        self.connect = []
        self.landmark = {}

        # streetviews
        self.sid = None            #  sid
        self.panorama = None       
        self.svcache = None

    def get_streetview(self, cache=None, path_only=False):
        path_list = []
        persp_list = []
        for connect, angle in self.connect:
            if cache is not None:
                img_path = os.path.join(cache, "{}_{}.jpg".format(self.sid, connect))
            else:
                img_path = os.path.join(self.svcache, "{}_{}.jpg".format(self.sid, connect))
            # if not os.path.exists(img_path):
            #     raise FileNotFoundError(img_path)
            path_list.append(img_path)
            if not path_only: persp_list.append(Image.open(img_path))
        return path_list, persp_list

    def get_connect_image(self, connect, cache=None, path_only=False):
        if cache is not None:
            img_path = os.path.join(cache, "{}_{}.jpg".format(self.sid, connect))
        else:
            img_path = os.path.join(self.svcache, "{}_{}.jpg".format(self.sid, connect))
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        if not path_only: 
            persp = Image.open(img_path)
        else: 
            persp = None
        return img_path, persp

    def __str__(self) -> str:
        return """<Location instance>\nid = {}\nxy= {}\nlnglat= {}\nconnect= {}\nlandmark= {}""".format(self.id, self.xy, self.lnglat, self.connect, self.landmark)


def load_dataset(data, sv=None) -> list[Location]:
    """
    func : load dataset
    input: data: dataset cache path
           sv  : streetview cache path
    """
    with open(data, "rb") as f:
        loc_dict = pickle.load(f)
    if sv is not None:
        for i, loc in loc_dict:
            loc.panorama = loc.get_streetview(sv)
    return loc_dict


def bfs_gen(loc_dict, root, depth) -> list[int]:
    """
    func: bfs to get the node with the depth from root
    """
    queue = []
    trace = [root]
    queue.append(root)
    while (queue) and depth > 0:
        depth = depth - 1
        n = len(queue)
        for i in range(n):
            q = queue.pop(0)
            loc = loc_dict[q]
            for connect, _ in loc.connect:
                if connect in trace:
                    continue
                else:
                    queue.append(connect)
                    trace.append(connect)
    return queue