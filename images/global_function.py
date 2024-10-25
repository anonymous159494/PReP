import math

def bfs(loc_dict, start_id, target_id):   
    """
    BFS, get the shortest path;
    start_id；target_id; loc_dict, dataset.
    """
    queue = [start_id]
    trace = [start_id]
    parents = dict()
    path = [target_id]
    while queue:
        q = queue.pop(0)
        for connect, _ in loc_dict[q].connect:
            if connect in trace:
                continue
            parents[connect] = q
            if connect == target_id:
                key = target_id
                while key != start_id:
                    father = parents[key]
                    path.append(father)
                    key = father
                path.reverse()
                return path
            queue.append(connect)
            trace.append(connect)
    return path


Direction = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']

def angle2dir(angle):
    """
    8 fundamental direction
    """
    s = 22.5
    angle = angle % 360
    if 0 <= angle < 12.25: return Direction[0]
    if 12.25 <= angle <= 77.25: return Direction[1]
    if 77.25 < angle < 102.25: return Direction[2]
    if 102.25 <= angle <= 167.75: return Direction[3]
    if 167.75 < angle < 192.25: return Direction[4]
    if 192.25 <= angle <= 257.75: return Direction[5]
    if 257.75 < angle < 282.25: return Direction[6]
    if 282.25 <= angle <= 347.75: return Direction[7]
    else: return Direction[0]
    for i in range(8):
        if angle < s + 45 * i:
            return Direction[i]
    return Direction[0]


def dir2angle(dir:str):

    for i in range(8):
        if dir == Direction[i]:
            return 45 * i
    return 0

def ang2vec(ang:float, dis:float) -> list[float, float]:
    ang = ((90 - ang) + 360) % 360
    ang = ang / 180 * math.pi
    vec = [dis * math.cos(ang), dis * math.sin(ang)]
    return vec

def vec2ang(vec:list[float, float]) -> float:
    ang = math.atan(vec[1]/vec[0]) * 180 / math.pi
    if vec[0] < 0:
        ang = (ang + 180) % 360
    ang = ((90 - ang) + 360) % 360
    return ang

def dir2vec(dir:str, dis:float) -> list[float, float]:
    ang = dir2angle(dir)
    vec = ang2vec(ang, dis)
    return vec

def vec2dir(vec:list[float, float]) -> str:
    ang = vec2ang(vec)
    return angle2dir(ang)

def oppsiteDirection(dir:str) -> str:
    assert dir in Direction
    return angle2dir((dir2angle(dir) + 180)% 360)

def cal_angle(start_point, end_point):

    return (round(90 - math.degrees(math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)), 2)) % 360


def cal_angle2(start_point, end_point):

    return (round(90 - math.degrees(math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])), 2)) % 360 


def cal_distance(point1, point2):

    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def direction_match_score(dir1, dir2):
    if dir1 not in Direction or dir2 not in Direction:
        return 5
    ang1 = dir2angle(dir1)
    ang2 = dir2angle(dir2)
    if ang1 is None or ang2 is None:
        return 5
    min_angle, max_angle = sorted([ang1, ang2])
    diff = min(max_angle - min_angle, min_angle + 360 - max_angle)
    return diff / 45


def get_realangle(box, heading):
    centx = (box[0] + box[2]) / 2
    angle = math.atan((centx - 0.5) / 0.5) * 180 / math.pi
    return heading + angle

def angle_reasoning(dir1:str|float, dis1:float, dir2:str|float, dis2:float) -> tuple[float, str]:
    if type(dir1) is str:
        vec1 = dir2vec(dir1, dis1)
    else:
        vec1 = ang2vec(dir1, dis1)
    if type(dir2) is str:
        vec2 = dir2vec(dir2, dis2)
    else:
        vec2 = ang2vec(dir2, dis2)
    vec3 = [i - j for i, j in zip(vec1,vec2)]
    ang3 = vec2ang(vec3)
    # print(vec1, vec2, vec3)
    dis = math.sqrt(vec3[0] ** 2 + vec3[1] ** 2)
    return ang3, angle2dir(ang3), dis


def angle_intersect(ref_scope, scope):
    # [0, 720]
    min_ref, max_ref = ref_scope
    min_val, max_val = scope
    assert min_ref <= max_ref
    assert min_val <= max_val
    min_new, max_new = 0, 0
    if min_val <= max_ref or max_val >= min_ref:
        pass
    else:
        if max_val < min_ref:
            min_val = min_val + 360
            max_val = max_val + 360
        else:
            min_val = min_val - 360
            max_val = max_val - 360
    if min_val > min_ref: min_new = min_val
    else: min_new = min_ref
    if max_val < max_ref: max_new = max_val
    else: max_new = max_ref
    if min_new < 0: min_new = 0
    if max_new > 720: max_new = 720
    if min_new <= max_new:
        return (min_new, max_new)
    else:
        return (min_ref, max_ref)



def angle_average(angle:list[float], coor:list[float]|None=None):
    """
    calculate an average angle 
    """
    if coor is None:
        coor = [1] * len(angle)
    
    ave_vec = [0, 0]
    for ang, co in zip(angle, coor):
        vec = ang2vec(ang, co)
        ave_vec[0] = ave_vec[0] + vec[0]
        ave_vec[1] = ave_vec[1] + vec[1]

    try:
        average_angle = vec2ang(ave_vec)
    except:
        average_angle = None
    return average_angle

        

def convert_json_key(param_dict):
    """
    json.dump() writes all int keys as str when encoding and storing them
    So after reading the json file, use this method to restore all int keys decoded to str to int
    """
    new_dict = dict()
    for key, value in param_dict.items():
        if isinstance(value, (dict,)):
            res_dict = convert_json_key(value)
            try:
                new_key = int(key)
                new_dict[new_key] = res_dict
            except:
                new_dict[key] = res_dict
        else:
            try:
                new_key = int(key)
                new_dict[new_key] = value
            except:
                new_dict[key] = value

    return new_dict
