#!/usr/bin/env python3
# 두번째 pathplan코드

import numpy as np
import rospy
import heapq
from visualization_msgs.msg import Marker
import math
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from data_pre import data_pre
from scipy.interpolate import make_interp_spline
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

gl_points = []
gl_colors = []
corner_data = None

# gl_points, gl_colors = data_processor.lidar_marker_cb(lidar_marker_array)

# publisher
marker_pub = rospy.Publisher("/cone_markers", Marker, queue_size=1)
path_marker_pub = rospy.Publisher("/spline_path_marker", Marker, queue_size=1)


# 라바콘 좌표를 큐형식으로 저장
node = []
# 색상별로 좌표 분류
yellow = []
blue = []

#단기 목표 설정
destination = (max(yellow) + max(blue))/2 # 가장 멀리 있는 노드들의 중점을 단기 목표로 설정

# 데이터 임포트 함수
def getData(raw_data):
    global gl_points, gl_colors
    process_data = data_pre()
    gl_points, gl_colors = process_data.lidar_marker_cb(raw_data)
    print(gl_points,gl_colors)

# 색 분류
def classification_color(gl_colors, gl_points):
    for color, coord in zip(gl_colors, gl_points):
        if color == "yellow":
            yellow.append(coord)
        elif color == "blue":
            blue.append(coord)
    return yellow, blue

# 사각 분할
def song_square(yellow, blue):
    # 최소 2개 이상의 좌표가 있어야 윗변/아랫변을 구할 수 있음
    if len(yellow) < 2 or len(blue) < 2:
        print("Error: Not enough Data")
        return

    # 어제-오늘에 해당하는 두 쌍의 좌표
    y_old, y_new = yellow[-2], yellow[-1]
    b_old, b_new = blue[-2],   blue[-1]

    # 윗변 중심 (어제 시점의 좌우 콘 중간)
    top_center = ((y_old[0] + b_old[0]) / 2,(y_old[1] + b_old[1]) / 2)

    # 아랫변 중심 (오늘 시점의 좌우 콘 중간)
    bottom_center = ((y_new[0] + b_new[0]) / 2,
                     (y_new[1] + b_new[1]) / 2)

    # 사각형 전체 중심 (네 점의 평균)
    cx = (y_old[0] + y_new[0] + b_old[0] + b_new[0]) / 4
    cy = (y_old[1] + y_new[1] + b_old[1] + b_new[1]) / 4
    center = (cx, cy)

    # (윗변 중심, 아랫변 중심, 사각형 중심) 순서로 반환
    return top_center, bottom_center, center

# 후보 노드 생성 함수
def node_maker_np(a, b, n=5):
    x1, y1 = a
    x2, y2 = b
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)
    return list(zip(xs, ys))

# 가중치를 위한 유클리드 거리 계산 함수   가중치는 직선거리 뿐만 아니라 대각선거리도 계산 가능
def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# 연결 리스트 기반 다익스트라 알고리즘
def dijkstra(nodes, edges, start_coord, goal_coord): # 노드, 엣지, 시작 좌표, 목표 좌표
    # 좌표 ↔ 정점 ID 매핑
    coord_to_id = {coord: i for i, coord in enumerate(nodes)}
    id_to_coord = {i: coord for coord, i in coord_to_id.items()}
    n = len(nodes)

    # 인접 리스트 생성
    graph = [[] for _ in range(n)]
    for a, b in edges:
        u = coord_to_id[a]
        v = coord_to_id[b]
        w = euclidean(a, b)
        graph[u].append((v, w))
        graph[v].append((u, w))  # 양방향 연결 (필요 없으면 제거 가능)

    # 다익스트라 초기화
    dist = [float('inf')] * n
    prev = [None] * n
    start = coord_to_id[start_coord]
    goal = coord_to_id[goal_coord]
    dist[start] = 0
    pq = [(0, start)]  # (누적거리, 정점 ID)

    # 탐색 루프
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    # 경로 복원
    path = []
    cur = goal
    if prev[cur] is not None or cur == start:
        while cur is not None:
            path.append(id_to_coord[cur])
            cur = prev[cur]
        path.reverse()

    return dist[goal], path

# 경로 생성 함수
def create_path(spline_path, frame_id="base_link"):
    path_msg = Path()
    path_msg.header.frame_id = frame_id
    path_msg.header.stamp = rospy.Time.now()

    for x, y in spline_path:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0  # 기본 방향
        path_msg.poses.append(pose)

    return path_msg

# 라바콘 Marker 생성 함수
def make_cone_marker(cones, color, marker_id, frame_id="base_link"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "cones"
    marker.id = marker_id
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.3
    marker.scale.y = 0.3
    marker.scale.z = 0.3
    marker.pose.orientation.w = 1.0

    # 색상 설정
    if color == "yellow":
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    elif color == "blue":
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
    marker.color.a = 1.0  # 불투명

    for p in cones:
        pt = Point()
        pt.x = p[0]
        pt.y = p[1]
        pt.z = 0.0
        marker.points.append(pt)

    return marker

def main():
    global gl_colors, gl_points, yellow, blue

    rospy.init_node("path_planner")
    path_pub = rospy.Publisher("/spline_path", Path, queue_size=1)

    rate = rospy.Rate(1.0 / 5.0)  # 5초마다 실행

    while not rospy.is_shutdown():
        yellow.clear()
        blue.clear()

        # 최신 라바콘 데이터 수신
        raw_data = rospy.wait_for_message("/lidar_marker_array", MarkerArray)
        getData(raw_data)
        classification_color(gl_colors, gl_points)

        min_len = min(len(yellow), len(blue))
        if min_len < 2:
            rospy.logwarn("라바콘 데이터가 부족합니다.")
            rate.sleep()
            continue

        nested_nodes = []  # 후보 노드들 (레벨 단위)

        # 각 구간에서 노드 생성
        for i in range(min_len - 1):
            y_old, y_new = yellow[i], yellow[i + 1]
            b_old, b_new = blue[i],   blue[i + 1]

            result = song_square([y_old, y_new], [b_old, b_new])
            if result is None:
                continue

            top, bottom, center = result
            level1 = node_maker_np(top, center, n=5)
            level2 = node_maker_np(center, bottom, n=5)
            nested_nodes.append(level1)
            nested_nodes.append(level2)

        if len(nested_nodes) < 2:
            rospy.logwarn("후보 노드 레벨이 부족합니다.")
            rate.sleep()
            continue

        # 노드, 엣지 구성
        flat_nodes = [pt for row in nested_nodes for pt in row]
        edges = []

        for row in nested_nodes:
            for i in range(len(row) - 1):
                edges.append((row[i], row[i + 1]))  # 가로 연결

        for c in range(len(nested_nodes[0])):
            for r in range(len(nested_nodes) - 1):
                edges.append((nested_nodes[r][c], nested_nodes[r + 1][c]))  # 세로 연결

        # 출발점: 맨 앞 노드
        start_coord = nested_nodes[0][0]

        # 도착점: 가장 먼 yellow/blue의 중점 → 그에 가장 가까운 후보 노드
        y, b = yellow[-1], blue[-1]
        mid_goal = ((y[0] + b[0]) / 2, (y[1] + b[1]) / 2)
        goal_coord = min(flat_nodes, key=lambda pt: euclidean(pt, mid_goal))

        # 다익스트라 수행
        _, path = dijkstra(flat_nodes, edges, start_coord, goal_coord)

        # Path 메시지 생성 및 퍼블리시
        path_msg = create_path(path)
        path_pub.publish(path_msg)

        rospy.loginfo(f"{len(nested_nodes)}레벨 기반 경로를 퍼블리시했습니다.")
        rate.sleep()


# ROS Subscriber 설정
lidar_marker_sub = rospy.Subscriber("/visualization_marker_array", MarkerArray, getData)
marker_pub = rospy.Publisher("/cone_markers", Marker, queue_size=1)
path_marker_pub = rospy.Publisher("/spline_path_marker", Marker, queue_size=1)

if __name__ == "__main__":
    main()

