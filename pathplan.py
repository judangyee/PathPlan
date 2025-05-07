import numpy as np
import rospy
from collections import deque
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from data_pre import data_pre
from scipy.spatial import Delaunay
from scipy.interpolate import make_interp_spline
from custom_msgs_pkg.msg import corner
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

data_processor = data_pre()
lidar_marker_array = None  # 실제 데이터로 대체

gl_points, gl_colors = data_processor.lidar_marker_cb(lidar_marker_array)

# publisher
marker_pub = rospy.Publisher("/cone_markers", Marker, queue_size=1)
path_marker_pub = rospy.Publisher("/spline_path_marker", Marker, queue_size=1)


# 라바콘 좌표를 큐형식으로 저장
node = deque(maxlen=5)
# 색상별로 좌표 분류
yellow = deque(maxlen=5)
blue = deque(maxlen=5)

def getData(raw_data):
    global gl_points, gl_colors
    process_data = data_pre()
    gl_points, gl_colors = process_data.lidar_marker_cb(raw_data)
    print(gl_points,gl_colors)

def printcorner(corner):
    return corner.isCorner

def corner_detect():
    global corner
    if corner.isConer == True:
        if corner.wallColor == 0:  # 파란색 벽 임으로 오른쪽 코너
            return 1
        elif corner.wallColor == 1: # 노란색 벽 임으로 왼쪽 코너
            return -1
    elif corner.isCorner == False:
        return 0

#색깔 분류 함수
def classification_color(gl_colors, gl_points):
    for color, coord in zip(gl_colors, gl_points):
        if color == "yellow":
            yellow.append(coord)
        elif color == "blue":
            blue.append(coord)
    return yellow, blue


#사각분할 함수
def song_square(yellow, blue):
    if len(yellow) < 2 or len(blue) < 2:
        return print("Error: Not enough Data")  # 충분한 데이터가 없으면 None 반환

    recent_yellow = list(yellow)[-2]
    recent_blue = list(blue)[-2]
    whole = recent_yellow + recent_blue
    cx = sum(p[0] for p in whole) / 4
    cy = sum(p[1] for p in whole) / 4
    return whole, (cx, cy)

# 코너 감지 함수


#들로네 삼각분할 함수
def Deuloney_Triangulation(node):
    if len(node) < 3:
        print("삼각분할을 위해서는 최소 3개의 점이 필요 합니다.")
        return
    else:
        points = np.array(node)  # deque를 numpy 배열로 변환
        # 들로네 삼각분할 수행
        tri = Delaunay(points)

        return tri

def slope(yellow, blue): #기울기 함수
    if corner_pos == -1:
        slope = (yellow[-1][1] - yellow[-2][1]) / (yellow[-1][0] - yellow[-2][0])  # 왼쪽 코너일 경우
        return slope
    elif corner_pos == 1:
        slope = (blue[-1][1] - blue[-2][1]) / (blue[-1][0] - blue[-2][0])  # 오른쪽 코너일 경우
        return slope


# B-spline 보간 함수
def b_spline_path(optimal_path, num_points_between=20):

    x = [pt[0] for pt in optimal_path]
    y = [pt[1] for pt in optimal_path]

    t = np.linspace(0, 1, len(x))  # 정규화된 파라미터 t
    t_new = np.linspace(0, 1, num_points_between * (len(x) - 1))

    # B-spline 보간기 생성 (기본적으로 3차 스플라인)
    spline_x = make_interp_spline(t, x, k=3)
    spline_y = make_interp_spline(t, y, k=3)

    x_smooth = spline_x(t_new)
    y_smooth = spline_y(t_new)

    spline_path = list(zip(x_smooth, y_smooth))
    return spline_path

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
# Apex 찾는 함수(Apex는 코너에서 가속 지점)
def find_Apex(slope):
    if slope == 0:
        return True
    else:
        return False

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

# Spline 경로 Marker 생성 함수
def make_path_marker(path_points, frame_id="base_link"):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "spline_path"
    marker.id = 100
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0

    for pt in path_points:
        point = Point()
        point.x = pt[0]
        point.y = pt[1]
        point.z = 0.0
        marker.points.append(point)

    return marker


def main():
    global corner_pos  # 전역 변수로 선언

    rospy.init_node("path_planner")
    path_pub = rospy.Publisher("/spline_path", Path, queue_size=1)

    rate = rospy.Rate(10)  # 10Hz

    while not rospy.is_shutdown():
        # 1. 데이터 입력
        lidar_marker_array = ...  # 실제 라이다 마커 입력받는 부분
        gl_points, gl_colors = data_processor.lidar_marker_cb(lidar_marker_array)

        # 2. 색상 분류
        yellow, blue = classification_color(gl_colors, gl_points)

        if len(yellow) < 2 or len(blue) < 2:
            rospy.logwarn("라바콘 수 부족. 대기 중...")
            rate.sleep()
            continue

        # 3. 곡선 판단
        corner_pos = corner_detect()  # 변경된 함수 반영

        # 4. 분할 방식 선택
        if corner_pos == 0:  # 직선
            rospy.loginfo("직선 구간입니다. 사각형 기반 중심점 사용.")
            square_result = song_square(yellow, blue)
            if square_result is None:
                rospy.logwarn("사각형 생성 실패")
                rate.sleep()
                continue
            _, center = square_result
            node.append(center)

        else:  # 곡선
            rospy.loginfo("곡선 구간입니다. 들로네 삼각분할 수행.")
            if len(node) < 3:
                rospy.loginfo("중앙점 누적 중... (곡선 삼각분할 필요)")
                square_result = song_square(yellow, blue)
                if square_result is not None:
                    _, center = square_result
                    node.append(center)
                rate.sleep()
                continue

            tri = Deuloney_Triangulation(node)
            if tri is None:
                rospy.logwarn("삼각분할 실패")
                rate.sleep()
                continue

            # 삼각형 중심점들로 경로 갱신
            triangle_centers = []
            for simplex in tri.simplices:
                pts = [node[i] for i in simplex]
                cx = sum(p[0] for p in pts) / 3
                cy = sum(p[1] for p in pts) / 3
                triangle_centers.append((cx, cy))

            node.clear()
            node.extend(triangle_centers)

        if len(node) < 3:
            rospy.loginfo("경로점 누적 중... (%d개)", len(node))
            rate.sleep()
            continue

        optimal_path = list(node)

        # 5. Spline 처리
        spline_path = b_spline_path(optimal_path)

        # 6. Apex 탐색 (곡선일 때만)
        if corner_pos != 0:
            current_slope = slope(yellow, blue)
            is_apex = find_Apex(current_slope)

            if is_apex:
                rospy.loginfo("Apex 도달! 재가속 준비")
            else:
                rospy.loginfo("Apex 아님. 경로 추종 유지")

        # 7. ROS Path 메시지로 변환 후 publish
        ros_path_msg = create_path(spline_path)
        path_pub.publish(ros_path_msg)

        # 라바콘 시각화
        yellow_marker = make_cone_marker(yellow, "yellow", marker_id=1)
        blue_marker = make_cone_marker(blue, "blue", marker_id=2)
        marker_pub.publish(yellow_marker)
        marker_pub.publish(blue_marker)

        # 경로 시각화
        spline_marker = make_path_marker(spline_path)
        path_marker_pub.publish(spline_marker)

        rate.sleep()


# ROS Subscriber 설정
lidar_marker_sub = rospy.Subscriber("/visualization_marker_array", MarkerArray, lidar_marker_cb)
corner_sub = rospy.Subscriber("/corner", corner, printcorner)
marker_pub = rospy.Publisher("/cone_markers", Marker, queue_size=1)
path_marker_pub = rospy.Publisher("/spline_path_marker", Marker, queue_size=1)