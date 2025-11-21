# map_generator.py
import numpy as np
import matplotlib.pyplot as plt

def create_map(width=50, height=30, obstacle_ratio=0.25):
    grid = np.zeros((height, width))

    # 랜덤 장애물 배치
    num_obstacles = int(width * height * obstacle_ratio)
    for _ in range(num_obstacles):
        y = np.random.randint(1, height - 1)
        x = np.random.randint(1, width - 1)
        grid[y, x] = 1

    # 경로가 완전히 막히지 않도록 시작/목표 근처는 비워둠
    grid[0:5, 0:5] = 0
    grid[height - 5:, width - 5:] = 0

    start = (2, 2)
    goal = (width - 3, height - 3)

    return grid, start, goal

def show_map(grid, start=None, goal=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(grid, cmap='gray_r', origin='lower')

    if start:
        plt.scatter(start[0], start[1], color='blue', s=100, label='Start')
    if goal:
        plt.scatter(goal[0], goal[1], color='red', s=100, label='Goal')

    plt.grid(True, color='lightgray', linewidth=0.5)
    plt.title("2D Path Planning Map")
    plt.legend()
    plt.show()

# 테스트용 실행
if __name__ == "__main__":
    grid, start, goal = create_map()
    show_map(grid, start, goal)
    print(create_map())
