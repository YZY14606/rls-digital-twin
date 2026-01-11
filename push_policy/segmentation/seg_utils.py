import cv2
import numpy as np

class PointSelector:
    def __init__(self, image_path = None, rgb = None):
        if image_path is None and rgb is not None:
            self.image = rgb
        elif image_path is not None and rgb is None:
            self.image = cv2.imread(image_path)
        else:
            print('Image loading error!')

        if self.image is None:
            raise FileNotFoundError(f"Cannot load image from path: {image_path}")
        
        self.img_display = self.image.copy()
        self.points = []
        self.labels = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键：正样本
            self.points.append((x, y))
            self.labels.append(1)
            print(f"Positive point added: ({x}, {y}) -> label=1")
            cv2.circle(self.img_display, (x, y), 5, (0, 255, 0), -1)  # 绿色实心圆
            
        elif event == cv2.EVENT_RBUTTONDOWN:  # 右键：负样本
            self.points.append((x, y))
            self.labels.append(0)
            print(f"Negative point added: ({x}, {y}) -> label=0")
            cv2.circle(self.img_display, (x, y), 5, (0, 0, 255), -1)  # 红色实心圆
        
        cv2.imshow("Select Points", self.img_display)

    def select_points(self):
        window_name = "Select Points"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("Instructions:")
        print("- Left click: add positive point (label=1)")
        print("- Right click: add negative point (label=0)")
        print("- Press 'q' to finish.")

        cv2.imshow(window_name, self.img_display)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
        cv2.destroyAllWindows()

        return self.points, self.labels

if __name__ == "__main__":
    image_path = "/home/yzy/图片/object/2025-04-29 10-50-58 的屏幕截图.png"  # ← 替换为你的图像路径
    try:
        selector = PointSelector(image_path)
        pos_neg_points, point_labels = selector.select_points()
        print(pos_neg_points)
        print(point_labels)
        
        # 分离正负样本（可选）
        pos_points = [p for p, l in zip(pos_neg_points, point_labels) if l == 1]
        neg_points = [p for p, l in zip(pos_neg_points, point_labels) if l == 0]

        print("\nFinal results:")
        print("Positive points:", pos_points)
        print("Negative points:", neg_points)
    except Exception as e:
        print("Error:", e)