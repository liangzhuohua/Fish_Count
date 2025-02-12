import sys
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QGridLayout
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont
from PyQt5.QtCore import Qt, QTimer
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.video_path = ''
        self.image_path = ''
        self.camera = cv2.VideoCapture(1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load('weight/improve.pt', map_location=self.device).fuse().eval()  # 修改模型文件路径
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4
        self.names = self.model.names
        self.colors = [[0, 255, 0] for _ in range(len(self.names))]
        self.init_ui()

    # def init_ui(self):
    #     # 创建主布局为水平布局
    #     main_layout = QHBoxLayout()
    #     # 左侧布局显示导入的视频、图片或者实时摄像头
    #     self.label_input = QLabel()
    #     self.label_input.setAlignment(Qt.AlignCenter)
    #     self.label_input.setFixedSize(400, 300)  # 可以根据需要调整大小
    #     main_layout.addWidget(self.label_input)
    #
    #     # 中间布局，放置功能选择按钮
    #     middle_layout = QVBoxLayout()
    #     middle_layout.addStretch(1)  # 添加伸缩项，使按钮居中
    #
    #     self.btn_image = QPushButton('图片检测')
    #     self.btn_image.clicked.connect(self.load_image)
    #     middle_layout.addWidget(self.btn_image)
    #
    #     self.btn_video = QPushButton('视频检测')
    #     self.btn_video.clicked.connect(self.load_video)
    #     middle_layout.addWidget(self.btn_video)
    #
    #     self.btn_camera = QPushButton('摄像头实时监测')
    #     self.btn_camera.clicked.connect(self.start_camera)
    #     middle_layout.addWidget(self.btn_camera)
    #
    #     middle_layout.addStretch(1)  # 添加伸缩项，使按钮居中
    #     main_layout.addLayout(middle_layout)
    #
    #     # 右侧布局显示检测结果图像或视频
    #     self.label_output = QLabel()
    #     self.label_output.setAlignment(Qt.AlignCenter)
    #     self.label_output.setFixedSize(400, 300)  # 可以根据需要调整大小
    #     main_layout.addWidget(self.label_output)
    #
    #     # 设置主窗口的布局
    #     self.setLayout(main_layout)
    #     self.setWindowTitle('YOLOv5鱼苗计数')

    def init_ui(self):
        # 使用网格布局
        grid_layout = QGridLayout()

        # 左侧标签显示输入图像/视频
        self.label_input = QLabel()
        self.label_input.setAlignment(Qt.AlignCenter)
        self.label_input.setFixedSize(800, 600)
        self.label_input.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.label_input, 0, 0, 1, 2)  # 放在第0行，第0列，跨越1行2列

        # 右侧标签显示输出图像/视频
        self.label_output = QLabel()
        self.label_output.setAlignment(Qt.AlignCenter)
        self.label_output.setFixedSize(800, 600)
        self.label_output.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.label_output, 0, 2, 1, 2)  # 放在第0行，第2列，跨越1行2列

        # 功能选择按钮
        self.btn_image = QPushButton('图片检测')
        self.btn_image.setIcon(QIcon('icon/打开.png'))  # 添加按钮图标
        self.btn_image.setFont(QFont('Arial', 10))
        self.btn_image.clicked.connect(self.load_image)
        grid_layout.addWidget(self.btn_image, 1, 0)

        self.btn_video = QPushButton('视频检测')
        self.btn_video.setIcon(QIcon('icon/视频.png'))  # 添加按钮图标
        self.btn_video.setFont(QFont('Arial', 10))
        self.btn_video.clicked.connect(self.load_video)
        grid_layout.addWidget(self.btn_video, 1, 1)

        self.btn_camera = QPushButton('摄像头实时监测')
        self.btn_camera.setIcon(QIcon('icon/摄像头开.png'))  # 添加按钮图标
        self.btn_camera.setFont(QFont('Arial', 10))
        self.btn_camera.clicked.connect(self.start_camera)
        grid_layout.addWidget(self.btn_camera, 1, 2)

        # self.setStyleSheet("""
        #             QWidget {
        #                 background-image: url('icon/图片1.png');
        #                 background-repeat: no-repeat;
        #                 background-position: center;
        #             }
        #             QPushButton {
        #                 margin: 10px;
        #                 padding: 5px 10px;
        #                 background-color: rgba(255, 255, 255, 150);
        #             }
        #         """)
        # 设置主窗口布局
        self.setLayout(grid_layout)
        self.setWindowTitle('YOLOv5鱼苗计数')
        self.setStyleSheet("QPushButton { margin: 10px; padding: 5px 10px; }")

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', '', 'Video files (*.mp4 *.avi)')
        self.detect_video()

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Image files (*.jpg *.png)')
        if self.image_path:
            print(f"Loaded image path: {self.image_path}")  # 确认路径正确
            frame = cv2.imread(self.image_path)
            if frame is not None:
                print("Image loaded successfully")  # 确认图像加载成功
                self.display_frame(frame, is_input=True)  # 显示原始图像进行测试
                self.detect_image()
            else:
                print("Failed to load image")  # 如果加载失败，打印错误消息

    def start_camera(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_camera)
        self.timer.start(30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # 设置摄像头窗口宽度
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # 设置摄像头窗口高度

    def detect_video(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.display_frame(frame, is_input=True)
            result_frame = self.detect(frame)
            self.display_frame(result_frame, is_input=False)
            if cv2.waitKey(1) == ord('1'):
                break
        cap.release()

    def detect_image(self):
        frame = cv2.imread(self.image_path)
        result_frame = self.detect(frame)
        self.display_frame(result_frame, is_input=False)

    def detect_camera(self):
        ret, frame = self.camera.read()
        if ret:
            self.display_frame(frame, is_input=True)
            result_frame = self.detect(frame)
            self.display_frame(result_frame, is_input=False)

    def detect(self, frame):
        img = letterbox(frame, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)[0]
        num_detections = 0
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for det in pred:
                x1, y1, x2, y2, conf, cls = det
                label = f'{self.names[int(cls)]} {conf:.2f}'
                color = self.colors[int(cls)]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                num_detections +=1
        cv2.putText(frame, f'number: {num_detections}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    def display_frame(self, frame, is_input=True):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()  # 使用.copy()确保数据的持久性
            pixmap = QPixmap.fromImage(q_img)
            if is_input:
                self.label_input.setPixmap(
                    pixmap.scaled(self.label_input.width(), self.label_input.height(), Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation))
            else:
                self.label_output.setPixmap(
                    pixmap.scaled(self.label_output.width(), self.label_output.height(), Qt.KeepAspectRatio,
                                  Qt.SmoothTransformation))
        except Exception as e:
            print(f"在更新图像显示时发生错误: {e}")
            
    def reset_buttons(self):
        self.btn_image.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_camera.setEnabled(True)

    def exit_app(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
        if self.camera.isOpened():
            self.camera.release()
        QApplication.quit()

    def process_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 对帧进行处理，例如调用 detect 方法
            # self.detect(frame)  # 假设你有一个处理帧并返回结果的方法
            # 显示原始帧或处理后的帧
            self.display_frame(frame, is_input=False)  # 假设你想在右侧显示处理后的结果
        else:
            self.timer.stop()  # 如果没有帧可读，则停止定时器


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
