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
        self.model = attempt_load('weight/original.pt', map_location=self.device).fuse().eval()  # Modify the file path of the model.
        self.conf_threshold = 0.5
        self.iou_threshold = 0.4
        self.names = self.model.names
        self.colors = [[0, 255, 0] for _ in range(len(self.names))]
        self.init_ui()

    # def init_ui(self):
    #     # Create the main layout as a horizontal layout.
    #     main_layout = QHBoxLayout()
    #     # The left layout displays imported videos, pictures, or live cameras
    #     self.label_input = QLabel()
    #     self.label_input.setAlignment(Qt.AlignCenter)
    #     self.label_input.setFixedSize(400, 300)  # The size can be adjusted as needed.
    #     main_layout.addWidget(self.label_input)
    #
    #     # Middle layout, place the function selection buttons
    #     middle_layout = QVBoxLayout()
    #     middle_layout.addStretch(1)  # Add an expansion item to center the button.
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
    #     middle_layout.addStretch(1)  # Add an expansion item to center the button.
    #     main_layout.addLayout(middle_layout)
    #
    #     # The right layout displays the detection result images or videos.
    #     self.label_output = QLabel()
    #     self.label_output.setAlignment(Qt.AlignCenter)
    #     self.label_output.setFixedSize(400, 300)  # The size can be adjusted as needed.
    #     main_layout.addWidget(self.label_output)
    #
    #     # Set the layout of the main window
    #     self.setLayout(main_layout)
    #     self.setWindowTitle('YOLOv5鱼苗计数')

    def init_ui(self):
        # Use grid layout
        grid_layout = QGridLayout()

        # The label on the left side indicates the input image/video.
        self.label_input = QLabel()
        self.label_input.setAlignment(Qt.AlignCenter)
        self.label_input.setFixedSize(800, 600)
        self.label_input.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.label_input, 0, 0, 1, 2)  # Place it at the 0th row, 0th column, spanning 1 row and 2 columns.
        
        # The label on the right side indicates the output image/video.
        self.label_output = QLabel()
        self.label_output.setAlignment(Qt.AlignCenter)
        self.label_output.setFixedSize(800, 600)
        self.label_output.setStyleSheet("border: 1px solid black;")
        grid_layout.addWidget(self.label_output, 0, 2, 1, 2)  # Be placed on the 0th row, the 2nd column, spanning 2 rows and 1 column.

        # Function selection button
        self.btn_image = QPushButton('图片检测')
        self.btn_image.setIcon(QIcon('icon/打开.png'))  # Add button icon
        self.btn_image.setFont(QFont('Arial', 10))
        self.btn_image.clicked.connect(self.load_image)
        grid_layout.addWidget(self.btn_image, 1, 0)

        self.btn_video = QPushButton('视频检测')
        self.btn_video.setIcon(QIcon('icon/视频.png'))  # Add button icon
        self.btn_video.setFont(QFont('Arial', 10))
        self.btn_video.clicked.connect(self.load_video)
        grid_layout.addWidget(self.btn_video, 1, 1)

        self.btn_camera = QPushButton('摄像头实时监测')
        self.btn_camera.setIcon(QIcon('icon/摄像头开.png'))  # Add button icon
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
            print(f"Loaded image path: {self.image_path}")  # Confirm that the path is correct.
            frame = cv2.imread(self.image_path)
            if frame is not None:
                print("Image loaded successfully")  # Confirm that the image has been loaded successfully.
                self.display_frame(frame, is_input=True)  # Display the original image for testing
                self.detect_image()
            else:
                print("Failed to load image")  # If the loading fails, print out the error message.

    def start_camera(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_camera)
        self.timer.start(30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # Set the width of the camera window
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # Set the height of the camera window

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
                num_detections += 1
        cv2.putText(frame, f'number: {num_detections}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2,
                    cv2.LINE_AA)
        return frame

    def display_frame(self, frame, is_input=True):
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()  # Use `.copy()` to ensure the persistence of data.
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
            # 对Process the frames, for example, by invoking the detect method.
            # self.detect(frame)  # Suppose you have a method that processes frames and returns results.
            # Display the original frame or the processed frame
            self.display_frame(frame, is_input=False)  # Suppose you want to display the processed results on the right side.
        else:
            self.timer.stop()  # If there is no frame available for reading, then stop the timer.


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
