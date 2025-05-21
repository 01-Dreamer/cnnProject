import sys
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import warnings
from cnnModel import CNN

warnings.filterwarnings("ignore")


# PyQt画板
class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super(DrawingWidget, self).__init__(parent)
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_image(self):
        try:
            image = self.image.scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image = image.convertToFormat(QImage.Format_Grayscale8)

            width, height = image.width(), image.height()
            if width != 28 or height != 28:
                print(f"图像尺寸错误: {width}x{height}, 预期尺寸: 28x28")
                return None

            image.save("debug_qt_image.png")
            print("调试图片已保存 'debug_qt_image.png'")

            img_array = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    pixel = image.pixel(x, y)
                    gray = (pixel & 0xFF)
                    img_array[y, x] = gray

            img_array = 255 - img_array

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            img_tensor = transform(img_array).unsqueeze(0)
            return img_tensor
        except Exception as e:
            print(f"图像获取错误: {e}")
            return None


# PyQt主界面
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("手写数字识别")
        self.setFixedSize(300, 500)

        self.device = torch.device("cpu")
        self.model = CNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load('digit_cnn.pth', map_location=self.device, weights_only=True))
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            sys.exit(1)
        self.model.eval()

        self.drawing_widget = DrawingWidget(self)
        self.predict_button = QPushButton("预测结果", self)
        self.clear_button = QPushButton("清除画板", self)
        self.result_label = QLabel("请写数字并点击预测", self)
        self.result_label.setAlignment(Qt.AlignCenter)

        button_width = self.drawing_widget.width()
        self.predict_button.setFixedWidth(button_width)
        self.clear_button.setFixedWidth(button_width)

        layout = QVBoxLayout()
        layout.addWidget(self.drawing_widget)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.drawing_widget.clear)

    def predict(self):
        try:
            img_tensor = self.drawing_widget.get_image()
            if img_tensor is None:
                self.result_label.setText("无法处理图像")
                return
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                output = self.model(img_tensor)
                prediction = torch.argmax(output, dim=1).item()
            self.result_label.setText(f"预测结果: {prediction}")
            print(f"预测结果: {prediction}")
        except Exception as e:
            print(f"预测过程错误: {e}")
            self.result_label.setText(f"预测过程错误: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
