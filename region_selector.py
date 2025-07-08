"""
区域框选模块
框选界面元素并指定类型
"""
import pygetwindow as gw
import pyautogui
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, 
                             QPushButton, QDialog, QFormLayout, QDialogButtonBox, QGroupBox, QLineEdit, QComboBox)
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QSize
import uuid
import os

TEMPLATE_DIR = "templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

class ScreenshotLabel(QLabel):
    region_selected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        self.drawing = False
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.current_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.current_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.current_rect is not None:
            self.drawing = False
            self.region_selected.emit(self.current_rect)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_rect and (self.drawing or not self.drawing):
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(self.current_rect)
            painter.end()

class FullScreenMask(QWidget):
    region_selected = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setWindowState(Qt.WindowFullScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.start_point = None
        self.end_point = None
        self.current_rect = None
        self.drawing = False
        self.setCursor(Qt.CrossCursor)
        # 允许属性对话框不被遮挡
        self.setWindowModality(Qt.NonModal)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.globalPos()
            self.end_point = event.globalPos()
            self.current_rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.globalPos()
            self.current_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.current_rect is not None:
            self.drawing = False
            self.region_selected.emit(self.current_rect)
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 半透明黑色遮罩
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))
        if self.current_rect:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 0, 0, 40))
            painter.drawRect(self.current_rect)

class RegionPropertyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("元素属性")
        layout = QFormLayout(self)
        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["按钮", "文本框", "下拉菜单", "复选框", "其他"])
        layout.addRow("元素名称:", self.name_edit)
        layout.addRow("元素类型:", self.type_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Window)

    def get_data(self):
        return self.name_edit.text().strip(), self.type_combo.currentText()

class RegionSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.regions = []
        self.window_title = None
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 标题
        title_label = QLabel("框选界面元素")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setStyleSheet("color:#2980b9;")
        layout.addWidget(title_label)

        instruction_label = QLabel(
            "【操作说明】\n"
            "1. 点击下方“开始框选”按钮。\n"
            "2. 在目标窗口上拖动鼠标选择区域。\n"
            "3. 框选后输入元素名称和类型。\n"
            "4. 可多次框选，所有元素会显示在下方列表。"
        )
        instruction_label.setFont(QFont("微软雅黑", 12))
        instruction_label.setStyleSheet("color: #888; margin-bottom:8px;")
        layout.addWidget(instruction_label)
        
        # 截图显示区域
        self.screenshot_label = QLabel()
        self.screenshot_label.setAlignment(Qt.AlignCenter)
        self.screenshot_label.setMinimumHeight(400)
        self.screenshot_label.setFont(QFont("Consolas", 11))
        layout.addWidget(self.screenshot_label)
        
        # 控制区域
        control_layout = QHBoxLayout()
        self.capture_btn = QPushButton("开始框选")
        self.capture_btn.setFont(QFont("微软雅黑", 13))
        self.capture_btn.setIcon(QIcon("icons/capture.png"))
        self.capture_btn.clicked.connect(self.start_capture)
        control_layout.addWidget(self.capture_btn)
        
        layout.addLayout(control_layout)
        
        # 已选择区域列表
        self.regions_group = QGroupBox("已选择的元素")
        self.regions_group.setFont(QFont("微软雅黑", 13, QFont.Bold))
        self.regions_group.setStyleSheet("QGroupBox { font-weight:bold; color:#2980b9; }")
        regions_layout = QVBoxLayout(self.regions_group)
        self.regions_list = QLabel("尚未选择任何元素")
        self.regions_list.setFont(QFont("微软雅黑", 12))
        self.regions_list.setStyleSheet("font-size:14px; color:#222;")
        regions_layout.addWidget(self.regions_list)
        layout.addWidget(self.regions_group)
    
    def set_window_info(self, window_title):
        self.window_title = window_title
        self.update_screenshot()
    
    def update_screenshot(self):
        """更新窗口截图"""
        if not self.window_title:
            return
            
        try:
            window = gw.getWindowsWithTitle(self.window_title)[0]
            self.screenshot = pyautogui.screenshot(region=(
                window.left, window.top, window.width, window.height
            ))
            from PIL.ImageQt import ImageQt
            qimage = ImageQt(self.screenshot)
            pixmap = QPixmap.fromImage(qimage)
            self.screenshot_label.setPixmap(pixmap.scaled(
                self.screenshot_label.width(),
                self.screenshot_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        except Exception as e:
            self.screenshot_label.setText(f"无法获取窗口截图: {str(e)}")

    def start_capture(self):
        """全屏遮罩进行跨窗口框选，框选结束后弹出属性框"""
        self.capture_btn.setEnabled(False)
        self.mask = FullScreenMask()
        self.mask.region_selected.connect(self.on_region_selected)
        self.mask.show()

    def on_region_selected(self, rect):
        """框选完成后弹窗输入属性并添加区域"""
        try:
            window = gw.getWindowsWithTitle(self.window_title)[0]
            wx, wy = window.left, window.top
            # 框选全局坐标
            global_x, global_y = rect.x(), rect.y()
            # 计算窗口内相对坐标
            rel_x = global_x - wx
            rel_y = global_y - wy
            region_rect = QRect(rel_x, rel_y, rect.width(), rect.height())
            print(f"DEBUG 框选: window.left={wx}, window.top={wy}, global_x={global_x}, global_y={global_y}, rel_x={rel_x}, rel_y={rel_y}, width={rect.width()}, height={rect.height()}")
            # 截取区域图片
            screenshot = pyautogui.screenshot(region=(
                global_x,
                global_y,
                rect.width(),
                rect.height()
            ))
            from PIL.ImageQt import ImageQt
            qimage = ImageQt(screenshot)
            pixmap = QPixmap.fromImage(qimage)
            # 保存模板图片
            template_filename = f"{uuid.uuid4().hex}.png"
            template_path = os.path.join(TEMPLATE_DIR, template_filename)
            screenshot.save(template_path)
        except Exception:
            region_rect = rect
            pixmap = None
            template_path = None

        dlg = RegionPropertyDialog(self)
        dlg.setWindowModality(Qt.ApplicationModal)
        if dlg.exec_() == QDialog.Accepted:
            name, type_ = dlg.get_data()
            if name:
                # 这里必须用 region_rect.x()/y()，不能用 rect.x()、global_x
                region_data = {
                    "name": name,
                    "type": type_,
                    "rect": {
                        "x": rel_x,  # 必须用 rel_x
                        "y": rel_y,  # 必须用 rel_y
                        "width": region_rect.width(),
                        "height": region_rect.height()
                    },
                    "thumbnail": pixmap,
                    "template_path": template_path if template_path else ""
                }
                print(f"DEBUG 保存: {region_data['rect']}")
                assert region_data['rect']['x'] == rel_x, f"保存的x不是相对坐标: {region_data['rect']['x']} != {rel_x}"
                assert region_data['rect']['y'] == rel_y, f"保存的y不是相对坐标: {region_data['rect']['y']} != {rel_y}"
                self.regions.append(region_data)
                self.update_regions_list()
        self.capture_btn.setEnabled(True)
        self.mask = None

    def update_regions_list(self):
        """更新已选择区域列表"""
        from PyQt5.QtWidgets import QListWidget, QListWidgetItem
        # 用列表控件替换label，显示图片和名称
        if not hasattr(self, "regions_widget"):
            self.regions_widget = QListWidget()
            self.regions_widget.setFont(QFont("微软雅黑", 12))
            self.regions_widget.setIconSize(QSize(64, 48))
            layout = self.regions_group.layout()
            layout.removeWidget(self.regions_list)
            self.regions_list.deleteLater()
            layout.addWidget(self.regions_widget)
        self.regions_widget.clear()
        if not self.regions:
            item = QListWidgetItem("尚未选择任何元素")
            item.setFont(QFont("微软雅黑", 12))
            self.regions_widget.addItem(item)
        else:
            for i, r in enumerate(self.regions):
                text = f"{i+1}. {r['name']} ({r['type']})"
                item = QListWidgetItem(text)
                item.setFont(QFont("微软雅黑", 12))
                if r.get("thumbnail"):
                    item.setIcon(QIcon(r["thumbnail"]))
                self.regions_widget.addItem(item)
    
    def get_regions(self):
        """获取所有选择的区域"""
        return self.regions
    
    def is_valid(self):
        """验证是否已选择至少一个区域"""
        return len(self.regions) > 0
        return len(self.regions) > 0
