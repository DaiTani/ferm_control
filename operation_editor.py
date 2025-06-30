"""
操作逻辑编辑模块
定义操作序列和逻辑
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, 
                            QListWidget, QGroupBox, QInputDialog, QListWidgetItem, QDialog, QDialogButtonBox)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import Qt, QSize

class OperationEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.operations = []
        self.regions = []
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("编辑操作逻辑：")
        title_label.setFont(QFont("微软雅黑", 15, QFont.Bold))
        title_label.setStyleSheet("color: #2980b9;")
        layout.addWidget(title_label)
        
        # 说明
        instruction_label = QLabel(
            "【操作说明】\n"
            "1. 左侧为可用元素，右侧为操作序列。\n"
            "2. 选中左侧元素后点击 ➡ 添加到操作序列。\n"
            "3. 选中右侧操作后可用 ⬅ 移除，⬆ 上移，⬇ 下移。\n"
            "4. 双击操作项或点击“配置操作”可设置点击次数/输入内容。"
        )
        instruction_label.setFont(QFont("微软雅黑", 12))
        instruction_label.setStyleSheet("color: #2980b9; margin-bottom:8px;")
        layout.addWidget(instruction_label)
        
        # 操作区域
        operation_layout = QHBoxLayout()
        
        # 可用元素列表
        elements_group = QGroupBox("可用元素")
        elements_group.setFont(QFont("微软雅黑", 13, QFont.Bold))
        elements_layout = QVBoxLayout(elements_group)
        self.elements_list = QListWidget()
        self.elements_list.setFont(QFont("微软雅黑", 12))
        self.elements_list.setIconSize(QSize(32, 32))
        elements_layout.addWidget(self.elements_list)
        operation_layout.addWidget(elements_group)
        
        # 操作按钮
        buttons_layout = QVBoxLayout()
        buttons_layout.setAlignment(Qt.AlignCenter)
        
        self.add_btn = QPushButton("➡")
        self.add_btn.setFont(QFont("微软雅黑", 12))
        self.add_btn.setIcon(QIcon("icons/right.png"))
        self.add_btn.setIconSize(QSize(24, 24))
        self.add_btn.setFixedSize(80, 40)
        self.add_btn.setToolTip("添加到操作序列")
        buttons_layout.addWidget(self.add_btn)
        self.add_btn.clicked.connect(self.add_to_sequence)
        
        self.remove_btn = QPushButton("⬅")
        self.remove_btn.setFont(QFont("微软雅黑", 12))
        self.remove_btn.setIcon(QIcon("icons/left.png"))
        self.remove_btn.setIconSize(QSize(24, 24))
        self.remove_btn.setFixedSize(80, 40)
        self.remove_btn.setToolTip("从操作序列移除")
        buttons_layout.addWidget(self.remove_btn)
        self.remove_btn.clicked.connect(self.remove_from_sequence)
        
        self.up_btn = QPushButton("⬆")
        self.up_btn.setFont(QFont("微软雅黑", 12))
        self.up_btn.setIcon(QIcon("icons/up.png"))
        self.up_btn.setIconSize(QSize(24, 24))
        self.up_btn.setFixedSize(80, 40)
        self.up_btn.setToolTip("上移")
        buttons_layout.addWidget(self.up_btn)
        self.up_btn.clicked.connect(self.move_up)
        
        self.down_btn = QPushButton("⬇")
        self.down_btn.setFont(QFont("微软雅黑", 12))
        self.down_btn.setIcon(QIcon("icons/down.png"))
        self.down_btn.setIconSize(QSize(24, 24))
        self.down_btn.setFixedSize(80, 40)
        self.down_btn.setToolTip("下移")
        buttons_layout.addWidget(self.down_btn)
        self.down_btn.clicked.connect(self.move_down)
        
        operation_layout.addLayout(buttons_layout)
        
        # 操作序列列表
        sequence_group = QGroupBox("操作序列")
        sequence_group.setFont(QFont("微软雅黑", 13, QFont.Bold))
        sequence_layout = QVBoxLayout(sequence_group)
        self.sequence_list = QListWidget()
        self.sequence_list.setFont(QFont("微软雅黑", 12))
        self.sequence_list.setIconSize(QSize(32, 32))
        self.sequence_list.itemDoubleClicked.connect(self.configure_operation)
        sequence_layout.addWidget(self.sequence_list)
        operation_layout.addWidget(sequence_group)
        
        layout.addLayout(operation_layout)
        
        # 配置操作按钮
        self.config_btn = QPushButton("⚙ 配置操作")
        self.config_btn.setFont(QFont("微软雅黑", 12))
        self.config_btn.setIcon(QIcon("icons/config.png"))
        self.config_btn.setEnabled(False)
        self.config_btn.clicked.connect(self.configure_operation)
        layout.addWidget(self.config_btn)
    
    def set_regions(self, regions):
        """设置区域数据"""
        self.regions = regions
        self.populate_elements_list()
        # 清空操作序列，避免上次遗留
        self.operations = []
        self.sequence_list.clear()
    
    def populate_elements_list(self):
        """填充可用元素列表"""
        self.elements_list.clear()
        for region in self.regions:
            item = QListWidgetItem(region["name"])
            item.setData(Qt.UserRole, region)
            # 显示缩略图
            if region.get("thumbnail"):
                item.setIcon(QIcon(region["thumbnail"]))
            self.elements_list.addItem(item)
    
    def add_to_sequence(self):
        """将元素添加到操作序列"""
        selected_items = self.elements_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            region = item.data(Qt.UserRole)
            operation = {
                "region": region,
                "action": "click",
                "config": {"times": 1}
            }
            
            # 创建序列项
            seq_item = QListWidgetItem(f"点击: {region['name']}")
            seq_item.setData(Qt.UserRole, operation)
            self.sequence_list.addItem(seq_item)
            self.operations.append(operation)
        
        self.config_btn.setEnabled(len(self.operations) > 0)
    
    def remove_from_sequence(self):
        """从操作序列中移除元素"""
        selected_items = self.sequence_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.sequence_list.row(item)
            self.sequence_list.takeItem(row)
            del self.operations[row]
        
        self.config_btn.setEnabled(len(self.operations) > 0)
    
    def move_up(self):
        """上移操作项"""
        current_row = self.sequence_list.currentRow()
        if current_row > 0:
            item = self.sequence_list.takeItem(current_row)
            operation = self.operations.pop(current_row)
            self.sequence_list.insertItem(current_row - 1, item)
            self.operations.insert(current_row - 1, operation)
            self.sequence_list.setCurrentRow(current_row - 1)
    
    def move_down(self):
        """下移操作项"""
        current_row = self.sequence_list.currentRow()
        if current_row < self.sequence_list.count() - 1 and current_row >= 0:
            item = self.sequence_list.takeItem(current_row)
            operation = self.operations.pop(current_row)
            self.sequence_list.insertItem(current_row + 1, item)
            self.operations.insert(current_row + 1, operation)
            self.sequence_list.setCurrentRow(current_row + 1)
    
    def configure_operation(self):
        """配置选中的操作"""
        current_item = self.sequence_list.currentItem()
        if not current_item:
            return
        
        row = self.sequence_list.row(current_item)
        operation = self.operations[row]
        region = operation["region"]
        
        # 根据区域类型配置
        if "按钮" in region["type"]:
            times, ok = QInputDialog.getInt(
                self, 
                "配置点击操作", 
                f"设置点击次数 ({region['name']}):", 
                operation["config"].get("times", 1), 
                1, 10, 1
            )
            if ok:
                operation["action"] = "click"
                operation["config"] = {"times": times}
                current_item.setText(f"点击({times}次): {region['name']}")
        
        elif "文本" in region["type"]:
            text, ok = QInputDialog.getText(
                self,
                "配置输入操作",
                f"输入文本内容 ({region['name']}):",
                text=operation["config"].get("text", "")
            )
            if ok:
                operation["action"] = "input"
                operation["config"] = {"text": text}
                current_item.setText(f"输入('{text}'): {region['name']}")
    
    def get_operations(self):
        """获取操作序列"""
        return self.operations
    
    def is_valid(self):
        """验证操作序列是否有效"""
        return len(self.operations) > 0

    def show_region_thumbnail(self, current, previous):
        """显示选中元素的框选图片"""
        if not current:
            self.thumbnail_label.setText("框选图片预览")
            self.thumbnail_label.setPixmap(QPixmap())
            return
        region = current.data(Qt.UserRole)
        pixmap = region.get("thumbnail")
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            self.thumbnail_label.setPixmap(pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.thumbnail_label.setText("")
        elif hasattr(pixmap, "toqpixmap"):  # 兼容PIL.ImageQt.ImageQt
            qpixmap = QPixmap.fromImage(pixmap)
            self.thumbnail_label.setPixmap(qpixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.thumbnail_label.setText("")
        else:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("无图片")
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            self.thumbnail_label.setPixmap(pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.thumbnail_label.setText("")
        elif hasattr(pixmap, "toqpixmap"):  # 兼容PIL.ImageQt.ImageQt
            qpixmap = QPixmap.fromImage(pixmap)
            self.thumbnail_label.setPixmap(qpixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.thumbnail_label.setText("")
        else:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("无图片")
