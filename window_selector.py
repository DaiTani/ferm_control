"""
窗口选择模块
选择要操作的目标应用程序窗口
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon, QFont
import pygetwindow as gw

class WindowSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_window = None
        self.setup_ui()
        self.populate_window_list()
    
    def setup_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("请选择目标应用程序窗口：")
        title_label.setFont(QFont("微软雅黑", 15, QFont.Bold))
        title_label.setStyleSheet("color: #2980b9;")
        layout.addWidget(title_label)
        
        # 窗口列表
        self.window_list = QListWidget()
        self.window_list.setFont(QFont("微软雅黑", 13))
        self.window_list.itemSelectionChanged.connect(self.on_window_selected)
        layout.addWidget(self.window_list)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新窗口列表")
        refresh_btn.setFont(QFont("微软雅黑", 13))
        refresh_btn.setIcon(QIcon("icons/refresh.png"))
        refresh_btn.clicked.connect(self.populate_window_list)
        layout.addWidget(refresh_btn)
    
    def populate_window_list(self):
        """填充窗口列表"""
        self.window_list.clear()
        
        # 获取所有窗口标题
        windows = gw.getAllTitles()
        for title in windows:
            if title:  # 过滤空标题
                self.window_list.addItem(title)
    
    def on_window_selected(self):
        """处理窗口选择事件"""
        selected_items = self.window_list.selectedItems()
        if selected_items:
            self.selected_window = selected_items[0].text()
    
    def get_selected_window(self):
        """获取选中的窗口"""
        return self.selected_window
    
    def is_valid(self):
        """验证是否已选择窗口"""
        return self.selected_window is not None
