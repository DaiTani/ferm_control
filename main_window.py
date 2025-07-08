"""
主界面模块
程序的主界面和主要功能入口
"""
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSizePolicy, QSpacerItem, QInputDialog)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from settings_wizard import SettingsWizard
from execution_window import ExecutionWindow
from config_manager import get_last_operation, list_configs, delete_config
from window_manager import WindowManager

class MainWindow(QMainWindow):
    def __init__(self, window_manager: WindowManager):
        super().__init__()
        self.window_manager = window_manager
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("自适应操作执行系统")
        self.setGeometry(100, 100, 900, 650)
        self.setMinimumSize(700, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(60, 40, 60, 40)
        main_layout.setSpacing(35)

        # 欢迎说明
        welcome_label = QLabel("欢迎使用自适应操作执行系统！\n\n请通过“程序设定”按钮配置自动化流程，或选择已有配置执行。")
        welcome_label.setFont(QFont("微软雅黑", 16))
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("color: #555; margin-bottom: 10px;")
        main_layout.addWidget(welcome_label)

        # 标题
        title_label = QLabel("自适应操作执行系统")
        title_label.setFont(QFont("Arial", 30, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2980b9; padding: 18px;")
        main_layout.addWidget(title_label)

        # 按钮区
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(25)

        select_config_btn = QPushButton("📂 选择配置并执行")
        select_config_btn.setIcon(QIcon("icons/repeat.png"))
        select_config_btn.setMinimumHeight(54)
        select_config_btn.setFont(QFont("微软雅黑", 17))
        select_config_btn.clicked.connect(self.select_and_execute_config)
        btn_layout.addWidget(select_config_btn)

        delete_config_btn = QPushButton("🗑 删除配置文件")
        delete_config_btn.setIcon(QIcon("icons/delete.png"))
        delete_config_btn.setMinimumHeight(54)
        delete_config_btn.setFont(QFont("微软雅黑", 17))
        delete_config_btn.clicked.connect(self.delete_config_file)
        btn_layout.addWidget(delete_config_btn)

        settings_btn = QPushButton("⚙ 程序设定")
        settings_btn.setIcon(QIcon("icons/settings.png"))
        settings_btn.setMinimumHeight(54)
        settings_btn.setFont(QFont("微软雅黑", 17))
        settings_btn.clicked.connect(self.open_settings_wizard)
        btn_layout.addWidget(settings_btn)

        exit_btn = QPushButton("⏻ 退出")
        exit_btn.setIcon(QIcon("icons/exit.png"))
        exit_btn.setMinimumHeight(54)
        exit_btn.setFont(QFont("微软雅黑", 17))
        exit_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        exit_btn.clicked.connect(self.close)
        btn_layout.addWidget(exit_btn)

        main_layout.addLayout(btn_layout)
        main_layout.addStretch()

        self.status_bar = self.statusBar()
        self.status_bar.setFont(QFont("微软雅黑", 11))
        self.status_bar.showMessage("就绪")

        # 美化
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f5fafd, stop:1 #e0eafc);
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 12px;
                padding: 14px 32px;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                background-color: #217dbb;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QLabel {
                color: #2d3436;
                font-size: 16px;
            }
        """)

    def select_and_execute_config(self):
        """弹出配置选择框并执行所选配置"""
        configs = list_configs()
        if not configs:
            self.status_bar.showMessage("没有可用的配置，请先进行程序设定。")
            return
        config_name, ok = QInputDialog.getItem(self, "选择配置", "请选择要执行的配置：", configs, 0, False)
        if ok and config_name:
            self.open_execution_window(config_name)

    def delete_config_file(self):
        """弹出配置选择框并删除所选配置"""
        configs = list_configs()
        if not configs:
            self.status_bar.showMessage("没有可用的配置。")
            return
        config_name, ok = QInputDialog.getItem(self, "删除配置", "请选择要删除的配置：", configs, 0, False)
        if ok and config_name:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(self, "确认删除", f"确定要删除配置 [{config_name}] 吗？", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                if delete_config(config_name):
                    self.status_bar.showMessage(f"配置 [{config_name}] 已删除。")
                else:
                    self.status_bar.showMessage(f"删除配置 [{config_name}] 失败。")
            else:
                self.status_bar.showMessage("已取消删除。")

    def open_settings_wizard(self):
        """打开程序设定向导"""
        settings_wizard = SettingsWizard(self)
        self.window_manager.open_window(settings_wizard)

    def open_execution_window(self, config_name):
        """打开执行窗口"""
        execution_window = ExecutionWindow(config_name)
        self.window_manager.open_window(execution_window)
        settings_wizard = SettingsWizard(self)
        self.window_manager.open_window(settings_wizard)
    
    def open_execution_window(self, config_name):
        """打开执行窗口"""
        execution_window = ExecutionWindow(config_name)
        self.window_manager.open_window(execution_window)
