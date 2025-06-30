"""
程序设定向导模块
多步骤的配置向导界面
"""
from PyQt5.QtWidgets import (QMainWindow, QStackedWidget, QVBoxLayout, QWidget, 
                            QPushButton, QHBoxLayout, QLabel, QMessageBox, QInputDialog)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from window_selector import WindowSelector
from region_selector import RegionSelector
from operation_editor import OperationEditor
from data_source_selector import DataSourceSelector
from config_manager import save_operation_config
from window_manager import WindowManager
from execution_window import ExecutionWindow
import pygetwindow as gw
import threading
import time

class SettingsWizard(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("程序设定向导")
        self.setMinimumSize(1000, 700)
        self.setup_ui()
        
        # 新增成员变量，用于控制窗口置顶
        self._keep_window_on_top = False
        self._on_top_thread = None
    
    def setup_ui(self):
        """初始化向导界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("程序设定向导")
        title_label.setFont(QFont("微软雅黑", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ecf0f1; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # 步骤指示器
        self.step_indicator = QLabel("步骤 1/4: 选择目标窗口")
        self.step_indicator.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.step_indicator.setAlignment(Qt.AlignCenter)
        # 步骤指示器美化
        self.step_indicator.setStyleSheet("background:#eaf6fb; color:#2980b9; border-radius:8px; padding:8px 0; margin-bottom:10px; font-size:16px;")
        main_layout.addWidget(self.step_indicator)
        
        # 堆叠窗口用于不同步骤
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # 创建步骤
        self.create_steps()
        
        # 导航按钮
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(18)

        self.prev_btn = QPushButton("← 上一步")
        self.prev_btn.setIcon(QIcon("icons/prev.png"))
        self.prev_btn.setMinimumHeight(38)
        self.prev_btn.setToolTip("返回上一步")
        self.prev_btn.clicked.connect(self.prev_step)
        self.prev_btn.setEnabled(False)

        self.next_btn = QPushButton("下一步 →")
        self.next_btn.setIcon(QIcon("icons/next.png"))
        self.next_btn.setMinimumHeight(38)
        self.next_btn.setToolTip("进入下一步")
        self.next_btn.clicked.connect(self.next_step)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setIcon(QIcon("icons/cancel.png"))
        self.cancel_btn.setMinimumHeight(38)
        self.cancel_btn.setStyleSheet("background-color: #e74c3c;")
        self.cancel_btn.setToolTip("取消并返回主界面")
        self.cancel_btn.clicked.connect(self.cancel_wizard)

        self.skip_and_execute_btn = QPushButton("跳过并直接执行")
        self.skip_and_execute_btn.setIcon(QIcon("icons/start.png"))
        self.skip_and_execute_btn.setStyleSheet("background-color: #27ae60; color: white;")
        # self.skip_and_execute_btn.clicked.connect(self.skip_and_execute)  # 删除此行
        self.skip_and_execute_btn.setVisible(False)

        self.test_operations_btn = QPushButton("测试操作逻辑")
        self.test_operations_btn.setIcon(QIcon("icons/test.png"))
        self.test_operations_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.test_operations_btn.clicked.connect(self.test_operations)
        self.test_operations_btn.setVisible(False)

        self.skip_data_source_btn = QPushButton("跳过数据源配置")
        self.skip_data_source_btn.setIcon(QIcon("icons/skip.png"))
        self.skip_data_source_btn.setStyleSheet("background-color: #8e44ad; color: white;")
        self.skip_data_source_btn.clicked.connect(self.skip_data_source)
        self.skip_data_source_btn.setVisible(False)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_btn)
        # 删除跳过并直接执行按钮
        # nav_layout.addWidget(self.skip_and_execute_btn)
        nav_layout.addWidget(self.test_operations_btn)
        nav_layout.addWidget(self.skip_data_source_btn)
        nav_layout.addWidget(self.next_btn)
        
        main_layout.addWidget(nav_widget)
        
        # 当前步骤索引
        self.current_step = 0
    
    def create_steps(self):
        """创建所有设定步骤"""
        # 步骤1: 窗口选择
        self.window_selector = WindowSelector()
        self.stacked_widget.addWidget(self.window_selector)

        # 步骤2: 区域选择
        self.region_selector = RegionSelector()
        self.stacked_widget.addWidget(self.region_selector)

        # 步骤3: 数据源选择（提前）
        self.data_source_selector = DataSourceSelector()
        self.stacked_widget.addWidget(self.data_source_selector)

        # 步骤4: 操作编辑（后置）
        self.operation_editor = OperationEditor()
        self.stacked_widget.addWidget(self.operation_editor)
    
    def prev_step(self):
        """返回到上一步"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step()
    
    def next_step(self):
        """前进到下一步"""
        if self.validate_current_step():
            if self.current_step == 0:
                # 1. 设置窗口信息
                self.region_selector.set_window_info(self.window_selector.get_selected_window())
                # 2. 激活目标窗口
                window_title = self.window_selector.get_selected_window()
                if window_title:
                    try:
                        win = gw.getWindowsWithTitle(window_title)
                        if win:
                            win[0].activate()
                            # 启动置顶线程
                            self._start_keep_on_top_thread(window_title)
                    except Exception:
                        pass
            elif self.current_step == 1:
                self.data_source_selector.set_operations(self.region_selector.get_regions())
            elif self.current_step == 2:
                # 进入第四步时，传递regions和data_source给操作编辑器
                self.operation_editor.set_regions(self.region_selector.get_regions())
                # 如果操作编辑器需要用到数据源，可以加下面一行
                # self.operation_editor.set_data_source(self.data_source_selector.get_data_source())
            elif self.current_step == 3:
                self.complete_setup()
                return

            self.current_step += 1
            self.update_step()

    def validate_current_step(self):
        """验证当前步骤是否完成"""
        if self.current_step == 0:
            return self.window_selector.is_valid()
        elif self.current_step == 1:
            return self.region_selector.is_valid()
        elif self.current_step == 2:
            return self.data_source_selector.is_valid()
        elif self.current_step == 3:
            return self.operation_editor.is_valid()
        return True
    
    def update_step(self):
        """更新UI以显示当前步骤"""
        self.stacked_widget.setCurrentIndex(self.current_step)

        # 更新步骤指示器
        step_names = [
            "步骤 1/4: 选择目标窗口",
            "步骤 2/4: 框选界面元素",
            "步骤 3/4: 选择数据源",
            "步骤 4/4: 编辑操作逻辑"
        ]
        self.step_indicator.setText(step_names[self.current_step])

        # 更新按钮状态
        self.prev_btn.setEnabled(self.current_step > 0)
        self.next_btn.setText("完成" if self.current_step == 3 else "下一步")
        # 删除第四步跳过按钮
        # self.skip_and_execute_btn.setVisible(self.current_step == 3)
        self.test_operations_btn.setVisible(self.current_step == 3)
        self.skip_data_source_btn.setVisible(self.current_step == 2)
        
        # 每次进入第四步都刷新操作编辑器的元素列表
        if self.current_step == 3:
            self.operation_editor.set_regions(self.region_selector.get_regions())
        # 只在第二步（框选）时保持目标窗口置顶，其他步骤自动停止
        if self.current_step != 1:
            self._keep_window_on_top = False
    
    def complete_setup(self):
        """完成设置并保存配置"""
        # 收集配置数据
        config = {
            "window_title": self.window_selector.get_selected_window(),
            "regions": self.region_selector.get_regions(),
            "data_source": self.data_source_selector.get_data_source(),
            "operations": self.operation_editor.get_operations()
        }
        
        # 弹出对话框让用户输入配置名称
        default_name = f"配置_{len(config['operations'])}步骤"
        config_name, ok = QInputDialog.getText(self, "保存配置", "请输入配置名称：", text=default_name)
        if not ok or not config_name.strip():
            QMessageBox.warning(self, "未保存", "未输入配置名称，配置未保存。")
            return
        config_name = config_name.strip()
        
        # 保存配置
        if save_operation_config(config_name, config):
            QMessageBox.information(self, "设置完成", f"配置已保存: {config_name}")
            WindowManager().return_to_main()
        else:
            QMessageBox.warning(self, "保存失败", "配置保存失败，请重试")
    
    def cancel_wizard(self):
        """取消向导并返回主窗口"""
        WindowManager().return_to_main()
    
    def test_operations(self):
        """无需数据源直接测试操作逻辑"""
        config = {
            "window_title": self.window_selector.get_selected_window(),
            "regions": self.region_selector.get_regions(),
            "data_source": None,
            "operations": self.operation_editor.get_operations()
        }
        config_name = f"测试_{len(config['operations'])}步骤"
        exec_win = ExecutionWindow(config_name)
        # 直接传递当前配置并覆盖load_config方法，使其不加载默认配置
        exec_win.config = config
        def fake_load_config():
            info_text = (
                f"窗口标题: {config.get('window_title', '无')}\n"
                f"操作步骤数量: {len(config.get('operations', []))}\n"
                f"循环次数: 1\n循环间隔: 0秒"
            )
            exec_win.config_info.setText(info_text)
            exec_win.progress_bar.setMaximum(len(config.get('operations', [])))
        exec_win.load_config = fake_load_config
        exec_win.load_config()
        exec_win.show()
    
    def skip_data_source(self):
        """跳过数据源配置，直接进入操作逻辑编辑"""
        # 跳过时，data_source 设为 None，直接进入下一步
        self.current_step = 3
        self.update_step()

    def _start_keep_on_top_thread(self, window_title):
        """
        后台线程定时将目标窗口置顶，仅在框选阶段（第二步）才置顶，进入属性输入或后续步骤时自动停止置顶
        """
        self._keep_window_on_top = True

        def keep_on_top():
            while self._keep_window_on_top and self.current_step == 1:
                try:
                    win = gw.getWindowsWithTitle(window_title)
                    if win:
                        try:
                            win[0].activate()
                        except Exception:
                            pass
                        try:
                            win[0].bringToFront()
                        except Exception:
                            pass
                        try:
                            win[0].set_foreground()
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(0.5)
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._keep_window_on_top = False
            self._on_top_thread.join()
        self._on_top_thread = threading.Thread(target=keep_on_top, daemon=True)
        self._on_top_thread.start()

    def closeEvent(self, event):
        """关闭时停止置顶线程"""
        self._keep_window_on_top = False
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._on_top_thread.join(timeout=1)
        super().closeEvent(event)
        self._keep_window_on_top = False
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._on_top_thread.join(timeout=1)
        super().closeEvent(event)
        time.sleep(0.5)
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._keep_window_on_top = False
            self._on_top_thread.join()
        self._on_top_thread = threading.Thread(target=keep_on_top, daemon=True)
        self._on_top_thread.start()

    def closeEvent(self, event):
        """关闭时停止置顶线程"""
        self._keep_window_on_top = False
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._on_top_thread.join(timeout=1)
        super().closeEvent(event)
        self._keep_window_on_top = False
        if self._on_top_thread and self._on_top_thread.is_alive():
            self._on_top_thread.join(timeout=1)
        super().closeEvent(event)
