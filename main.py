"""
主程序入口
初始化应用和启动主窗口
"""
import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow
from window_manager import WindowManager

def main():
    app = QApplication(sys.argv)
    
    # 创建窗口管理器
    window_manager = WindowManager()
    
    # 创建主窗口
    main_window = MainWindow(window_manager)
    
    # 注册主窗口到窗口管理器
    window_manager.register_main_window(main_window)
    
    # 显示主窗口
    main_window.show()
    
    # 启动应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
