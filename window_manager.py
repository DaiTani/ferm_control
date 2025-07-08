from PyQt5.QtWidgets import QApplication

class WindowManager:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register_main_window(self, main_window):
        """
        注册主窗口
        :param main_window: 主窗口实例
        """
        self.main_window = main_window

    def open_window(self, window):
        """
        打开新窗口并隐藏主窗口
        :param window: 要打开的窗口实例
        """
        if hasattr(self, 'main_window'):
            self.main_window.hide()
        window.show()
        self.current_window = window

    def return_to_main(self):
        """
        返回主窗口
        """
        if hasattr(self, 'main_window'):
            self.main_window.show()
        # 可选：关闭当前窗口
        if hasattr(self, 'current_window'):
            self.current_window.close()
            self.current_window = None