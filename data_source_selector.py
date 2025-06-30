"""
数据源选择模块
选择数据来源
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QIcon, QFont
import os

class DataSourceSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.data_source = None
        self.excel_path = None
        self.csv_path = None
        self.db_path = None
        self.setup_ui()
    
    def setup_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title_label = QLabel("选择数据源")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setStyleSheet("color:#2980b9;")
        layout.addWidget(title_label)

        instruction_label = QLabel(
            "【操作说明】\n"
            "请选择输入数据的来源（如Excel、CSV、数据库等）。\n"
            "后续版本将支持更多数据源类型。"
        )
        instruction_label.setFont(QFont("微软雅黑", 12))
        instruction_label.setStyleSheet("color: #888; margin-bottom:8px;")
        layout.addWidget(instruction_label)
        
        # 数据源选择区域
        source_group = QGroupBox("可用数据源")
        source_group.setFont(QFont("微软雅黑", 13, QFont.Bold))
        source_layout = QVBoxLayout(source_group)
        
        # Excel 数据源按钮
        excel_btn = QPushButton("Excel 文件")
        excel_btn.setFont(QFont("微软雅黑", 13))
        excel_btn.setIcon(QIcon("icons/excel.png"))
        excel_btn.clicked.connect(self.select_excel_file)
        excel_btn.setMinimumHeight(40)
        source_layout.addWidget(excel_btn)
        
        # CSV 数据源按钮
        csv_btn = QPushButton("CSV 文件")
        csv_btn.setFont(QFont("微软雅黑", 13))
        csv_btn.setIcon(QIcon("icons/csv.png"))
        csv_btn.clicked.connect(self.select_csv_file)
        csv_btn.setMinimumHeight(40)
        source_layout.addWidget(csv_btn)
        
        # 数据库数据源按钮
        db_btn = QPushButton("本地数据库")
        db_btn.setFont(QFont("微软雅黑", 13))
        db_btn.setIcon(QIcon("icons/database.png"))
        db_btn.clicked.connect(self.select_db_file)
        db_btn.setMinimumHeight(40)
        source_layout.addWidget(db_btn)
        
        layout.addWidget(source_group)
        
        # 已选择数据源显示
        self.selected_group = QGroupBox("已选择的数据源")
        self.selected_group.setFont(QFont("微软雅黑", 13, QFont.Bold))
        selected_layout = QVBoxLayout(self.selected_group)
        self.selected_label = QLabel("尚未选择数据源")
        self.selected_label.setFont(QFont("微软雅黑", 12))
        self.selected_label.setStyleSheet("font-size:14px; color:#222;")
        selected_layout.addWidget(self.selected_label)
        self.selected_group.setStyleSheet("QGroupBox { font-weight:bold; color:#2980b9; }")
        layout.addWidget(self.selected_group)
        
        # 配置按钮
        self.config_btn = QPushButton("配置数据源")
        self.config_btn.setFont(QFont("微软雅黑", 13))
        self.config_btn.setIcon(QIcon("icons/config.png"))
        self.config_btn.setEnabled(False)
        self.config_btn.setMinimumHeight(36)
        self.config_btn.clicked.connect(self.configure_data_source)
        layout.addWidget(self.config_btn)
    
    def set_operations(self, operations):
        """设置操作信息"""
        self.operations = operations
    
    def select_excel_file(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "选择Excel文件", "", "Excel Files (*.xlsx *.xls)")
        if file_path:
            self.data_source = {"type": "excel", "path": file_path}
            self.selected_label.setText(f"Excel 文件: {os.path.basename(file_path)}")
            self.config_btn.setEnabled(True)\
            
    def select_csv_file(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            self.data_source = {"type": "csv", "path": file_path}
            self.selected_label.setText(f"CSV 文件: {os.path.basename(file_path)}")
            self.config_btn.setEnabled(True)

    def select_db_file(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据库文件", "", "数据库文件 (*.db *.sqlite *.sqlite3);;所有文件 (*)")
        if file_path:
            self.data_source = {"type": "database", "path": file_path}
            self.selected_label.setText(f"本地数据库: {os.path.basename(file_path)}")
            self.config_btn.setEnabled(True)

    def configure_data_source(self):
        """配置数据源"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "数据源配置", "数据源配置功能将在后续版本中实现")
    
    def get_data_source(self):
        """获取数据源配置"""
        return self.data_source
    
    def is_valid(self):
        """验证是否已选择数据源"""
        return self.data_source is not None
