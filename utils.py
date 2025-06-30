"""
工具函数模块
提供通用工具函数
"""
import os
import sys

def resource_path(relative_path):
    """
    获取资源的绝对路径
    
    参数:
        relative_path (str): 资源相对路径
        
    返回:
        str: 资源的绝对路径
    """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
