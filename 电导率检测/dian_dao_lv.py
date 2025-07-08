import machine
import time
import math
from machine import Pin, ADC, UART
import network
import socket
import errno

# 定义TDS传感器引脚 - 使用ESP32-S3的ADC通道
TDS_SENSOR_PIN = 1   # GPIO1 (ADC1_CH0)
TDS_SENSOR_PIN2 = 2  # GPIO2 (ADC1_CH1)
TDS_SENSOR_PIN3 = 3  # GPIO3 (ADC1_CH2)
TDS_SENSOR_PIN4 = 4  # GPIO4 (ADC1_CH3)

# 控制引脚
CONTROL_PIN2 = 5  # GPIO5
CONTROL_PIN4 = 6  # GPIO6
CONTROL_PIN7 = 7  # GPIO7

# LED引脚
LED_PIN = 8  # GPIO8

# ESP32-S3 ADC参数
ADC_RESOLUTION = 4095.0  # 12位ADC最大值
ADC_REF_VOLTAGE = 3.3    # 参考电压

# 其他常量
SCOUNT = 30
analog_buffer = [0] * SCOUNT
analog_buffer_index = 0
analog_buffer2 = [0] * SCOUNT
analog_buffer_index2 = 0
analog_buffer3 = [0] * SCOUNT
analog_buffer_index3 = 0
analog_buffer4 = [0] * SCOUNT
analog_buffer_index4 = 0

average_voltage = 0
tds_value = 0
average_voltage2 = 0
tds_value2 = 0
average_voltage3 = 0
tds_value3 = 0
average_voltage4 = 0
tds_value4 = 0
temperature = 25

pin2_active = False
pin4_active = False
pin7_active = False
trigger_time = 0

previous_millis = 0
led_state = False  # False = LOW, True = HIGH
HIGH_DURATION = 1000  # 高电平持续时间(毫秒)
LOW_DURATION = 1000   # 低电平持续时间(毫秒)

# WiFi设置
WIFI_SSID = "test" #WiFi名称
WIFI_PASSWORD = "88888888"  # WiFi密码

# UDP设置 - 修改为你的接收端IP和端口
UDP_IP = "10.185.17.192"  # 接收数据的电脑IP
UDP_PORT = 1222           # 接收数据的端口,默认端口为1222 

# 初始化ADC
adc1 = ADC(Pin(TDS_SENSOR_PIN))
adc1.atten(ADC.ATTN_11DB)  # 设置输入电压范围为0-3.3V
adc1.width(ADC.WIDTH_12BIT)  # 12位分辨率

adc2 = ADC(Pin(TDS_SENSOR_PIN2))
adc2.atten(ADC.ATTN_11DB)
adc2.width(ADC.WIDTH_12BIT)

adc3 = ADC(Pin(TDS_SENSOR_PIN3))
adc3.atten(ADC.ATTN_11DB)
adc3.width(ADC.WIDTH_12BIT)

adc4 = ADC(Pin(TDS_SENSOR_PIN4))
adc4.atten(ADC.ATTN_11DB)
adc4.width(ADC.WIDTH_12BIT)

# 初始化控制引脚
control_pin2 = Pin(CONTROL_PIN2, Pin.OUT, value=1)  # HIGH
control_pin4 = Pin(CONTROL_PIN4, Pin.OUT, value=1)  # HIGH
control_pin7 = Pin(CONTROL_PIN7, Pin.OUT, value=0)  # LOW

# 初始化LED引脚
led_pin = Pin(LED_PIN, Pin.OUT, value=led_state)

# 初始化串口 (UART1: TX=17, RX=18)
uart = UART(1, baudrate=9600, tx=17, rx=18)

wlan = network.WLAN(network.STA_IF)
wlan.active(True)

# 全局变量存储最新数据
latest_data = "#0.00,0.00,0.00,0.00$"
last_sent_data = None  # 存储上次发送的数据
DATA_CHANGE_THRESHOLD = 0.1  # 数据变化阈值（0.1ppm）
udp_socket = None  # UDP套接字
last_wifi_check = 0
WIFI_CHECK_INTERVAL = 10000  # 每10秒检查一次WiFi连接
last_udp_send = 0
UDP_SEND_INTERVAL = 200  # 每200ms发送一次数据

def get_median_num(b_array, i_filter_len):
    """中值滤波函数"""
    b_tab = b_array[:]  # 复制列表
    # 冒泡排序
    for j in range(i_filter_len - 1):
        for i in range(i_filter_len - 1 - j):
            if b_tab[i] > b_tab[i + 1]:
                b_temp = b_tab[i]
                b_tab[i] = b_tab[i + 1]
                b_tab[i + 1] = b_temp
    # 返回中值
    if i_filter_len % 2 == 1:
        return b_tab[(i_filter_len - 1) // 2]
    else:
        return (b_tab[i_filter_len // 2 - 1] + b_tab[i_filter_len // 2]) // 2

def connect_wifi(max_retries=5, retry_delay=2):
    """连接到WiFi网络"""
    global udp_socket
    
    if wlan.isconnected():
        print("Already connected to WiFi")
        return True
        
    print("Connecting to WiFi...")
    wlan.active(True)
    
    for attempt in range(max_retries):
        try:
            wlan.connect(WIFI_SSID, WIFI_PASSWORD)
            max_wait = 10
            while max_wait > 0:
                if wlan.isconnected():
                    print("WiFi Connected!")
                    print("IP Address:", wlan.ifconfig()[0])
                    
                    # 创建UDP套接字
                    try:
                        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        udp_socket.settimeout(0.1)  # 设置超时时间
                        print("UDP socket created")
                    except Exception as e:
                        print("Error creating UDP socket:", e)
                        udp_socket = None
                    
                    return True
                max_wait -= 1
                print('Waiting for connection...')
                time.sleep(1)
                
            print(f"WiFi connection attempt {attempt+1}/{max_retries} failed")
        except Exception as e:
            print(f"Error during WiFi connection: {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        
    print("WiFi Connection Failed after retries")
    return False

def send_udp_data():
    """通过UDP发送数据到指定IP和端口，仅在数据有效且变化时发送"""
    global udp_socket, latest_data, last_sent_data
    
    # 检查数据有效性
    if latest_data is None or len(latest_data) < 10 or "#0.00,0.00,0.00,0.00$" in latest_data:
        print("Invalid data, skipping send")
        return False
    
    # 检查数据是否有显著变化
    if last_sent_data is not None:
        # 解析当前数据
        try:
            curr_values = [float(x) for x in latest_data[1:-1].split(',')]
            prev_values = [float(x) for x in last_sent_data[1:-1].split(',')]
            
            # 检查是否有任一值变化超过阈值
            has_significant_change = False
            for i in range(4):
                if abs(curr_values[i] - prev_values[i]) >= DATA_CHANGE_THRESHOLD:
                    has_significant_change = True
                    break
            
            if not has_significant_change:
                return False  # 没有显著变化，不发送
        except:
            # 如果解析失败，继续发送
            pass
    
    # 网络连接检查
    if not wlan.isconnected():
        print("WiFi not connected")
        return False
        
    if udp_socket is None:
        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.settimeout(0.1)
        except Exception as e:
            print("Error recreating UDP socket:", e)
            return False
    
    try:
        # 发送数据并记录
        udp_socket.sendto(latest_data.encode('utf-8'), (UDP_IP, UDP_PORT))
        last_sent_data = latest_data
        print(f"UDP sent: {latest_data}")
        return True
    except OSError as e:
        print("UDP send error:", e)
        # 尝试重建socket
        try:
            udp_socket.close()
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.settimeout(0.1)
        except Exception as e:
            print("Error recreating UDP socket:", e)
            udp_socket = None
    except Exception as e:
        print("Unexpected error in UDP send:", e)
    
    return False

# 主程序开始
print("Starting application...")

# 初始化网络连接
connect_wifi()

# 初始化时间点变量
analog_sample_timepoint = time.ticks_ms()
print_timepoint = time.ticks_ms()

while True:
    current_millis = time.ticks_ms()
    
    # LED状态控制
    if led_state:
        if time.ticks_diff(current_millis, previous_millis) >= HIGH_DURATION:
            led_state = False
            led_pin.value(led_state)
            previous_millis = current_millis
    else:
        if time.ticks_diff(current_millis, previous_millis) >= LOW_DURATION:
            led_state = True
            led_pin.value(led_state)
            previous_millis = current_millis

    # 传感器采样逻辑
    if time.ticks_diff(current_millis, analog_sample_timepoint) > 40:
        analog_sample_timepoint = current_millis
        
        # 读取所有传感器（12位ADC值）
        analog_buffer[analog_buffer_index] = adc1.read()
        analog_buffer2[analog_buffer_index2] = adc2.read()
        analog_buffer3[analog_buffer_index3] = adc3.read()
        analog_buffer4[analog_buffer_index4] = adc4.read()
        
        # 更新缓冲区索引
        analog_buffer_index = (analog_buffer_index + 1) % SCOUNT
        analog_buffer_index2 = (analog_buffer_index2 + 1) % SCOUNT
        analog_buffer_index3 = (analog_buffer_index3 + 1) % SCOUNT
        analog_buffer_index4 = (analog_buffer_index4 + 1) % SCOUNT

    # 数据处理逻辑
    if time.ticks_diff(current_millis, print_timepoint) > 200:
        print_timepoint = current_millis
        
        # 处理传感器1 (GPIO1)
        analog_buffer_temp = analog_buffer[:]
        median_val = get_median_num(analog_buffer_temp, SCOUNT)
        average_voltage = median_val * (ADC_REF_VOLTAGE / ADC_RESOLUTION)
        compensation_coefficient = 1.0 + 0.02 * (temperature - 25.0)
        compensation_voltage = average_voltage / compensation_coefficient
        tds_value = (133.42 * pow(compensation_voltage, 3) - 255.86 * pow(compensation_voltage, 2) + 857.39 * compensation_voltage) * 0.5

        # 处理传感器2 (GPIO2)
        analog_buffer_temp = analog_buffer2[:]
        median_val = get_median_num(analog_buffer_temp, SCOUNT)
        average_voltage2 = median_val * (ADC_REF_VOLTAGE / ADC_RESOLUTION)
        compensation_voltage = average_voltage2 / compensation_coefficient
        tds_value2 = (133.42 * pow(compensation_voltage, 3) - 255.86 * pow(compensation_voltage, 2) + 857.39 * compensation_voltage) * 0.5

        # 处理传感器3 (GPIO3)
        analog_buffer_temp = analog_buffer3[:]
        median_val = get_median_num(analog_buffer_temp, SCOUNT)
        average_voltage3 = median_val * (ADC_REF_VOLTAGE / ADC_RESOLUTION)
        compensation_voltage = average_voltage3 / compensation_coefficient
        tds_value3 = (133.42 * pow(compensation_voltage, 3) - 255.86 * pow(compensation_voltage, 2) + 857.39 * compensation_voltage) * 0.5

        # 处理传感器4 (GPIO4)
        analog_buffer_temp = analog_buffer4[:]
        median_val = get_median_num(analog_buffer_temp, SCOUNT)
        average_voltage4 = median_val * (ADC_REF_VOLTAGE / ADC_RESOLUTION)
        compensation_voltage = average_voltage4 / compensation_coefficient
        tds_value4 = (133.42 * pow(compensation_voltage, 3) - 255.86 * pow(compensation_voltage, 2) + 857.39 * compensation_voltage) * 0.5

        # 更新全局数据变量（仅当数据有效时）
        if all(math.isfinite(x) for x in [tds_value, tds_value2, tds_value3, tds_value4]):
            new_data = "#{:.2f},{:.2f},{:.2f},{:.2f}$".format(
                max(0, tds_value), 
                max(0, tds_value2), 
                max(0, tds_value3), 
                max(0, tds_value4)
            )
            latest_data = new_data
            
            # 串口输出
            output = latest_data + "\n"
            uart.write(output)
            print(output.strip())  # 同时输出到REPL用于调试

    # 通过UDP发送数据
    if time.ticks_diff(current_millis, last_udp_send) > UDP_SEND_INTERVAL:
        last_udp_send = current_millis
        if wlan.isconnected():
            send_udp_data()

    # 串口控制逻辑 - 增强错误处理
    if uart.any() > 0:
        try:
            # 读取所有可用数据但只处理第一个字节
            data = uart.read()
            if data:
                # 只处理第一个字节
                command_byte = data[0]
                try:
                    # 尝试解码为ASCII字符
                    command = chr(command_byte)
                    if command == '1':
                        pin2_active = True
                        control_pin2.value(0)  # LOW
                        control_pin4.value(1)  # HIGH
                        control_pin7.value(0)  # LOW
                        pin4_active = False
                        trigger_time = time.ticks_ms()
                        uart.write("pump 1\n")
                        print("pump 1")
                    elif command == '0':
                        pin4_active = True
                        control_pin4.value(0)  # LOW
                        control_pin2.value(1)  # HIGH
                        control_pin7.value(0)  # LOW
                        pin2_active = False
                        trigger_time = time.ticks_ms()
                        uart.write("pump 0\n")
                        print("pump 0")
                    else:
                        # 忽略其他字符
                        print(f"Ignored unknown command: 0x{command_byte:02X} ({command})")
                except:
                    # 处理无法解码为字符的情况
                    print(f"Received non-ASCII byte: 0x{command_byte:02X}")
        except Exception as e:
            print("Error processing UART command:", e)

    # 状态恢复逻辑
    if (pin2_active or pin4_active) and time.ticks_diff(time.ticks_ms(), trigger_time) >= 3000:
        if pin2_active:
            control_pin2.value(1)  # HIGH
            pin2_active = False
        if pin4_active:
            control_pin4.value(1)  # HIGH
            pin4_active = False
        control_pin7.value(0)  # LOW
    
    # 定期检查WiFi连接
    if time.ticks_diff(current_millis, last_wifi_check) > WIFI_CHECK_INTERVAL:
        last_wifi_check = current_millis
        
        if not wlan.isconnected():
            print("WiFi disconnected, attempting to reconnect...")
            connect_wifi()
    
    # 添加短暂延时防止忙等待
    time.sleep_ms(1)
