"""
ESP32 Hardware Communication Examples
====================================

This file contains example code for ESP32 devices to communicate with the Vitanza backend
using the three different methods: HTTP, WebSocket, and MQTT.

Hardware Requirements:
- ESP32 development board
- Sensors (temperature, humidity, water quality, etc.)
- WiFi connection

Installation:
1. Install MicroPython on ESP32
2. Install required libraries: urequests, websocket, umqtt.simple
3. Upload this code to your ESP32

Configuration:
- Update WIFI_SSID and WIFI_PASSWORD
- Update SERVER_IP to your FastAPI server IP
- Update MQTT_BROKER if using MQTT
"""

import network
import time
import json
import urequests
import websocket
import umqtt.simple as mqtt
from machine import Pin, ADC, I2C
import ujson

# Configuration
WIFI_SSID = "your_wifi_ssid"
WIFI_PASSWORD = "your_wifi_password"
SERVER_IP = "192.168.1.100"  # Your FastAPI server IP
SERVER_PORT = 8000
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883

# Device configuration
DEVICE_ID = "esp32_001"
DEVICE_TYPE = "water_quality_sensor"
LOCATION = "Riverside Village"

# Sensor pins (adjust based on your hardware)
TEMP_SENSOR_PIN = 34
HUMIDITY_SENSOR_PIN = 35
PH_SENSOR_PIN = 32
TURBIDITY_SENSOR_PIN = 33

class VitanzaHardwareClient:
    """Main class for ESP32 hardware communication with Vitanza backend."""
    
    def __init__(self):
        self.wifi = network.WLAN(network.STA_IF)
        self.wifi.active(True)
        self.server_url = f"http://{SERVER_IP}:{SERVER_PORT}/api/hardware"
        self.ws_url = f"ws://{SERVER_IP}:{SERVER_PORT}/api/hardware/ws/device/{DEVICE_ID}"
        self.mqtt_client = None
        self.sensors = {}
        self.setup_sensors()
        
    def setup_sensors(self):
        """Initialize sensor pins and calibration."""
        self.sensors = {
            'temperature': ADC(Pin(TEMP_SENSOR_PIN)),
            'humidity': ADC(Pin(HUMIDITY_SENSOR_PIN)),
            'ph': ADC(Pin(PH_SENSOR_PIN)),
            'turbidity': ADC(Pin(TURBIDITY_SENSOR_PIN))
        }
        
        # Sensor calibration values (adjust based on your sensors)
        self.calibration = {
            'temperature': {'offset': -10, 'scale': 0.1},
            'humidity': {'offset': 0, 'scale': 0.1},
            'ph': {'offset': 0, 'scale': 0.01},
            'turbidity': {'offset': 0, 'scale': 0.1}
        }
    
    def connect_wifi(self):
        """Connect to WiFi network."""
        if not self.wifi.isconnected():
            print(f"Connecting to WiFi: {WIFI_SSID}")
            self.wifi.connect(WIFI_SSID, WIFI_PASSWORD)
            
            # Wait for connection
            timeout = 20
            while not self.wifi.isconnected() and timeout > 0:
                time.sleep(1)
                timeout -= 1
                print(".", end="")
            
            if self.wifi.isconnected():
                print(f"\nWiFi connected! IP: {self.wifi.ifconfig()[0]}")
                return True
            else:
                print("\nWiFi connection failed!")
                return False
        return True
    
    def read_sensor(self, sensor_name):
        """Read value from a specific sensor."""
        if sensor_name not in self.sensors:
            return None
            
        # Read raw ADC value
        raw_value = self.sensors[sensor_name].read()
        
        # Apply calibration
        cal = self.calibration[sensor_name]
        calibrated_value = (raw_value * cal['scale']) + cal['offset']
        
        return round(calibrated_value, 2)
    
    def read_all_sensors(self):
        """Read all sensor values."""
        readings = {}
        for sensor_name in self.sensors:
            value = self.read_sensor(sensor_name)
            if value is not None:
                readings[sensor_name] = value
        return readings
    
    # ============================================================================
    # METHOD 1: HTTP REQUESTS
    # ============================================================================
    
    def send_sensor_data_http(self, sensor_data):
        """Send sensor data via HTTP POST request."""
        try:
            url = f"{self.server_url}/sensor-data"
            headers = {"Content-Type": "application/json"}
            
            response = urequests.post(url, json=sensor_data, headers=headers)
            
            if response.status_code == 200:
                print(f"HTTP: Sensor data sent successfully")
                return True
            else:
                print(f"HTTP: Failed to send data. Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"HTTP: Error sending sensor data: {e}")
            return False
        finally:
            if 'response' in locals():
                response.close()
    
    def send_multi_sensor_data_http(self, readings):
        """Send multiple sensor readings via HTTP."""
        try:
            url = f"{self.server_url}/multi-sensor-data"
            headers = {"Content-Type": "application/json"}
            
            # Prepare multi-sensor data
            sensor_readings = []
            for sensor_type, value in readings.items():
                sensor_data = {
                    "device_id": DEVICE_ID,
                    "sensor_type": sensor_type,
                    "value": value,
                    "unit": self.get_sensor_unit(sensor_type),
                    "location": LOCATION,
                    "timestamp": time.time()
                }
                sensor_readings.append(sensor_data)
            
            multi_data = {
                "device_id": DEVICE_ID,
                "readings": sensor_readings,
                "location": LOCATION,
                "timestamp": time.time()
            }
            
            response = urequests.post(url, json=multi_data, headers=headers)
            
            if response.status_code == 200:
                print(f"HTTP: Multi-sensor data sent successfully")
                return True
            else:
                print(f"HTTP: Failed to send multi-sensor data. Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"HTTP: Error sending multi-sensor data: {e}")
            return False
        finally:
            if 'response' in locals():
                response.close()
    
    def send_alert_http(self, alert_type, message, severity="medium"):
        """Send alert via HTTP."""
        try:
            url = f"{self.server_url}/alert"
            headers = {"Content-Type": "application/json"}
            
            alert_data = {
                "device_id": DEVICE_ID,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "location": LOCATION,
                "timestamp": time.time()
            }
            
            response = urequests.post(url, json=alert_data, headers=headers)
            
            if response.status_code == 200:
                print(f"HTTP: Alert sent successfully")
                return True
            else:
                print(f"HTTP: Failed to send alert. Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"HTTP: Error sending alert: {e}")
            return False
        finally:
            if 'response' in locals():
                response.close()
    
    # ============================================================================
    # METHOD 2: WEBSOCKET COMMUNICATION
    # ============================================================================
    
    def connect_websocket(self):
        """Connect to WebSocket for real-time communication."""
        try:
            print(f"Connecting to WebSocket: {self.ws_url}")
            self.ws = websocket.WebSocket()
            self.ws.connect(self.ws_url)
            print("WebSocket connected successfully")
            return True
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    def send_sensor_data_websocket(self, sensor_data):
        """Send sensor data via WebSocket."""
        try:
            message = {
                "type": "sensor_data",
                "data": sensor_data,
                "device_id": DEVICE_ID,
                "timestamp": time.time()
            }
            
            self.ws.send(ujson.dumps(message))
            print("WebSocket: Sensor data sent")
            return True
            
        except Exception as e:
            print(f"WebSocket: Error sending sensor data: {e}")
            return False
    
    def send_alert_websocket(self, alert_type, message, severity="medium"):
        """Send alert via WebSocket."""
        try:
            alert_data = {
                "device_id": DEVICE_ID,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "location": LOCATION,
                "timestamp": time.time()
            }
            
            message = {
                "type": "alert",
                "data": alert_data,
                "device_id": DEVICE_ID,
                "timestamp": time.time()
            }
            
            self.ws.send(ujson.dumps(message))
            print("WebSocket: Alert sent")
            return True
            
        except Exception as e:
            print(f"WebSocket: Error sending alert: {e}")
            return False
    
    def listen_websocket_commands(self):
        """Listen for commands from the server via WebSocket."""
        try:
            if self.ws.recv():
                data = self.ws.recv()
                message = ujson.loads(data)
                
                if message.get("type") == "command":
                    command = message.get("data", {})
                    print(f"WebSocket: Received command: {command}")
                    self.handle_command(command)
                    
        except Exception as e:
            print(f"WebSocket: Error receiving command: {e}")
    
    def handle_command(self, command):
        """Handle commands received from the server."""
        cmd = command.get("command")
        params = command.get("parameters", {})
        
        if cmd == "calibrate_sensor":
            sensor_name = params.get("sensor_name")
            if sensor_name in self.calibration:
                self.calibration[sensor_name] = params.get("calibration", self.calibration[sensor_name])
                print(f"Calibrated sensor: {sensor_name}")
        
        elif cmd == "set_reading_interval":
            interval = params.get("interval", 60)
            print(f"Reading interval set to: {interval} seconds")
        
        elif cmd == "emergency_shutdown":
            print("Emergency shutdown command received!")
            # Implement emergency shutdown logic here
    
    # ============================================================================
    # METHOD 3: MQTT COMMUNICATION
    # ============================================================================
    
    def connect_mqtt(self):
        """Connect to MQTT broker."""
        try:
            self.mqtt_client = mqtt.MQTTClient(
                client_id=f"vitanza_{DEVICE_ID}",
                server=MQTT_BROKER,
                port=MQTT_PORT
            )
            
            self.mqtt_client.set_callback(self.mqtt_callback)
            self.mqtt_client.connect()
            
            # Subscribe to device-specific topics
            self.mqtt_client.subscribe(f"vitanza/devices/{DEVICE_ID}/commands")
            self.mqtt_client.subscribe(f"vitanza/community/{LOCATION}/announcements")
            
            print("MQTT connected successfully")
            return True
            
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            return False
    
    def mqtt_callback(self, topic, message):
        """Handle incoming MQTT messages."""
        try:
            topic_str = topic.decode('utf-8')
            message_str = message.decode('utf-8')
            data = ujson.loads(message_str)
            
            print(f"MQTT: Received message on {topic_str}: {data}")
            
            if "commands" in topic_str:
                self.handle_command(data)
            elif "announcements" in topic_str:
                print(f"Community announcement: {data.get('message', '')}")
                
        except Exception as e:
            print(f"MQTT: Error processing message: {e}")
    
    def send_sensor_data_mqtt(self, sensor_data):
        """Send sensor data via MQTT."""
        try:
            topic = f"sensors/{DEVICE_ID}/{sensor_data['sensor_type']}"
            message = ujson.dumps(sensor_data)
            
            self.mqtt_client.publish(topic, message)
            print(f"MQTT: Sensor data published to {topic}")
            return True
            
        except Exception as e:
            print(f"MQTT: Error publishing sensor data: {e}")
            return False
    
    def send_alert_mqtt(self, alert_type, message, severity="medium"):
        """Send alert via MQTT."""
        try:
            topic = f"alerts/{DEVICE_ID}"
            alert_data = {
                "device_id": DEVICE_ID,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "location": LOCATION,
                "timestamp": time.time()
            }
            
            self.mqtt_client.publish(topic, ujson.dumps(alert_data))
            print(f"MQTT: Alert published to {topic}")
            return True
            
        except Exception as e:
            print(f"MQTT: Error publishing alert: {e}")
            return False
    
    def send_device_status_mqtt(self):
        """Send device status via MQTT."""
        try:
            topic = f"devices/{DEVICE_ID}"
            status_data = {
                "device_id": DEVICE_ID,
                "status": "online",
                "last_seen": time.time(),
                "battery_level": 85,  # Simulate battery level
                "signal_strength": -45,  # Simulate signal strength
                "firmware_version": "1.0.0",
                "location": LOCATION
            }
            
            self.mqtt_client.publish(topic, ujson.dumps(status_data))
            print(f"MQTT: Device status published to {topic}")
            return True
            
        except Exception as e:
            print(f"MQTT: Error publishing device status: {e}")
            return False
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_sensor_unit(self, sensor_type):
        """Get the unit for a sensor type."""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'ph': 'pH',
            'turbidity': 'NTU',
            'chlorine': 'ppm'
        }
        return units.get(sensor_type, '')
    
    def check_thresholds(self, readings):
        """Check if any readings exceed thresholds."""
        thresholds = {
            'temperature': {'min': -10, 'max': 50},
            'humidity': {'min': 0, 'max': 100},
            'ph': {'min': 6.5, 'max': 8.5},
            'turbidity': {'min': 0, 'max': 5.0}
        }
        
        alerts = []
        for sensor_type, value in readings.items():
            if sensor_type in thresholds:
                threshold = thresholds[sensor_type]
                if value < threshold['min'] or value > threshold['max']:
                    alerts.append({
                        'sensor': sensor_type,
                        'value': value,
                        'threshold': threshold
                    })
        
        return alerts
    
    # ============================================================================
    # MAIN OPERATION LOOP
    # ============================================================================
    
    def run_http_mode(self, interval=60):
        """Run in HTTP mode - send data every interval seconds."""
        print("Starting HTTP mode...")
        
        while True:
            try:
                # Read all sensors
                readings = self.read_all_sensors()
                print(f"Sensor readings: {readings}")
                
                # Check for threshold alerts
                alerts = self.check_thresholds(readings)
                if alerts:
                    for alert in alerts:
                        message = f"{alert['sensor']} reading {alert['value']} exceeds threshold"
                        self.send_alert_http("threshold_exceeded", message, "high")
                
                # Send sensor data
                for sensor_type, value in readings.items():
                    sensor_data = {
                        "device_id": DEVICE_ID,
                        "sensor_type": sensor_type,
                        "value": value,
                        "unit": self.get_sensor_unit(sensor_type),
                        "location": LOCATION,
                        "timestamp": time.time()
                    }
                    self.send_sensor_data_http(sensor_data)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in HTTP mode: {e}")
                time.sleep(10)
    
    def run_websocket_mode(self, interval=60):
        """Run in WebSocket mode - maintain persistent connection."""
        print("Starting WebSocket mode...")
        
        if not self.connect_websocket():
            return
        
        while True:
            try:
                # Read and send sensor data
                readings = self.read_all_sensors()
                for sensor_type, value in readings.items():
                    sensor_data = {
                        "device_id": DEVICE_ID,
                        "sensor_type": sensor_type,
                        "value": value,
                        "unit": self.get_sensor_unit(sensor_type),
                        "location": LOCATION,
                        "timestamp": time.time()
                    }
                    self.send_sensor_data_websocket(sensor_data)
                
                # Listen for commands
                self.listen_websocket_commands()
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in WebSocket mode: {e}")
                # Try to reconnect
                if not self.connect_websocket():
                    time.sleep(10)
    
    def run_mqtt_mode(self, interval=60):
        """Run in MQTT mode - publish data and listen for commands."""
        print("Starting MQTT mode...")
        
        if not self.connect_mqtt():
            return
        
        while True:
            try:
                # Read and send sensor data
                readings = self.read_all_sensors()
                for sensor_type, value in readings.items():
                    sensor_data = {
                        "device_id": DEVICE_ID,
                        "sensor_type": sensor_type,
                        "value": value,
                        "unit": self.get_sensor_unit(sensor_type),
                        "location": LOCATION,
                        "timestamp": time.time()
                    }
                    self.send_sensor_data_mqtt(sensor_data)
                
                # Send device status
                self.send_device_status_mqtt()
                
                # Check for messages
                self.mqtt_client.check_msg()
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in MQTT mode: {e}")
                time.sleep(10)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Main function - choose your communication method."""
    
    # Initialize hardware client
    client = VitanzaHardwareClient()
    
    # Connect to WiFi
    if not client.connect_wifi():
        print("Failed to connect to WiFi. Exiting.")
        return
    
    # Choose communication method
    # Uncomment one of the following:
    
    # Method 1: HTTP Requests (simplest)
    # client.run_http_mode(interval=60)  # Send data every 60 seconds
    
    # Method 2: WebSocket (real-time)
    # client.run_websocket_mode(interval=30)  # Send data every 30 seconds
    
    # Method 3: MQTT (IoT standard)
    client.run_mqtt_mode(interval=60)  # Send data every 60 seconds

if __name__ == "__main__":
    main()
