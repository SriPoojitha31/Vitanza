#!/usr/bin/env python3
"""
Test script for hardware communication endpoints.
This script tests all three communication methods.
"""

import asyncio
import json
import time
import requests
import websocket
import paho.mqtt.client as mqtt
from datetime import datetime

# Configuration
SERVER_IP = "localhost"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_IP}:{SERVER_PORT}/api/hardware"
WS_URL = f"ws://{SERVER_IP}:{SERVER_PORT}/api/hardware/ws"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883

# Test device configuration
DEVICE_ID = "test_device_001"
LOCATION = "Test Village"

def test_http_endpoints():
    """Test HTTP endpoints for sensor data and alerts."""
    print("Testing HTTP endpoints...")
    
    # Test single sensor data
    sensor_data = {
        "device_id": DEVICE_ID,
        "sensor_type": "temperature",
        "value": 25.5,
        "unit": "°C",
        "location": LOCATION,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/sensor-data", json=sensor_data)
        print(f"Single sensor data: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Single sensor data error: {e}")
    
    # Test multi-sensor data
    multi_sensor_data = {
        "device_id": DEVICE_ID,
        "readings": [
            {
                "device_id": DEVICE_ID,
                "sensor_type": "temperature",
                "value": 25.5,
                "unit": "°C",
                "location": LOCATION,
                "timestamp": datetime.utcnow().isoformat()
            },
            {
                "device_id": DEVICE_ID,
                "sensor_type": "humidity",
                "value": 60.0,
                "unit": "%",
                "location": LOCATION,
                "timestamp": datetime.utcnow().isoformat()
            }
        ],
        "location": LOCATION,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/multi-sensor-data", json=multi_sensor_data)
        print(f"Multi-sensor data: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Multi-sensor data error: {e}")
    
    # Test alert
    alert_data = {
        "device_id": DEVICE_ID,
        "alert_type": "threshold_exceeded",
        "severity": "high",
        "message": "Temperature exceeds safe limits",
        "location": LOCATION,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/alert", json=alert_data)
        print(f"Alert data: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Alert data error: {e}")
    
    # Test device data retrieval
    try:
        response = requests.get(f"{BASE_URL}/sensor-data/{DEVICE_ID}")
        print(f"Device data retrieval: {response.status_code} - {len(response.json().get('data', []))} records")
    except Exception as e:
        print(f"Device data retrieval error: {e}")
    
    # Test health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check error: {e}")

def test_websocket():
    """Test WebSocket communication."""
    print("\nTesting WebSocket communication...")
    
    def on_message(ws, message):
        print(f"WebSocket received: {message}")
    
    def on_error(ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print("WebSocket closed")
    
    def on_open(ws):
        print("WebSocket connected")
        
        # Send sensor data
        sensor_message = {
            "type": "sensor_data",
            "data": {
                "device_id": DEVICE_ID,
                "sensor_type": "temperature",
                "value": 25.5,
                "unit": "°C",
                "location": LOCATION,
                "timestamp": datetime.utcnow().isoformat()
            },
            "device_id": DEVICE_ID,
            "timestamp": datetime.utcnow().isoformat()
        }
        ws.send(json.dumps(sensor_message))
        print("Sent sensor data via WebSocket")
        
        # Send alert
        alert_message = {
            "type": "alert",
            "data": {
                "device_id": DEVICE_ID,
                "alert_type": "test_alert",
                "severity": "medium",
                "message": "Test alert from WebSocket",
                "location": LOCATION,
                "timestamp": datetime.utcnow().isoformat()
            },
            "device_id": DEVICE_ID,
            "timestamp": datetime.utcnow().isoformat()
        }
        ws.send(json.dumps(alert_message))
        print("Sent alert via WebSocket")
        
        time.sleep(2)
        ws.close()
    
    try:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except Exception as e:
        print(f"WebSocket test error: {e}")

def test_mqtt():
    """Test MQTT communication."""
    print("\nTesting MQTT communication...")
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("MQTT connected successfully")
            # Subscribe to test topics
            client.subscribe(f"vitanza/devices/{DEVICE_ID}/commands")
            client.subscribe(f"test/responses")
        else:
            print(f"MQTT connection failed with code {rc}")
    
    def on_message(client, userdata, msg):
        print(f"MQTT received on {msg.topic}: {msg.payload.decode()}")
    
    def on_disconnect(client, userdata, rc):
        print("MQTT disconnected")
    
    try:
        client = mqtt.Client(f"test_client_{DEVICE_ID}")
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect
        
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        # Publish sensor data
        sensor_topic = f"sensors/{DEVICE_ID}/temperature"
        sensor_data = {
            "device_id": DEVICE_ID,
            "sensor_type": "temperature",
            "value": 25.5,
            "unit": "°C",
            "location": LOCATION,
            "timestamp": datetime.utcnow().isoformat()
        }
        client.publish(sensor_topic, json.dumps(sensor_data))
        print(f"Published sensor data to {sensor_topic}")
        
        # Publish alert
        alert_topic = f"alerts/{DEVICE_ID}"
        alert_data = {
            "device_id": DEVICE_ID,
            "alert_type": "test_alert",
            "severity": "medium",
            "message": "Test alert from MQTT",
            "location": LOCATION,
            "timestamp": datetime.utcnow().isoformat()
        }
        client.publish(alert_topic, json.dumps(alert_data))
        print(f"Published alert to {alert_topic}")
        
        # Publish device status
        status_topic = f"devices/{DEVICE_ID}"
        status_data = {
            "device_id": DEVICE_ID,
            "status": "online",
            "last_seen": datetime.utcnow().isoformat(),
            "battery_level": 85,
            "signal_strength": -45,
            "firmware_version": "1.0.0",
            "location": LOCATION
        }
        client.publish(status_topic, json.dumps(status_data))
        print(f"Published device status to {status_topic}")
        
        # Wait for messages
        time.sleep(3)
        
        client.loop_stop()
        client.disconnect()
        
    except Exception as e:
        print(f"MQTT test error: {e}")

def test_device_management():
    """Test device management endpoints."""
    print("\nTesting device management...")
    
    # Test device status
    try:
        response = requests.get(f"{BASE_URL}/devices/status")
        print(f"All device status: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Device status error: {e}")
    
    # Test specific device status
    try:
        response = requests.get(f"{BASE_URL}/devices/{DEVICE_ID}/status")
        print(f"Device {DEVICE_ID} status: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Specific device status error: {e}")
    
    # Test sending command to device
    command_data = {
        "device_id": DEVICE_ID,
        "command": "calibrate_sensor",
        "parameters": {
            "sensor_name": "temperature",
            "calibration": {"offset": 0, "scale": 1.0}
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/devices/{DEVICE_ID}/command", json=command_data)
        print(f"Device command: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Device command error: {e}")
    
    # Test connections info
    try:
        response = requests.get(f"{BASE_URL}/connections")
        print(f"Connections info: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Connections info error: {e}")

def main():
    """Run all tests."""
    print("Vitanza Hardware Communication Test Suite")
    print("=" * 50)
    
    # Test HTTP endpoints
    test_http_endpoints()
    
    # Test WebSocket
    test_websocket()
    
    # Test MQTT
    test_mqtt()
    
    # Test device management
    test_device_management()
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()
