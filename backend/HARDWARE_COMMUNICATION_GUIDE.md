# Vitanza Hardware Communication Guide

This guide explains how to implement hardware communication between IoT devices (like ESP32) and the Vitanza FastAPI backend using three different methods.

## Table of Contents
1. [Overview](#overview)
2. [Method 1: HTTP Requests](#method-1-http-requests)
3. [Method 2: WebSocket Communication](#method-2-websocket-communication)
4. [Method 3: MQTT Communication](#method-3-mqtt-communication)
5. [Hardware Setup](#hardware-setup)
6. [API Endpoints](#api-endpoints)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Overview

The Vitanza backend supports three communication methods for IoT devices:

- **HTTP Requests**: Simple, widely supported, good for basic data transmission
- **WebSocket**: Real-time bidirectional communication, ideal for interactive applications
- **MQTT**: IoT standard protocol, perfect for multiple devices and scalable deployments

## Method 1: HTTP Requests

### Pros
- ✅ Simple to implement
- ✅ Widely supported by all hardware platforms
- ✅ Easy to debug and test
- ✅ Works with any HTTP client

### Cons
- ❌ Not real-time
- ❌ Requires polling for bidirectional communication
- ❌ Higher overhead for frequent updates

### Hardware Implementation (ESP32)

```python
import urequests
import ujson

# Send single sensor reading
def send_sensor_data(device_id, sensor_type, value, unit, location):
    url = "http://YOUR_SERVER_IP:8000/api/hardware/sensor-data"
    data = {
        "device_id": device_id,
        "sensor_type": sensor_type,
        "value": value,
        "unit": unit,
        "location": location,
        "timestamp": time.time()
    }
    
    response = urequests.post(url, json=data)
    return response.status_code == 200

# Send multiple sensor readings
def send_multi_sensor_data(device_id, readings, location):
    url = "http://YOUR_SERVER_IP:8000/api/hardware/multi-sensor-data"
    sensor_readings = []
    
    for sensor_type, value in readings.items():
        sensor_readings.append({
            "device_id": device_id,
            "sensor_type": sensor_type,
            "value": value,
            "unit": get_unit(sensor_type),
            "location": location,
            "timestamp": time.time()
        })
    
    data = {
        "device_id": device_id,
        "readings": sensor_readings,
        "location": location,
        "timestamp": time.time()
    }
    
    response = urequests.post(url, json=data)
    return response.status_code == 200

# Send alert
def send_alert(device_id, alert_type, message, severity="medium"):
    url = "http://YOUR_SERVER_IP:8000/api/hardware/alert"
    data = {
        "device_id": device_id,
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "timestamp": time.time()
    }
    
    response = urequests.post(url, json=data)
    return response.status_code == 200
```

### Backend Endpoints

- `POST /api/hardware/sensor-data` - Receive single sensor reading
- `POST /api/hardware/multi-sensor-data` - Receive multiple sensor readings
- `POST /api/hardware/alert` - Receive alert from device
- `GET /api/hardware/sensor-data/{device_id}` - Retrieve device data

## Method 2: WebSocket Communication

### Pros
- ✅ Real-time bidirectional communication
- ✅ Low latency
- ✅ Persistent connection
- ✅ Can receive commands from server

### Cons
- ❌ More complex than HTTP
- ❌ Requires WebSocket support on hardware
- ❌ Connection management needed

### Hardware Implementation (ESP32)

```python
import websocket
import ujson

class WebSocketClient:
    def __init__(self, server_ip, device_id):
        self.ws_url = f"ws://{server_ip}:8000/api/hardware/ws/device/{device_id}"
        self.ws = None
    
    def connect(self):
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.ws_url)
            print("WebSocket connected")
            return True
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    def send_sensor_data(self, sensor_data):
        message = {
            "type": "sensor_data",
            "data": sensor_data,
            "device_id": DEVICE_ID,
            "timestamp": time.time()
        }
        self.ws.send(ujson.dumps(message))
    
    def send_alert(self, alert_type, message, severity="medium"):
        alert_data = {
            "device_id": DEVICE_ID,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": time.time()
        }
        
        message = {
            "type": "alert",
            "data": alert_data,
            "device_id": DEVICE_ID,
            "timestamp": time.time()
        }
        self.ws.send(ujson.dumps(message))
    
    def listen_for_commands(self):
        try:
            if self.ws.recv():
                data = self.ws.recv()
                message = ujson.loads(data)
                if message.get("type") == "command":
                    self.handle_command(message.get("data", {}))
        except:
            pass
    
    def handle_command(self, command):
        cmd = command.get("command")
        if cmd == "calibrate_sensor":
            # Handle calibration command
            pass
        elif cmd == "set_interval":
            # Handle interval change command
            pass
```

### Backend WebSocket Endpoints

- `WS /api/hardware/ws` - General WebSocket endpoint
- `WS /api/hardware/ws/device/{device_id}` - Device-specific WebSocket endpoint

## Method 3: MQTT Communication

### Pros
- ✅ Designed for IoT
- ✅ Scalable for many devices
- ✅ Lightweight protocol
- ✅ Built-in QoS levels
- ✅ Works with unreliable networks

### Cons
- ❌ Requires MQTT broker
- ❌ More complex setup
- ❌ Additional infrastructure needed

### Hardware Implementation (ESP32)

```python
import umqtt.simple as mqtt

class MQTTClient:
    def __init__(self, broker, device_id):
        self.broker = broker
        self.device_id = device_id
        self.client = mqtt.MQTTClient(
            client_id=f"vitanza_{device_id}",
            server=broker,
            port=1883
        )
        self.client.set_callback(self.on_message)
    
    def connect(self):
        try:
            self.client.connect()
            # Subscribe to device commands
            self.client.subscribe(f"vitanza/devices/{self.device_id}/commands")
            print("MQTT connected")
            return True
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            return False
    
    def on_message(self, topic, message):
        topic_str = topic.decode('utf-8')
        data = ujson.loads(message.decode('utf-8'))
        print(f"Received: {topic_str} -> {data}")
        
        if "commands" in topic_str:
            self.handle_command(data)
    
    def send_sensor_data(self, sensor_type, value, unit):
        topic = f"sensors/{self.device_id}/{sensor_type}"
        data = {
            "device_id": self.device_id,
            "sensor_type": sensor_type,
            "value": value,
            "unit": unit,
            "timestamp": time.time()
        }
        self.client.publish(topic, ujson.dumps(data))
    
    def send_alert(self, alert_type, message, severity="medium"):
        topic = f"alerts/{self.device_id}"
        data = {
            "device_id": self.device_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": time.time()
        }
        self.client.publish(topic, ujson.dumps(data))
    
    def send_device_status(self):
        topic = f"devices/{self.device_id}"
        data = {
            "device_id": self.device_id,
            "status": "online",
            "last_seen": time.time(),
            "battery_level": 85,
            "signal_strength": -45
        }
        self.client.publish(topic, ujson.dumps(data))
    
    def check_messages(self):
        self.client.check_msg()
```

### MQTT Topics

The backend subscribes to these MQTT topics:

- `sensors/+/+` - Sensor data from any device
- `alerts/+` - Alerts from any device  
- `devices/+` - Device status updates
- `community/+` - Community-specific data
- `vitanza/+/+` - Vitanza-specific device data

## Hardware Setup

### ESP32 Requirements

1. **Hardware**:
   - ESP32 development board
   - Sensors (temperature, humidity, water quality, etc.)
   - Power supply
   - WiFi antenna

2. **Software**:
   - MicroPython or Arduino IDE
   - Required libraries:
     - `urequests` (for HTTP)
     - `websocket` (for WebSocket)
     - `umqtt.simple` (for MQTT)

3. **Installation**:
   ```bash
   # Install MicroPython on ESP32
   # Then install required libraries
   import upip
   upip.install('urequests')
   upip.install('websocket')
   upip.install('umqtt.simple')
   ```

### Configuration

Update these variables in your hardware code:

```python
# WiFi Configuration
WIFI_SSID = "your_wifi_network"
WIFI_PASSWORD = "your_wifi_password"

# Server Configuration
SERVER_IP = "192.168.1.100"  # Your FastAPI server IP
SERVER_PORT = 8000

# MQTT Configuration (if using MQTT)
MQTT_BROKER = "broker.hivemq.com"  # or your own MQTT broker
MQTT_PORT = 1883

# Device Configuration
DEVICE_ID = "esp32_001"
LOCATION = "Riverside Village"
```

## API Endpoints

### HTTP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/hardware/sensor-data` | Receive single sensor reading |
| POST | `/api/hardware/multi-sensor-data` | Receive multiple sensor readings |
| POST | `/api/hardware/alert` | Receive alert from device |
| GET | `/api/hardware/sensor-data/{device_id}` | Get device sensor data |
| GET | `/api/hardware/devices/status` | Get all device statuses |
| GET | `/api/hardware/devices/{device_id}/status` | Get specific device status |
| POST | `/api/hardware/devices/{device_id}/command` | Send command to device |
| GET | `/api/hardware/connections` | Get connection information |
| GET | `/api/hardware/health` | Health check |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `WS /api/hardware/ws` | General WebSocket endpoint |
| `WS /api/hardware/ws/device/{device_id}` | Device-specific WebSocket |

### MQTT Topics

| Topic Pattern | Description |
|---------------|-------------|
| `sensors/{device_id}/{sensor_type}` | Sensor data |
| `alerts/{device_id}` | Device alerts |
| `devices/{device_id}` | Device status |
| `community/{community_id}` | Community data |
| `vitanza/{device_type}/{device_id}` | Vitanza-specific data |

## Security Considerations

### Authentication

For production use, implement authentication:

```python
# Add API key to requests
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

# Or use device certificates for MQTT
client = mqtt.MQTTClient(
    client_id=device_id,
    server=broker,
    port=8883,  # Use secure port
    ssl=True,
    ssl_params={
        "certfile": "device.crt",
        "keyfile": "device.key"
    }
)
```

### Data Validation

The backend validates all incoming data using Pydantic schemas:

- Sensor data must match `SensorData` schema
- Alerts must match `AlertData` schema
- Device status must match `DeviceStatus` schema

### Network Security

1. Use HTTPS/WSS for production
2. Implement API rate limiting
3. Use VPN for device connections
4. Regular security updates

## Troubleshooting

### Common Issues

1. **Connection Timeouts**:
   - Check WiFi connection
   - Verify server IP and port
   - Check firewall settings

2. **MQTT Connection Failed**:
   - Verify broker address and port
   - Check broker authentication
   - Ensure broker is running

3. **WebSocket Connection Issues**:
   - Check if WebSocket is supported
   - Verify server is running
   - Check for proxy issues

4. **Data Not Received**:
   - Check JSON format
   - Verify schema compliance
   - Check database connection

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing

Use these tools to test your implementation:

1. **HTTP**: Use Postman or curl
2. **WebSocket**: Use WebSocket testing tools
3. **MQTT**: Use MQTT Explorer or mosquitto_pub/sub

## Example Usage

See `backend/examples/esp32_hardware_examples.py` for complete working examples of all three communication methods.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example code
3. Check server logs for errors
4. Verify network connectivity

---

This guide provides everything needed to implement hardware communication with the Vitanza backend. Choose the method that best fits your use case and hardware capabilities.
