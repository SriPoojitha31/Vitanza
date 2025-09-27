import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import paho.mqtt.client as mqtt
from paho.mqtt.client import MQTTMessage

from schemas.hardware import SensorData, MultiSensorData, AlertData, DeviceStatus
from mongo import get_database

logger = logging.getLogger(__name__)

class MQTTSubscriber:
    """
    MQTT subscriber for receiving IoT device data.
    Handles connection to MQTT broker and message processing.
    """
    
    def __init__(self, broker_host: str = "broker.hivemq.com", broker_port: int = 1883, 
                 client_id: str = "vitanza_backend", username: Optional[str] = None, 
                 password: Optional[str] = None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        self.is_connected = False
        self.subscribed_topics = set()
        
        # Callbacks for different message types
        self.sensor_data_callback: Optional[Callable] = None
        self.alert_callback: Optional[Callable] = None
        self.device_status_callback: Optional[Callable] = None
        
    def on_connect(self, client, userdata, flags, rc):
        """Called when the client connects to the broker."""
        if rc == 0:
            self.is_connected = True
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            # Subscribe to default topics
            self.subscribe_to_topics()
        else:
            logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
            self.is_connected = False
    
    def on_disconnect(self, client, userdata, rc):
        """Called when the client disconnects from the broker."""
        self.is_connected = False
        logger.warning(f"Disconnected from MQTT broker. Return code: {rc}")
    
    def on_message(self, client, userdata, msg: MQTTMessage):
        """Called when a message is received from the broker."""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logger.debug(f"Received message on topic '{topic}': {payload}")
            
            # Parse JSON payload
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON payload: {e}")
                return
            
            # Route message based on topic
            asyncio.create_task(self.process_message(topic, data))
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    async def process_message(self, topic: str, data: Dict[str, Any]):
        """Process incoming MQTT messages based on topic."""
        try:
            if topic.startswith("sensors/"):
                await self.handle_sensor_data(topic, data)
            elif topic.startswith("alerts/"):
                await self.handle_alert_data(topic, data)
            elif topic.startswith("devices/"):
                await self.handle_device_data(topic, data)
            elif topic.startswith("community/"):
                await self.handle_community_data(topic, data)
            else:
                logger.warning(f"Unknown topic pattern: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
    
    async def handle_sensor_data(self, topic: str, data: Dict[str, Any]):
        """Handle sensor data messages."""
        try:
            # Extract device ID from topic (e.g., "sensors/device_001/temperature")
            topic_parts = topic.split('/')
            if len(topic_parts) >= 3:
                device_id = topic_parts[1]
                sensor_type = topic_parts[2]
                
                # Create SensorData object
                sensor_data = SensorData(
                    device_id=device_id,
                    sensor_type=sensor_type,
                    value=float(data.get('value', 0)),
                    unit=data.get('unit', ''),
                    location=data.get('location'),
                    latitude=data.get('latitude'),
                    longitude=data.get('longitude'),
                    timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat())),
                    battery_level=data.get('battery_level'),
                    signal_strength=data.get('signal_strength'),
                    metadata=data.get('metadata')
                )
                
                # Store in database
                await self.store_sensor_data(sensor_data)
                
                # Call callback if set
                if self.sensor_data_callback:
                    await self.sensor_data_callback(sensor_data)
                
                logger.info(f"Processed sensor data from {device_id}: {sensor_type} = {sensor_data.value}")
                
        except Exception as e:
            logger.error(f"Error handling sensor data: {e}")
    
    async def handle_alert_data(self, topic: str, data: Dict[str, Any]):
        """Handle alert messages."""
        try:
            topic_parts = topic.split('/')
            if len(topic_parts) >= 2:
                device_id = topic_parts[1]
                
                alert_data = AlertData(
                    device_id=device_id,
                    alert_type=data.get('alert_type', 'unknown'),
                    severity=data.get('severity', 'medium'),
                    message=data.get('message', ''),
                    location=data.get('location'),
                    timestamp=datetime.fromisoformat(data.get('timestamp', datetime.utcnow().isoformat()))
                )
                
                # Store in database
                await self.store_alert_data(alert_data)
                
                # Call callback if set
                if self.alert_callback:
                    await self.alert_callback(alert_data)
                
                logger.warning(f"Processed alert from {device_id}: {alert_data.alert_type}")
                
        except Exception as e:
            logger.error(f"Error handling alert data: {e}")
    
    async def handle_device_data(self, topic: str, data: Dict[str, Any]):
        """Handle device status messages."""
        try:
            topic_parts = topic.split('/')
            if len(topic_parts) >= 2:
                device_id = topic_parts[1]
                
                device_status = DeviceStatus(
                    device_id=device_id,
                    status=data.get('status', 'unknown'),
                    last_seen=datetime.fromisoformat(data.get('last_seen', datetime.utcnow().isoformat())),
                    battery_level=data.get('battery_level'),
                    signal_strength=data.get('signal_strength'),
                    firmware_version=data.get('firmware_version'),
                    location=data.get('location')
                )
                
                # Store in database
                await self.store_device_status(device_status)
                
                # Call callback if set
                if self.device_status_callback:
                    await self.device_status_callback(device_status)
                
                logger.info(f"Processed device status from {device_id}: {device_status.status}")
                
        except Exception as e:
            logger.error(f"Error handling device data: {e}")
    
    async def handle_community_data(self, topic: str, data: Dict[str, Any]):
        """Handle community-specific data."""
        try:
            # This could be community alerts, announcements, etc.
            logger.info(f"Processed community data from topic {topic}")
            
            # Store community data
            db = await get_database()
            collection = db["community_data"]
            
            community_data = {
                "topic": topic,
                "data": data,
                "received_at": datetime.utcnow(),
                "source": "mqtt"
            }
            
            await collection.insert_one(community_data)
            
        except Exception as e:
            logger.error(f"Error handling community data: {e}")
    
    async def store_sensor_data(self, sensor_data: SensorData):
        """Store sensor data in database."""
        try:
            db = await get_database()
            collection = db["sensor_readings"]
            
            sensor_dict = sensor_data.dict()
            sensor_dict["source"] = "mqtt"
            sensor_dict["received_at"] = datetime.utcnow()
            
            await collection.insert_one(sensor_dict)
            
        except Exception as e:
            logger.error(f"Error storing sensor data: {e}")
    
    async def store_alert_data(self, alert_data: AlertData):
        """Store alert data in database."""
        try:
            db = await get_database()
            collection = db["alerts"]
            
            alert_dict = alert_data.dict()
            alert_dict["source"] = "mqtt"
            alert_dict["received_at"] = datetime.utcnow()
            
            await collection.insert_one(alert_dict)
            
        except Exception as e:
            logger.error(f"Error storing alert data: {e}")
    
    async def store_device_status(self, device_status: DeviceStatus):
        """Store device status in database."""
        try:
            db = await get_database()
            collection = db["device_status"]
            
            status_dict = device_status.dict()
            status_dict["source"] = "mqtt"
            status_dict["updated_at"] = datetime.utcnow()
            
            await collection.insert_one(status_dict)
            
        except Exception as e:
            logger.error(f"Error storing device status: {e}")
    
    def subscribe_to_topics(self):
        """Subscribe to MQTT topics."""
        topics = [
            "sensors/+/+",      # sensors/device_id/sensor_type
            "alerts/+",         # alerts/device_id
            "devices/+",        # devices/device_id
            "community/+",      # community/community_id
            "vitanza/+/+",      # vitanza/device_type/device_id
        ]
        
        for topic in topics:
            result = self.client.subscribe(topic)
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.subscribed_topics.add(topic)
                logger.info(f"Subscribed to topic: {topic}")
            else:
                logger.error(f"Failed to subscribe to topic: {topic}")
    
    def subscribe_to_topic(self, topic: str):
        """Subscribe to a specific topic."""
        result = self.client.subscribe(topic)
        if result[0] == mqtt.MQTT_ERR_SUCCESS:
            self.subscribed_topics.add(topic)
            logger.info(f"Subscribed to topic: {topic}")
            return True
        else:
            logger.error(f"Failed to subscribe to topic: {topic}")
            return False
    
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from a specific topic."""
        result = self.client.unsubscribe(topic)
        if result[0] == mqtt.MQTT_ERR_SUCCESS:
            self.subscribed_topics.discard(topic)
            logger.info(f"Unsubscribed from topic: {topic}")
            return True
        else:
            logger.error(f"Failed to unsubscribe from topic: {topic}")
            return False
    
    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            logger.info(f"Attempting to connect to MQTT broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    def start_loop(self):
        """Start the MQTT client loop."""
        try:
            self.client.loop_start()
            logger.info("Started MQTT client loop")
        except Exception as e:
            logger.error(f"Failed to start MQTT client loop: {e}")
            raise
    
    def stop_loop(self):
        """Stop the MQTT client loop."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("Stopped MQTT client loop")
        except Exception as e:
            logger.error(f"Error stopping MQTT client loop: {e}")
    
    def publish_message(self, topic: str, message: Dict[str, Any], qos: int = 0):
        """Publish a message to a topic."""
        try:
            payload = json.dumps(message)
            result = self.client.publish(topic, payload, qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.debug(f"Published message to topic {topic}")
                return True
            else:
                logger.error(f"Failed to publish message to topic {topic}")
                return False
                
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    def set_sensor_data_callback(self, callback: Callable):
        """Set callback for sensor data processing."""
        self.sensor_data_callback = callback
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for alert processing."""
        self.alert_callback = callback
    
    def set_device_status_callback(self, callback: Callable):
        """Set callback for device status processing."""
        self.device_status_callback = callback

# Global MQTT subscriber instance
mqtt_subscriber: Optional[MQTTSubscriber] = None

async def initialize_mqtt_subscriber():
    """Initialize the global MQTT subscriber."""
    global mqtt_subscriber
    
    try:
        # Configuration from environment variables
        import os
        broker_host = os.getenv("MQTT_BROKER_HOST", "broker.hivemq.com")
        broker_port = int(os.getenv("MQTT_BROKER_PORT", "1883"))
        username = os.getenv("MQTT_USERNAME")
        password = os.getenv("MQTT_PASSWORD")
        
        mqtt_subscriber = MQTTSubscriber(
            broker_host=broker_host,
            broker_port=broker_port,
            username=username,
            password=password
        )
        
        mqtt_subscriber.connect()
        mqtt_subscriber.start_loop()
        
        logger.info("MQTT subscriber initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MQTT subscriber: {e}")
        raise

async def get_mqtt_subscriber() -> MQTTSubscriber:
    """Get the global MQTT subscriber instance."""
    global mqtt_subscriber
    if mqtt_subscriber is None:
        await initialize_mqtt_subscriber()
    return mqtt_subscriber
