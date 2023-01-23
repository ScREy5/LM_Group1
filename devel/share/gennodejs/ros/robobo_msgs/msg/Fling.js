// Auto-generated. Do not edit!

// (in-package robobo_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class Fling {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.angle = null;
      this.time = null;
      this.distance = null;
    }
    else {
      if (initObj.hasOwnProperty('angle')) {
        this.angle = initObj.angle
      }
      else {
        this.angle = new std_msgs.msg.Int16();
      }
      if (initObj.hasOwnProperty('time')) {
        this.time = initObj.time
      }
      else {
        this.time = new std_msgs.msg.Int32();
      }
      if (initObj.hasOwnProperty('distance')) {
        this.distance = initObj.distance
      }
      else {
        this.distance = new std_msgs.msg.Int16();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Fling
    // Serialize message field [angle]
    bufferOffset = std_msgs.msg.Int16.serialize(obj.angle, buffer, bufferOffset);
    // Serialize message field [time]
    bufferOffset = std_msgs.msg.Int32.serialize(obj.time, buffer, bufferOffset);
    // Serialize message field [distance]
    bufferOffset = std_msgs.msg.Int16.serialize(obj.distance, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Fling
    let len;
    let data = new Fling(null);
    // Deserialize message field [angle]
    data.angle = std_msgs.msg.Int16.deserialize(buffer, bufferOffset);
    // Deserialize message field [time]
    data.time = std_msgs.msg.Int32.deserialize(buffer, bufferOffset);
    // Deserialize message field [distance]
    data.distance = std_msgs.msg.Int16.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 8;
  }

  static datatype() {
    // Returns string type for a message object
    return 'robobo_msgs/Fling';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '223606e2db34b4f54252a5c13a50f185';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int16 angle
    std_msgs/Int32 time
    std_msgs/Int16 distance
    
    ================================================================================
    MSG: std_msgs/Int16
    int16 data
    
    ================================================================================
    MSG: std_msgs/Int32
    int32 data
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Fling(null);
    if (msg.angle !== undefined) {
      resolved.angle = std_msgs.msg.Int16.Resolve(msg.angle)
    }
    else {
      resolved.angle = new std_msgs.msg.Int16()
    }

    if (msg.time !== undefined) {
      resolved.time = std_msgs.msg.Int32.Resolve(msg.time)
    }
    else {
      resolved.time = new std_msgs.msg.Int32()
    }

    if (msg.distance !== undefined) {
      resolved.distance = std_msgs.msg.Int16.Resolve(msg.distance)
    }
    else {
      resolved.distance = new std_msgs.msg.Int16()
    }

    return resolved;
    }
};

module.exports = Fling;
