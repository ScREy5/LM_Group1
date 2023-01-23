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

class Wheels {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.wheelPosR = null;
      this.wheelPosL = null;
      this.wheelSpeedR = null;
      this.wheelSpeedL = null;
    }
    else {
      if (initObj.hasOwnProperty('wheelPosR')) {
        this.wheelPosR = initObj.wheelPosR
      }
      else {
        this.wheelPosR = new std_msgs.msg.Int64();
      }
      if (initObj.hasOwnProperty('wheelPosL')) {
        this.wheelPosL = initObj.wheelPosL
      }
      else {
        this.wheelPosL = new std_msgs.msg.Int64();
      }
      if (initObj.hasOwnProperty('wheelSpeedR')) {
        this.wheelSpeedR = initObj.wheelSpeedR
      }
      else {
        this.wheelSpeedR = new std_msgs.msg.Int64();
      }
      if (initObj.hasOwnProperty('wheelSpeedL')) {
        this.wheelSpeedL = initObj.wheelSpeedL
      }
      else {
        this.wheelSpeedL = new std_msgs.msg.Int64();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Wheels
    // Serialize message field [wheelPosR]
    bufferOffset = std_msgs.msg.Int64.serialize(obj.wheelPosR, buffer, bufferOffset);
    // Serialize message field [wheelPosL]
    bufferOffset = std_msgs.msg.Int64.serialize(obj.wheelPosL, buffer, bufferOffset);
    // Serialize message field [wheelSpeedR]
    bufferOffset = std_msgs.msg.Int64.serialize(obj.wheelSpeedR, buffer, bufferOffset);
    // Serialize message field [wheelSpeedL]
    bufferOffset = std_msgs.msg.Int64.serialize(obj.wheelSpeedL, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Wheels
    let len;
    let data = new Wheels(null);
    // Deserialize message field [wheelPosR]
    data.wheelPosR = std_msgs.msg.Int64.deserialize(buffer, bufferOffset);
    // Deserialize message field [wheelPosL]
    data.wheelPosL = std_msgs.msg.Int64.deserialize(buffer, bufferOffset);
    // Deserialize message field [wheelSpeedR]
    data.wheelSpeedR = std_msgs.msg.Int64.deserialize(buffer, bufferOffset);
    // Deserialize message field [wheelSpeedL]
    data.wheelSpeedL = std_msgs.msg.Int64.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 32;
  }

  static datatype() {
    // Returns string type for a message object
    return 'robobo_msgs/Wheels';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e3007c5c2ec7faa3f8f71bd3a0d670c5';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Int64 wheelPosR
    std_msgs/Int64 wheelPosL
    std_msgs/Int64 wheelSpeedR
    std_msgs/Int64 wheelSpeedL
    
    ================================================================================
    MSG: std_msgs/Int64
    int64 data
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Wheels(null);
    if (msg.wheelPosR !== undefined) {
      resolved.wheelPosR = std_msgs.msg.Int64.Resolve(msg.wheelPosR)
    }
    else {
      resolved.wheelPosR = new std_msgs.msg.Int64()
    }

    if (msg.wheelPosL !== undefined) {
      resolved.wheelPosL = std_msgs.msg.Int64.Resolve(msg.wheelPosL)
    }
    else {
      resolved.wheelPosL = new std_msgs.msg.Int64()
    }

    if (msg.wheelSpeedR !== undefined) {
      resolved.wheelSpeedR = std_msgs.msg.Int64.Resolve(msg.wheelSpeedR)
    }
    else {
      resolved.wheelSpeedR = new std_msgs.msg.Int64()
    }

    if (msg.wheelSpeedL !== undefined) {
      resolved.wheelSpeedL = std_msgs.msg.Int64.Resolve(msg.wheelSpeedL)
    }
    else {
      resolved.wheelSpeedL = new std_msgs.msg.Int64()
    }

    return resolved;
    }
};

module.exports = Wheels;
